"""
Download DROID dataset from HuggingFace (cadene/droid).

1. Metadata: load_dataset() for frame-level state/action (cached as arrow).
2. Videos: choose method for ~400GB of MP4s:
   - snapshot (default): HuggingFace snapshot_download with many parallel workers (videos only).
   - git-lfs: clone repo with GIT_LFS_SKIP_SMUDGE then git lfs pull (videos only). No Xet required.
   - one-by-one: legacy per-file download (very slow, ~250h for full set).

Usage:
    python download_droid.py [--cache-dir DIR] [--method snapshot|git-lfs|one-by-one] [--video-only]
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

# Video path pattern from DROID meta/info.json:
#   video_path: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DROID_REPO_ID = "cadene/droid"
TOTAL_EPISODES = 92233
CHUNK_SIZE = 1000
VIDEO_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
    "observation.images.wrist_image_left",
)


def video_path_for_episode(episode_index: int, video_key: str) -> str:
    """Path of one MP4 file in the repo. Hub uses full key, e.g. observation.images.exterior_image_1_left."""
    chunk = episode_index // CHUNK_SIZE
    return f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def list_mp4_paths_from_hub():
    """List all .mp4 filenames in the dataset repo (slower but accurate)."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(DROID_REPO_ID, repo_type="dataset")
    return [f for f in files if f.endswith(".mp4")]


def download_videos(
    cache_dir: str,
    resume: bool = True,
    max_episodes: Optional[int] = None,
    list_from_hub: bool = False,
):
    """Download all DROID MP4s into cache_dir, preserving repo directory structure."""
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    local_dir = os.path.join(cache_dir, "droid_videos")
    os.makedirs(local_dir, exist_ok=True)

    # Use constructed paths by default (instant). Hub listing is slow for 270k+ files.
    if list_from_hub:
        try:
            print("Listing MP4 files from Hub (this can take several minutes)...")
            all_mp4 = list_mp4_paths_from_hub()
            paths_to_download = sorted(all_mp4)
            print(f"Listed {len(paths_to_download)} MP4 files.")
        except Exception as e:
            print(f"Listing failed: {e}. Falling back to constructed paths.")
            list_from_hub = False
    if not list_from_hub:
        n = max_episodes if max_episodes is not None else TOTAL_EPISODES
        paths_to_download = []
        for ep in range(n):
            for vk in VIDEO_KEYS:
                paths_to_download.append(video_path_for_episode(ep, vk))
        print(f"Using constructed paths: {len(paths_to_download)} MP4s ({n} episodes × {len(VIDEO_KEYS)} cameras).")

    skipped_existing = 0
    skipped_404 = 0
    failed_other = 0
    for rel_path in tqdm(paths_to_download, desc="Downloading MP4s"):
        local_path = os.path.join(local_dir, rel_path)
        if resume and os.path.isfile(local_path):
            skipped_existing += 1
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id=DROID_REPO_ID,
                filename=rel_path,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "Entry Not Found" in err_str or "not found" in err_str.lower():
                skipped_404 += 1
            else:
                failed_other += 1
                tqdm.write(f"Failed {rel_path}: {e}")
    if skipped_existing:
        print(f"Skipped {skipped_existing} existing files (resume=True).")
    if skipped_404:
        print(f"Skipped {skipped_404} files not found on Hub (404).")
    if failed_other:
        print(f"Other failures: {failed_other}.")
    print(f"Videos saved under: {local_dir}")


def download_metadata(cache_dir: str):
    """Download dataset metadata (arrow tables) via load_dataset."""
    from datasets import load_dataset
    load_dataset(
        DROID_REPO_ID,
        cache_dir=cache_dir,
    )
    print(f"Metadata cached under: {cache_dir}")


def download_videos_snapshot(cache_dir: str, max_workers: int = 32):
    """
    Download only the videos/ tree using snapshot_download with parallel workers.
    Much faster than one-by-one (single API listing + many concurrent downloads).
    """
    from huggingface_hub import snapshot_download

    # Download into a folder that mirrors the repo; only videos/** are fetched
    local_dir = os.path.join(cache_dir, "droid_repo")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading cadene/droid (videos only) with {max_workers} workers → {local_dir}")
    snapshot_download(
        repo_id=DROID_REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=["videos/**"],
        max_workers=max_workers,
    )
    print(f"Videos saved under: {local_dir}/videos/")


def download_videos_git_lfs(cache_dir: str, concurrent_transfers: int = 20):
    """
    Clone repo with GIT_LFS_SKIP_SMUDGE (fast), then git lfs pull only videos/.
    Works with standard git + git-lfs (no Xet). Run from a machine with git and git-lfs installed.
    """
    repo_url = "https://huggingface.co/datasets/cadene/droid"
    target_dir = os.path.join(cache_dir, "droid_repo")
    if os.path.isdir(target_dir):
        print(f"Directory exists: {target_dir}. Pulling LFS (videos only) inside it.")
        clone_dir = target_dir
        need_clone = False
    else:
        print(f"Cloning {repo_url} (LFS pointers only, no blobs yet)...")
        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        r = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            raise RuntimeError("git clone failed")
        clone_dir = target_dir
        need_clone = False

    print("Pulling LFS files for videos/ only (this will take a while for ~400GB)...")
    # Include pattern: all files under videos/ (gitignore-style)
    cmd = [
        "git",
        "-c",
        f"lfs.concurrenttransfers={concurrent_transfers}",
        "lfs",
        "pull",
        "--include",
        "videos/**",
    ]
    r = subprocess.run(cmd, cwd=clone_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise RuntimeError("git lfs pull failed")
    print(f"Videos saved under: {clone_dir}/videos/")


def main():
    parser = argparse.ArgumentParser(description="Download cadene/droid (metadata and/or videos).")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp2/hubertchang/data/droid",
        help="Directory for cache (metadata) and droid_videos/ subdir for MP4s.",
    )
    parser.add_argument("--video-only", action="store_true", help="Only download MP4 videos.")
    parser.add_argument("--metadata-only", action="store_true", help="Only run load_dataset (arrow cache).")
    parser.add_argument("--no-resume", action="store_true", help="Re-download even if file exists.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="If set, only download episodes 0..max_episodes-1.",
    )
    parser.add_argument(
        "--list-from-hub",
        action="store_true",
        help="(one-by-one only) List all MP4s from the Hub before downloading (slow).",
    )
    parser.add_argument(
        "--method",
        choices=["snapshot", "git-lfs", "one-by-one"],
        default="snapshot",
        help="How to download videos: snapshot (parallel HF download, default), git-lfs (clone + lfs pull), one-by-one (slow).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="For --method snapshot: number of parallel download workers (default 32).",
    )
    parser.add_argument(
        "--lfs-concurrent",
        type=int,
        default=20,
        help="For --method git-lfs: concurrent LFS transfers (default 20).",
    )
    args = parser.parse_args()

    if not args.video_only:
        download_metadata(args.cache_dir)
    if not args.metadata_only:
        if args.method == "snapshot":
            download_videos_snapshot(args.cache_dir, max_workers=args.max_workers)
        elif args.method == "git-lfs":
            download_videos_git_lfs(args.cache_dir, concurrent_transfers=args.lfs_concurrent)
        else:
            download_videos(
                args.cache_dir,
                resume=not args.no_resume,
                max_episodes=args.max_episodes,
                list_from_hub=args.list_from_hub,
            )


if __name__ == "__main__":
    main()
