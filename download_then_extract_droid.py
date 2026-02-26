"""
Pipeline: download DROID one chunk at a time, extract subgoal vectors, then delete chunk videos.

Flow per chunk:
  1. Download only that chunk's videos (exterior_image_1_left, exterior_image_2_left) into cache.
  2. Run extract_droid_subgoal_vectors.py for that chunk; write chunk_XXX.h5 and chunk_XXX_manifest.json.
  3. Remove the chunk's video directory to free disk.

Useful on a single GPU (e.g. H100) to avoid storing the full ~400GB of videos at once.

Usage:
  cd /path/to/vjepa2
  python download_then_extract_droid.py --cache-dir /path/to/data/droid --output-dir /path/to/data/droid/subgoal
"""

import argparse
import os
import shutil
import subprocess
import sys

# DROID layout (match download_droid.py)
DROID_REPO_ID = "cadene/droid"
TOTAL_EPISODES = 92233
CHUNK_SIZE = 1000
EXTERIOR_VIDEO_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_left",
)


def video_path_for_episode(episode_index: int, video_key: str) -> str:
    chunk = episode_index // CHUNK_SIZE
    return f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def download_one_chunk(cache_dir: str, chunk_id: int, resume: bool = True) -> None:
    """Download all exterior-camera MP4s for this chunk into cache_dir/droid_repo/."""
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    local_dir = os.path.join(cache_dir, "droid_repo")
    os.makedirs(local_dir, exist_ok=True)
    start_ep = chunk_id * CHUNK_SIZE
    end_ep = min(start_ep + CHUNK_SIZE, TOTAL_EPISODES)
    paths = []
    for ep in range(start_ep, end_ep):
        for vk in EXTERIOR_VIDEO_KEYS:
            paths.append(video_path_for_episode(ep, vk))

    for rel_path in tqdm(paths, desc=f"Download chunk-{chunk_id:03d}"):
        local_path = os.path.join(local_dir, rel_path)
        if resume and os.path.isfile(local_path):
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
            if "404" not in str(e) and "not found" not in str(e).lower():
                tqdm.write(f"Failed {rel_path}: {e}")


def run_extract(
    repo_root: str,
    videos_root: str,
    output_dir: str,
    chunk_id: int,
    pt_model_path: str,
    device: str,
    video_batch_size: int,
    skip_existing: bool,
    subgoal_threshold: float,
    subgoal_time_threshold: int,
    subgoal_smoothing_bandwidth: float,
    distance_metric: str,
) -> bool:
    """Run extract_droid_subgoal_vectors.py for one chunk. Returns True on success."""
    output_h5 = os.path.join(output_dir, f"chunk_{chunk_id:03d}.h5")
    output_manifest = os.path.join(output_dir, f"chunk_{chunk_id:03d}_manifest.json")
    script = os.path.join(repo_root, "notebooks", "extract_droid_subgoal_vectors.py")
    if not os.path.isfile(script):
        print(f"Extract script not found: {script}", file=sys.stderr)
        return False

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        script,
        "--videos_root", videos_root,
        "--output_h5", output_h5,
        "--output_manifest", output_manifest,
        "--chunk_start", str(chunk_id),
        "--chunk_end", str(chunk_id + 1),
        "--pt_model_path", pt_model_path,
        "--device", device,
        "--video_batch_size", str(video_batch_size),
        "--subgoal_threshold", str(subgoal_threshold),
        "--subgoal_time_threshold", str(subgoal_time_threshold),
        "--subgoal_smoothing_bandwidth", str(subgoal_smoothing_bandwidth),
        "--distance_metric", distance_metric,
    ]
    if skip_existing:
        cmd.append("--skip_existing")

    ret = subprocess.run(cmd, cwd=repo_root)
    return ret.returncode == 0


def delete_chunk_videos(videos_root: str, chunk_id: int) -> None:
    """Remove the chunk's video directory to free disk."""
    chunk_dir = os.path.join(videos_root, f"chunk-{chunk_id:03d}")
    if os.path.isdir(chunk_dir):
        shutil.rmtree(chunk_dir)
        print(f"Removed {chunk_dir}")
    else:
        print(f"Chunk dir not found (already removed?): {chunk_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download DROID chunk-by-chunk, extract subgoal vectors, then delete chunk videos."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp2/hubertchang/data/droid",
        help="Base cache dir; videos go to cache_dir/droid_repo/videos/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for chunk_XXX.h5 and manifests. Default: cache_dir/subgoal",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to vjepa2 repo (default: parent of this script)",
    )
    parser.add_argument(
        "--chunk-start",
        type=int,
        default=0,
        help="First chunk index (inclusive)",
    )
    parser.add_argument(
        "--chunk-end",
        type=int,
        default=93,
        help="Last chunk index (exclusive). Default 93 for full DROID.",
    )
    parser.add_argument(
        "--skip-chunk-if-exists",
        action="store_true",
        help="Skip download+extract+delete for a chunk if its output .h5 already exists.",
    )
    parser.add_argument(
        "--no-delete-videos",
        action="store_true",
        help="Do not delete chunk videos after extraction (keep for debugging).",
    )
    parser.add_argument(
        "--no-resume-download",
        action="store_true",
        help="Re-download chunk files even if present.",
    )
    # Pass-through for extract script
    parser.add_argument("--pt_model_path", type=str, default="/tmp2/hubertchang/p-jepa/models/vitg-384.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--video_batch_size", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos already in HDF5 (extract script).")
    parser.add_argument("--subgoal_threshold", type=float, default=-0.99)
    parser.add_argument("--subgoal_time_threshold", type=int, default=20)
    parser.add_argument("--subgoal_smoothing_bandwidth", type=float, default=0.02)
    parser.add_argument("--distance_metric", type=str, default="l1", choices=["l1", "l2"])
    args = parser.parse_args()

    repo_root = args.repo_root or os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(cache_dir, "subgoal"))
    videos_root = os.path.join(cache_dir, "droid_repo", "videos")

    for chunk_id in range(args.chunk_start, args.chunk_end):
        output_h5 = os.path.join(output_dir, f"chunk_{chunk_id:03d}.h5")
        if args.skip_chunk_if_exists and os.path.isfile(output_h5):
            print(f"Chunk {chunk_id}: output exists, skipping.")
            continue

        print(f"--- Chunk {chunk_id} ---")
        download_one_chunk(cache_dir, chunk_id, resume=not args.no_resume_download)
        ok = run_extract(
            repo_root=repo_root,
            videos_root=videos_root,
            output_dir=output_dir,
            chunk_id=chunk_id,
            pt_model_path=args.pt_model_path,
            device=args.device,
            video_batch_size=args.video_batch_size,
            skip_existing=args.skip_existing,
            subgoal_threshold=args.subgoal_threshold,
            subgoal_time_threshold=args.subgoal_time_threshold,
            subgoal_smoothing_bandwidth=args.subgoal_smoothing_bandwidth,
            distance_metric=args.distance_metric,
        )
        if not ok:
            print(f"Chunk {chunk_id}: extract failed; not deleting videos.", file=sys.stderr)
            sys.exit(1)
        if not args.no_delete_videos:
            delete_chunk_videos(videos_root, chunk_id)
    print("Done.")


if __name__ == "__main__":
    main()
