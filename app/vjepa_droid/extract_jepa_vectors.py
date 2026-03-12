import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
from PIL import Image
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from app.vjepa_droid.transforms import make_transforms
from src.models.vision_transformer import vit_giant_xformers_rope


def find_mp4s(root: str) -> List[str]:
    """
    Collect all MP4s under `root`, restricted to the exterior cameras
    (matches the DROID naming used elsewhere and excludes the wrist camera).
    """
    allowed_cams = {
        "observation.images.exterior_image_1_left",
        "observation.images.exterior_image_2_left",
    }

    mp4s: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        cam_name = os.path.basename(dirpath)
        if cam_name not in allowed_cams:
            continue
        for f in filenames:
            if f.lower().endswith(".mp4"):
                mp4s.append(os.path.join(dirpath, f))
    # Fallback for non-DROID layouts (e.g., a flat folder of mp4s).
    # Keep the camera restriction when matching folders are present,
    # but don't fail if users point to a generic videos directory.
    if not mp4s:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(dirpath, f))
    mp4s.sort()
    return mp4s


def load_video_as_tensor(
    path: str,
    transform,
    target_fps: float = None,
    min_video_seconds: float = 4.0,
    max_video_seconds: Optional[float] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load an MP4 with torchvision.io.read_video, sample frames (optionally downsampled by target_fps),
    and apply the VJEPA video transform.

    Returns:
        clip: tensor of shape [C, T, H, W] (float, normalized)
        indices: numpy array of original frame indices in the video
    """
    try:
        video, _, info = io.read_video(path, pts_unit="sec")  # video: [T, H, W, C], uint8
    except Exception as e:
        print(f"Failed to load video {path}: {e}")
        return None, None

    vlen = video.shape[0]
    if vlen == 0:
        print(f"Empty video: {path}")
        return None, None

    video_fps = info.get("video_fps", None)
    if video_fps is None or video_fps <= 0:
        # Fallback only for duration/step estimation. If target_fps is None,
        # default to 30 to avoid divide-by-zero while keeping behavior predictable.
        video_fps = target_fps if target_fps is not None else 30.0

    duration_sec = float(vlen) / float(video_fps)
    if duration_sec < float(min_video_seconds):
        print(
            f"Skipping short video {path}: duration={duration_sec:.2f}s "
            f"< min_video_seconds={min_video_seconds:.2f}s"
        )
        return None, None
    if max_video_seconds is not None and duration_sec > float(max_video_seconds):
        print(
            f"Skipping long video {path}: duration={duration_sec:.2f}s "
            f"> max_video_seconds={max_video_seconds:.2f}s"
        )
        return None, None

    if target_fps is None:
        indices = np.arange(0, vlen, 1, dtype=np.int64)
    else:
        frame_step = max(int(round(video_fps / target_fps)), 1)
        indices = np.arange(0, vlen, frame_step, dtype=np.int64)

    video = video[indices]  # [T', H, W, C]
    clip = transform(video)  # [C, T', H, W]
    return clip, indices


class Mp4Dataset(Dataset):
    def __init__(
        self,
        videos_root: str,
        transform,
        target_fps: float = None,
        min_video_seconds: float = 4.0,
        max_video_seconds: Optional[float] = None,
    ):
        self.videos_root = videos_root
        self.transform = transform
        self.target_fps = target_fps
        self.min_video_seconds = min_video_seconds
        self.max_video_seconds = max_video_seconds
        self.paths = find_mp4s(videos_root)
        if not self.paths:
            raise RuntimeError(f"No MP4 files found under {videos_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        rel = os.path.relpath(path, self.videos_root)
        h5_key = rel.replace(os.sep, "__").replace(".mp4", "")
        clip, frame_indices = load_video_as_tensor(
            path,
            transform=self.transform,
            target_fps=self.target_fps,
            min_video_seconds=self.min_video_seconds,
            max_video_seconds=self.max_video_seconds,
        )  # clip: [C, T, H, W]
        if clip is None:
            # Signal to collate_fn to drop this sample
            return None, None, None
        return clip, frame_indices, h5_key


def collate_padded(batch):
    """
    Collate function that pads clips in time so we can batch them:
    - inputs: list of (clip [C, T_i, H, W], frame_indices [T_i], h5_key)
    - outputs:
        clips_padded: [B, C, T_max, H, W]
        lengths: [B] (original T_i)
        frame_indices_padded: [B, T_max]
        keys: list of h5_keys
    """
    # Filter out failed loads (clip is None)
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None, None, []

    clips, frame_indices_list, keys = zip(*batch)
    lengths = [c.shape[1] for c in clips]
    C, _, H, W = clips[0].shape
    B = len(clips)
    T_max = max(lengths)

    clips_padded = torch.empty((B, C, T_max, H, W), dtype=clips[0].dtype)
    frame_indices_padded = np.zeros((B, T_max), dtype=np.int64)

    for i, (clip, fi) in enumerate(zip(clips, frame_indices_list)):
        T = clip.shape[1]
        clips_padded[i, :, :T] = clip
        frame_indices_padded[i, :T] = fi
        if T < T_max:
            # Pad by repeating the last frame and last index
            clips_padded[i, :, T:] = clip[:, T - 1 : T]
            frame_indices_padded[i, T:] = fi[-1]

    lengths = np.array(lengths, dtype=np.int64)
    return clips_padded, lengths, frame_indices_padded, list(keys)


def forward_target_batched(
    clips: torch.Tensor,
    target_encoder: torch.nn.Module,
    device: torch.device,
    crop_size: int,   # kept for API symmetry with train.py, unused here
    patch_size: int,  # kept for API symmetry with train.py, unused here
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Mirror forward_target() from train.py, but:
      - take batched clips [B, C, T, H, W]
      - return per-frame embeddings [B, T, D]
    """
    clips = clips.to(device, non_blocking=True)  # [B, C, T, H, W]
    B, C, T, H, W = clips.shape

    # [B, C, T, H, W] -> [B, T, C, H, W] -> [B*T, C, H, W]
    c = clips.permute(0, 2, 1, 3, 4).flatten(0, 1)
    # Add temporal dim of length 2 and repeat, as in forward_target:
    # [B*T, C, 1, H, W] -> [B*T, C, 2, H, W]
    c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)

    use_amp = dtype in (torch.float16, torch.bfloat16)
    amp_dtype = dtype if use_amp else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
        h = target_encoder(c)  # [B*T, N_tokens, D]

    BT, n_tokens, D = h.shape
    if BT != B * T:
        raise RuntimeError(f"Unexpected BT: {BT}, expected {B*T}")

    # Match train.py semantics: treat all tokens per frame as a set, without
    # enforcing a specific temporal structure in the token dimension.
    # [B*T, N_tokens, D] -> [B, T, N_tokens, D] -> mean over tokens -> [B, T, D]
    # h = h.view(B, T, n_tokens, D).mean(dim=2)
    h = h.view(B, T, n_tokens, D)
    return h


def calculate_monotonicity_score(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    indices = np.arange(len(values))
    if np.all(values == values[0]):
        return 0.0
    correlation, _ = spearmanr(indices, values)
    return 0.0 if np.isnan(correlation) else float(correlation)


def kernel_regression_smoothing(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    if len(x) < 2:
        return y
    x_col = x[:, np.newaxis]
    dist_sq = (x_col - x_col.T) ** 2
    weights = np.exp(-dist_sq / (2 * bandwidth**2))
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0
    return (weights @ y) / sum_weights


def _discover_subgoals_internal(
    vectors: np.ndarray,
    frame_indices: np.ndarray,
    threshold: float = 0.0,
    time_threshold: int = 15,
    smoothing_bandwidth: Optional[float] = None,
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """
    Internal helper for backward subgoal discovery using VJEPA energy:
    mean(|a-b|) over (tokens, D).

    Returns:
        goals_rel: positions in [0..T-1] of subgoals (for indexing into vectors).
        goals_abs: original frame indices of subgoals (frame_indices[goals_rel]).
        dynamic_distances: per-step distance used for discovery (possibly smoothed).
        dynamic_distances_raw: per-step raw distance without smoothing.
    """
    if len(vectors) == 0:
        return [], [], np.array([]), np.array([])

    vectors_tensor = torch.from_numpy(vectors)
    T = vectors_tensor.shape[0]

    goals_rel = [T - 1]
    current_goal_idx = T - 1
    min_goal_index = max(int(time_threshold), 0)

    dynamic_distances = np.zeros(T, dtype=np.float32)
    dynamic_distances_raw = np.zeros(T, dtype=np.float32)

    ts_norm = None
    if smoothing_bandwidth is not None and len(frame_indices) > 1:
        ts_arr = np.array(frame_indices, dtype=np.float64)
        duration = ts_arr[-1] - ts_arr[0]
        ts_norm = (ts_arr - ts_arr[0]) / duration if duration > 1e-6 else np.zeros_like(ts_arr)

    for t in range(T - 2, -1, -1):
        segment_vectors = vectors_tensor[t : current_goal_idx + 1]  # [L, N, D]
        target_goal = vectors_tensor[current_goal_idx]  # [N, D]
        diff = segment_vectors - target_goal.unsqueeze(0)  # [L, N, D]
        dists_raw = torch.mean(torch.abs(diff), dim=(1, 2)).numpy()  # [L]

        dists_for_discovery = dists_raw
        if smoothing_bandwidth is not None and ts_norm is not None:
            seg_ts_norm = ts_norm[t : current_goal_idx + 1]
            dists_for_discovery = kernel_regression_smoothing(seg_ts_norm, dists_raw, smoothing_bandwidth)

        score = calculate_monotonicity_score(dists_for_discovery)
        dynamic_distances[t] = dists_for_discovery[0]
        dynamic_distances_raw[t] = dists_raw[0]

        # Only accept/reset to a discovered subgoal when it is far enough from
        # the start. This avoids introducing artificial dips for subgoals that
        # would later be filtered out for being too close to initialization.
        if (
            t >= min_goal_index
            and len(dists_for_discovery) >= time_threshold
            and score > threshold
        ):
            current_goal_idx = t
            goals_rel.append(t)
            dynamic_distances[t] = 0.0
            dynamic_distances_raw[t] = 0.0

    goals_rel = sorted(set(goals_rel))
    goals_rel = [g for g in goals_rel if g == (T - 1) or g >= min_goal_index]
    # Always keep the very first sampled frame as a subgoal anchor so downstream
    # pairwise planning can start from the true initial state.
    if T > 0 and 0 not in goals_rel:
        goals_rel = [0] + goals_rel
    if T > 0:
        dynamic_distances[T - 1] = 0.0
        dynamic_distances_raw[T - 1] = 0.0

    goals_abs = [int(frame_indices[g]) for g in goals_rel]
    return goals_rel, goals_abs, dynamic_distances, dynamic_distances_raw


def discover_subgoals(
    vectors: np.ndarray,
    frame_indices: np.ndarray,
    threshold: float = 0.0,
    time_threshold: int = 15,
    smoothing_bandwidth: Optional[float] = None,
) -> Tuple[List[int], List[int]]:
    """
    Backward subgoal discovery using VJEPA energy: mean(|a-b|) over (tokens, D).

    vectors: [T, N_tokens, D]
    frame_indices: [T] original video frame index for each position (used for timestamps in smoothing).

    Returns:
        goals_rel: positions in [0..T-1] of subgoals (for indexing into vectors).
        goals_abs: original frame indices of subgoals (frame_indices[goals_rel]).
    """
    goals_rel, goals_abs, _, _ = _discover_subgoals_internal(
        vectors=vectors,
        frame_indices=frame_indices,
        threshold=threshold,
        time_threshold=time_threshold,
        smoothing_bandwidth=smoothing_bandwidth,
    )
    return goals_rel, goals_abs


def discover_subgoals_with_distances(
    vectors: np.ndarray,
    frame_indices: np.ndarray,
    threshold: float = 0.0,
    time_threshold: int = 15,
    smoothing_bandwidth: Optional[float] = None,
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """
    Variant of discover_subgoals that also returns the per-step distances
    used for visualization/debugging.

    Returns:
        goals_rel: positions in [0..T-1] of subgoals.
        goals_abs: original frame indices of subgoals.
        dynamic_distances: per-step distance used for discovery (possibly smoothed).
        dynamic_distances_raw: per-step raw distance without smoothing.
    """
    return _discover_subgoals_internal(
        vectors=vectors,
        frame_indices=frame_indices,
        threshold=threshold,
        time_threshold=time_threshold,
        smoothing_bandwidth=smoothing_bandwidth,
    )


def plot_subgoal_discovery(
    timestamps: np.ndarray,
    distances_smooth: np.ndarray,
    distances_raw: Optional[np.ndarray],
    goals_rel: List[int],
    save_path: str,
    title_suffix: str = "",
    monotonicity_threshold: Optional[float] = None,
    smoothing_bandwidth: Optional[float] = None,
) -> None:
    """
    Save a PNG visualization of the subgoal discovery curve and selected subgoals.
    """
    if len(timestamps) == 0 or len(distances_smooth) == 0:
        return

    plt.figure(figsize=(12, 6))

    all_dists = [np.max(distances_smooth)]
    if distances_raw is not None and len(distances_raw) == len(timestamps):
        all_dists.append(np.max(distances_raw))
    max_dist = max(all_dists) if all_dists else 1.0

    plt.plot(
        timestamps,
        distances_smooth,
        label="Smoothed Distance",
        linewidth=2,
    )

    if distances_raw is not None and len(distances_raw) == len(timestamps):
        plt.plot(
            timestamps,
            distances_raw,
            label="Raw Distance",
            linewidth=1,
            alpha=0.6,
            linestyle="--",
        )

    if goals_rel:
        goals_time = [timestamps[g] for g in goals_rel]
        goal_dists = [distances_smooth[g] for g in goals_rel]
        plt.scatter(
            goals_time,
            goal_dists,
            c="red",
            marker="x",
            s=100,
            label="Discovered Subgoals",
            zorder=5,
        )
        for gt in goals_time:
            plt.axvline(x=gt, color="r", linestyle="--", alpha=0.3)

    title = "Subgoal Discovery"
    if title_suffix:
        title += f" {title_suffix}"
    if monotonicity_threshold is not None:
        title += f"\nMono. Thresh: {monotonicity_threshold}"
    if smoothing_bandwidth is not None:
        title += f"\nSmoothing BW: {smoothing_bandwidth}"

    plt.xlabel("Time (s)")
    plt.ylabel("Distance to Current Subgoal")
    plt.title(title)
    plt.ylim(0, max_dist * 1.1)
    plt.xlim(0, timestamps[-1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_subgoal_discovery_gif(
    frames: np.ndarray,
    timestamps: np.ndarray,
    distances_smooth: np.ndarray,
    distances_raw: Optional[np.ndarray],
    goals_rel: List[int],
    output_path: str,
    fps: float = 10.0,
    title_suffix: str = "",
    monotonicity_threshold: Optional[float] = None,
    smoothing_bandwidth: Optional[float] = None,
    pause_seconds_at_subgoal: float = 1.0,
) -> None:
    """
    Create an animated GIF showing the evolution of the distance curve along with the video.
    """
    if len(frames) == 0 or len(timestamps) == 0 or len(distances_smooth) == 0:
        return

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    max_time = float(timestamps[-1])
    all_dists = [np.max(distances_smooth)]
    if distances_raw is not None and len(distances_raw) == len(timestamps):
        all_dists.append(np.max(distances_raw))
    max_dist = max(all_dists) if all_dists else 1.0

    title = "Subgoal Discovery"
    if title_suffix:
        title += f" {title_suffix}"
    if monotonicity_threshold is not None:
        title += f"\nMono. Thresh: {monotonicity_threshold}"
    if smoothing_bandwidth is not None:
        title += f"\nSmoothing BW: {smoothing_bandwidth}"

    def animate(i: int):
        if i >= len(frames):
            return ax

        for a in ax:
            a.clear()

        current_time = timestamps[i]
        valid_mask = timestamps <= current_time
        curr_timestamps = timestamps[valid_mask]
        curr_dists_smooth = distances_smooth[valid_mask]

        ax[0].plot(curr_timestamps, curr_dists_smooth, label="Smoothed Distance", linewidth=2)

        if distances_raw is not None and len(distances_raw) == len(timestamps):
            curr_dists_raw = distances_raw[valid_mask]
            ax[0].plot(
                curr_timestamps,
                curr_dists_raw,
                label="Raw Distance",
                linewidth=1,
                alpha=0.6,
                linestyle="--",
            )

        passed_goals = [g for g in goals_rel if g <= i]
        if passed_goals:
            passed_goal_times = [timestamps[g] for g in passed_goals]
            passed_goal_dists = [distances_smooth[g] for g in passed_goals]
            ax[0].scatter(
                passed_goal_times,
                passed_goal_dists,
                c="red",
                marker="x",
                s=100,
                label="Subgoals",
                zorder=5,
            )
            for gt in passed_goal_times:
                ax[0].axvline(x=gt, color="r", linestyle="--", alpha=0.3)

        ax[0].set_title(title, fontsize=14)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Distance to Current Subgoal")
        ax[0].set_xlim(0, max_time)
        ax[0].set_ylim(0, max_dist * 1.1)
        ax[0].legend(loc="upper right")

        ax[1].imshow(frames[i])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(f"Frame {i}", fontsize=14)

        return ax

    gif_fps = float(max(fps, 1.0))

    # Pause on discovered subgoal frames by repeating those frame indices.
    pause_frames_count = max(int(round(gif_fps * max(pause_seconds_at_subgoal, 0.0))), 0)
    goals_set = set(goals_rel)
    animation_frames: List[int] = []
    for i in range(len(frames)):
        animation_frames.append(i)
        if i in goals_set and pause_frames_count > 0:
            animation_frames.extend([i] * pause_frames_count)

    ani = FuncAnimation(fig, animate, interval=1000 / gif_fps, repeat=False, frames=animation_frames)
    ani.save(output_path, dpi=100, writer=PillowWriter(fps=gif_fps))
    plt.close()


def save_discovered_subgoal_frames(
    frames: np.ndarray,
    goals_rel: List[int],
    frame_indices: np.ndarray,
    output_dir: str,
    image_size: int = 256,
) -> None:
    """
    Save discovered subgoal frames as resized PNG images.
    """
    if len(frames) == 0 or not goals_rel:
        return

    os.makedirs(output_dir, exist_ok=True)
    for subgoal_idx, rel_idx in enumerate(goals_rel):
        if rel_idx < 0 or rel_idx >= len(frames):
            continue

        frame_np = frames[rel_idx]
        # Ensure uint8 RGB image for PIL.
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame_np)
        img = img.resize((image_size, image_size), resample=Image.BICUBIC)

        abs_idx = int(frame_indices[rel_idx]) if rel_idx < len(frame_indices) else -1
        out_name = f"subgoal_{subgoal_idx:03d}_rel_{rel_idx:05d}_abs_{abs_idx:06d}.png"
        img.save(os.path.join(output_dir, out_name))


def make_output_h5_path(base_path: str, file_index: int) -> str:
    """
    Given a base H5 path like /path/to/output.h5, return a sharded path such as
    /path/to/output_part0000.h5, /path/to/output_part0001.h5, ...
    """
    root, ext = os.path.splitext(base_path)
    if not ext:
        return f"{base_path}_part{file_index:04d}.h5"
    return f"{root}_part{file_index:04d}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen VJEPA frame embeddings from MP4 videos using the DROID AC-predictor encoder, "
            "run subgoal discovery, and save subgoal-only embeddings per video."
        )
    )
    parser.add_argument(
        "--videos_root",
        type=str,
        required=True,
        help="Root directory; all *.mp4 files under this tree will be processed.",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        required=True,
        help="Output HDF5 file to store embeddings (one group per video).",
    )
    parser.add_argument(
        "--vjepa_checkpoint",
        type=str,
        required=True,
        help="Path to VJEPA checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of videos per encoder forward.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for Decord loading.")
    parser.add_argument("--target_fps", type=float, default=None, help="If set, downsample videos to this fps.")
    parser.add_argument(
        "--min_video_seconds",
        type=float,
        default=4.0,
        help="Skip videos shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--max_video_seconds",
        type=float,
        default=None,
        help="Skip videos longer than this duration in seconds.",
    )
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=512,
        help="Max frames for init_video_model (matches training config; not a hard limit here).",
    )
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Inference dtype for the encoder.",
    )
    parser.add_argument(
        "--monotonicity_threshold",
        type=float,
        default=-0.99,
        help="Monotonicity threshold for subgoal discovery (higher = fewer subgoals).",
    )
    parser.add_argument(
        "--time_threshold",
        type=int,
        default=15,
        help="Minimum segment length in sampled-frame steps required before a new subgoal is accepted.",
    )
    parser.add_argument(
        "--time_threshold_seconds",
        type=float,
        default=None,
        help=(
            "Optional seconds-based threshold. If set, per-video threshold steps become "
            "round(time_threshold_seconds * target_fps). Requires --target_fps."
        ),
    )
    parser.add_argument(
        "--smoothing_bandwidth",
        type=float,
        default=0.02,
        help="Bandwidth for kernel regression smoothing in subgoal discovery (None to disable).",
    )
    parser.add_argument(
        "--videos_per_file",
        type=int,
        default=500,
        help="Number of videos to store per output HDF5 file before starting a new file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug visualization mode (process a few videos and save GIFs/plots).",
    )
    parser.add_argument(
        "--debug_num_videos",
        type=int,
        default=3,
        help="Maximum number of videos to process when --debug is enabled.",
    )
    parser.add_argument(
        "--debug_start_video_index",
        type=int,
        default=-1,
        help="Start video index when --debug is enabled. If -1, start from random.",
    )
    parser.add_argument(
        "--debug_output_dir",
        type=str,
        default=None,
        help="Directory to save debug GIFs/plots (defaults to <output_h5>_debug).",
    )
    args = parser.parse_args()
    if args.min_video_seconds < 0:
        parser.error("--min_video_seconds must be >= 0.")
    if args.max_video_seconds is not None and args.max_video_seconds < 0:
        parser.error("--max_video_seconds must be >= 0 when set.")
    if args.max_video_seconds is not None and args.max_video_seconds < args.min_video_seconds:
        parser.error("--max_video_seconds must be >= --min_video_seconds.")
    if args.time_threshold_seconds is not None:
        if args.time_threshold_seconds <= 0:
            parser.error("--time_threshold_seconds must be > 0.")
        if args.target_fps is None:
            parser.error("--time_threshold_seconds requires --target_fps.")

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.time_threshold_seconds is None:
        effective_time_threshold = args.time_threshold
    else:
        effective_time_threshold = max(1, int(round(args.time_threshold_seconds * args.target_fps)))
        print(
            f"Using time_threshold_seconds={args.time_threshold_seconds:.3f}s "
            f"with target_fps={args.target_fps} -> effective_time_threshold={effective_time_threshold} steps"
        )

    target_encoder = vit_giant_xformers_rope(img_size=(args.crop_size, args.crop_size), num_frames=args.max_num_frames)
    ckpt = torch.load(args.vjepa_checkpoint, map_location="cpu")
    if "encoder" in ckpt:
        ckpt = ckpt["encoder"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items()}
    msg = target_encoder.load_state_dict(ckpt, strict=False)
    print(f"Loaded VJEPA encoder with msg: {msg}")


    target_encoder = target_encoder.to(device)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=args.crop_size,
    )

    import h5py
    import random

    # Dataset + DataLoader over all MP4s
    dataset = Mp4Dataset(
        args.videos_root,
        transform=transform,
        target_fps=args.target_fps,
        min_video_seconds=args.min_video_seconds,
        max_video_seconds=args.max_video_seconds,
    )
    if args.debug:
        # In debug mode, restrict to a smaller number of videos.
        # dataset.paths = dataset.paths[: max(1, args.debug_num_videos)]
        #random sample debug_num_videos videos
        if args.debug_start_video_index == -1:
            dataset.paths = random.sample(dataset.paths, args.debug_num_videos)
        else:
            dataset.paths = dataset.paths[args.debug_start_video_index:]

    # Map HDF5 keys back to original video paths for debug visualizations.
    key_to_path = {}
    for p in dataset.paths:
        rel = os.path.relpath(p, args.videos_root)
        k = rel.replace(os.sep, "__").replace(".mp4", "")
        key_to_path[k] = p
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_padded,
    )

    # HDF5 output (rotate to a new file every args.videos_per_file videos)
    out_dir = os.path.dirname(args.output_h5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    debug_output_dir = None
    if args.debug:
        if args.debug_output_dir is not None:
            debug_output_dir = args.debug_output_dir
        else:
            base, _ = os.path.splitext(args.output_h5)
            debug_output_dir = base + "_debug"
        os.makedirs(debug_output_dir, exist_ok=True)

    start_total = time.perf_counter()
    processed = 0
    current_file_index = 0
    current_output_path = make_output_h5_path(args.output_h5, current_file_index)
    print(f"Writing to HDF5 file: {current_output_path}")
    f = h5py.File(current_output_path, "a")

    try:
        for batch_idx, (clips, lengths, frame_indices, keys) in enumerate(loader):
            # collate_padded can return (None, None, None, []) if all videos in the batch failed to load
            if clips is None or not keys:
                continue

            t0 = time.perf_counter()
            B = clips.shape[0]

            # Run encoder on full batch [B, C, T, H, W]
            enc_start = time.perf_counter()
            try:
                embs_bt = forward_target_batched(
                    clips,
                    target_encoder=target_encoder,
                    device=device,
                    crop_size=args.crop_size,
                    patch_size=args.patch_size,
                    dtype=dtype,
                )  # [B, T, N_tokens, D]
            except Exception as e:
                print(f"Failed to encode batch {batch_idx}: {e}")
                continue
            enc_dt = time.perf_counter() - enc_start
            embs_bt = embs_bt.cpu().numpy()  # [B, T, N_tokens, D]
            lengths_np = lengths

            saved_this_batch = 0

            # Save per-video (subgoal-only embeddings)
            for local_idx, key in enumerate(keys):
                # Rotate to a new HDF5 file once we've reached the target number of videos
                if processed > 0 and processed % args.videos_per_file == 0:
                    f.close()
                    current_file_index += 1
                    current_output_path = make_output_h5_path(args.output_h5, current_file_index)
                    print(f"Rotating to new HDF5 file: {current_output_path}")
                    f = h5py.File(current_output_path, "a")

                # Skip videos already present in the current HDF5 file
                if key in f:
                    continue

                T_i = int(lengths_np[local_idx])
                embs_i = embs_bt[local_idx, :T_i]  # [T_i, N_tokens, D]
                fi_i = frame_indices[local_idx, :T_i]

                goals_rel, goals_abs = discover_subgoals(
                    embs_i,
                    fi_i,
                    threshold=args.monotonicity_threshold,
                    time_threshold=effective_time_threshold,
                    smoothing_bandwidth=args.smoothing_bandwidth,
                )

                subgoal_embs = embs_i[np.array(goals_rel)]  # [num_subgoals, N_tokens, D]
                subgoal_fi = np.array(goals_abs, dtype=np.int64)

                grp = f.create_group(key)
                # Note: we keep dataset names "embeddings" / "frame_indices" but they now contain only subgoal frames.
                grp.create_dataset("embeddings", data=subgoal_embs.astype(np.float32), compression="gzip")
                grp.create_dataset("frame_indices", data=subgoal_fi, compression="gzip")
                processed += 1
                saved_this_batch += 1

                # Optional debug visualization for this video.
                if args.debug and debug_output_dir is not None and key in key_to_path:
                    video_path = key_to_path[key]
                    try:
                        video_full, _, info = io.read_video(video_path, pts_unit="sec")
                        source_fps = float(info.get("video_fps", 30.0) or 30.0)
                    except Exception as e:
                        print(f"Failed to reload video {video_path} for debug viz: {e}")
                        continue

                    # Frames corresponding to the indices used for embeddings.
                    try:
                        frames_debug = video_full[fi_i]  # [T_i, H, W, C]
                    except Exception as e:
                        print(f"Failed to slice frames for debug viz ({video_path}): {e}")
                        continue

                    timestamps = fi_i.astype(np.float32) / source_fps

                    # Reconstruct the effective sampled frame rate from frame indices.
                    if len(fi_i) > 1:
                        step = float(np.median(np.diff(fi_i)))
                        sampled_fps = source_fps / max(step, 1.0)
                    else:
                        sampled_fps = source_fps

                    print(
                        f"[debug] {key}: source_fps={source_fps:.3f}, "
                        f"target_fps={args.target_fps}, sampled_fps={sampled_fps:.3f}, "
                        f"sampled_frames={len(fi_i)}, time_threshold_steps={effective_time_threshold}"
                    )

                    (
                        goals_rel_dbg,
                        _goals_abs_dbg,
                        dyn_dists,
                        dyn_dists_raw,
                    ) = discover_subgoals_with_distances(
                        embs_i,
                        fi_i,
                        threshold=args.monotonicity_threshold,
                        time_threshold=effective_time_threshold,
                        smoothing_bandwidth=args.smoothing_bandwidth,
                    )

                    video_debug_dir = os.path.join(debug_output_dir, key, f"target_fps={args.target_fps}", f"smoothing_bandwidth={args.smoothing_bandwidth}")

                    os.makedirs(video_debug_dir, exist_ok=True)

                    png_path = os.path.join(video_debug_dir, "subgoal_discovery.png")
                    gif_path = os.path.join(video_debug_dir, "subgoal_discovery.gif")
                    subgoal_frames_dir = os.path.join(video_debug_dir, "subgoal_frames")

                    plot_subgoal_discovery(
                        timestamps=timestamps,
                        distances_smooth=dyn_dists,
                        distances_raw=dyn_dists_raw,
                        goals_rel=goals_rel_dbg,
                        save_path=png_path,
                        title_suffix="",
                        monotonicity_threshold=args.monotonicity_threshold,
                        smoothing_bandwidth=args.smoothing_bandwidth,
                    )

                    create_subgoal_discovery_gif(
                        frames=frames_debug.numpy(),
                        timestamps=timestamps,
                        distances_smooth=dyn_dists,
                        distances_raw=dyn_dists_raw,
                        goals_rel=goals_rel_dbg,
                        output_path=gif_path,
                        fps=sampled_fps,
                        title_suffix="",
                        monotonicity_threshold=args.monotonicity_threshold,
                        smoothing_bandwidth=args.smoothing_bandwidth,
                    )
                    save_discovered_subgoal_frames(
                        frames=frames_debug.numpy(),
                        goals_rel=goals_rel_dbg,
                        frame_indices=fi_i,
                        output_dir=subgoal_frames_dir,
                        image_size=256,
                    )

            dt = time.perf_counter() - t0
            print(
                f"[batch {batch_idx+1}] saved {saved_this_batch} / {B} videos, "
                f"time={dt:.2f}s, enc_time={enc_dt:.2f}s"
            )
    finally:
        f.close()

    total_dt = time.perf_counter() - start_total
    print(
        f"Done. Processed {processed} videos in {total_dt/60.0:.1f} min. "
        f"Output prefix: {args.output_h5}"
    )


if __name__ == "__main__":
    main()

