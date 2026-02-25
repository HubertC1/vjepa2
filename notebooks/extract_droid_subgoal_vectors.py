# Copyright (c) Meta Platforms, Inc. and affiliates.
# Extract subgoal latent vectors from DROID videos using VJEPA + subgoal discovery.
# Saves one sequence per mp4 (first + subgoal + last frame embeddings) to HDF5.
# Run from vjepa2 repo root: python notebooks/extract_droid_subgoal_vectors.py [args]

import json
import os
import sys
import argparse

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import numpy as np
import torch
import torchvision.io as io
from scipy.stats import spearmanr
from tqdm.auto import tqdm

# VJEPA (from compare_curves)
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# --- VJEPA (extracted from compare_curves.py) ---

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load checkpoint on CPU to avoid holding multiple full copies on GPU.
    print(f"Loading weights from {pretrained_weights}...")
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")
    if "encoder" in pretrained_dict:
        pretrained_dict = pretrained_dict["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Loaded with msg:", msg)
    del pretrained_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def load_vjepa_model(model_path, img_size=384, device="cuda"):
    print("Initializing VJEPA model...")
    model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
    if os.path.exists(model_path):
        load_pretrained_vjepa_pt_weights(model, model_path)
    else:
        raise FileNotFoundError(f"VJEPA model not found at {model_path}")
    model.to(device).eval()
    transform = build_pt_video_transform(img_size)
    return model, transform

def get_video_frames(video_path):
    video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    fps = info.get("video_fps", 30.0)
    video_np = video.permute(0, 2, 3, 1).numpy()
    return video_np, fps

def get_vjepa_embeddings(model, transform, frames, tau=1, stride=1, device="cuda"):
    embeddings = []
    if tau == 1:
        indices = list(range(0, len(frames), stride))
    else:
        indices = list(range(tau, len(frames) + 1, stride))
    for t in indices:
        if tau == 1:
            frame = frames[t]
            clip = [frame, frame]
        else:
            clip = frames[t - tau : t]
            if len(clip) < tau:
                missing = tau - len(clip)
                clip = [clip[0]] * missing + list(clip)
        x = transform(clip)
        x = x.unsqueeze(0).to(device)
        with torch.inference_mode():
            features = model(x)
            embedding = features.mean(dim=1).squeeze(0)
        embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings), indices

# --- Subgoal discovery (extracted from compare_curves.py) ---

def calculate_monotonicity_score(values):
    if len(values) < 2:
        return 0.0
    indices = np.arange(len(values))
    if np.all(values == values[0]):
        return 0.0
    correlation, _ = spearmanr(indices, values)
    return 0.0 if np.isnan(correlation) else correlation

def kernel_regression_smoothing(x, y, bandwidth):
    if len(x) < 2:
        return y
    x_col = x[:, np.newaxis]
    dist_sq = (x_col - x_col.T) ** 2
    weights = np.exp(-dist_sq / (2 * bandwidth**2))
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0
    return (weights @ y) / sum_weights

def discover_subgoals(vectors, indices, timestamps, threshold=0.0, time_threshold=15, distance_metric="l2", smoothing_bandwidth=None):
    if len(vectors) == 0:
        return [], np.array([]), np.array([])
    vectors_tensor = torch.from_numpy(vectors)
    T = len(vectors)
    goals_rel = [T - 1]
    dynamic_distances = np.zeros(T)
    dynamic_distances_raw = np.zeros(T)
    current_goal_idx = T - 1
    ts_norm = None
    if smoothing_bandwidth is not None and len(timestamps) > 1:
        ts_arr = np.array(timestamps)
        duration = ts_arr[-1] - ts_arr[0]
        ts_norm = (ts_arr - ts_arr[0]) / duration if duration > 1e-6 else np.zeros_like(ts_arr)
    for t in range(T - 2, -1, -1):
        segment_vectors = vectors_tensor[t : current_goal_idx + 1]
        target_goal = vectors_tensor[current_goal_idx]
        diff = segment_vectors - target_goal.unsqueeze(0)
        dists_raw = torch.norm(diff, p=1 if distance_metric == "l1" else 2, dim=1).numpy()
        dists_for_discovery = dists_raw
        if smoothing_bandwidth is not None and ts_norm is not None:
            seg_ts_norm = ts_norm[t : current_goal_idx + 1]
            dists_for_discovery = kernel_regression_smoothing(seg_ts_norm, dists_raw, smoothing_bandwidth)
        score = calculate_monotonicity_score(dists_for_discovery)
        dynamic_distances[t] = dists_for_discovery[0]
        dynamic_distances_raw[t] = dists_raw[0]
        if len(dists_for_discovery) >= time_threshold and score > threshold:
            current_goal_idx = t
            goals_rel.append(t)
            dynamic_distances[t] = 0.0
            dynamic_distances_raw[t] = 0.0
    goals_rel = sorted(goals_rel)
    goals_abs = [indices[g] for g in goals_rel]
    return goals_abs, dynamic_distances, dynamic_distances_raw


def collect_mp4_paths(videos_root, chunk_start=0, chunk_end=93):
    """
    Collect all mp4 paths under exterior_image_1_left and exterior_image_2_left
    (excludes wrist camera). Returns list of (key, path) where key is e.g.
    chunk_000/episode_000000/exterior_image_1_left.
    """
    out = []
    for c in range(chunk_start, chunk_end):
        chunk_name = f"chunk-{c:03d}"
        for cam in ("observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left"):
            dir_path = os.path.join(videos_root, chunk_name, cam)
            if not os.path.isdir(dir_path):
                continue
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(".mp4"):
                    episode_name = f.replace(".mp4", "")
                    key = f"{chunk_name}/{episode_name}/{cam}"
                    out.append((key, os.path.join(dir_path, f)))
    return out


def process_one_video(
    video_path,
    key,
    model,
    transform,
    device,
    subgoal_threshold=-0.99,
    subgoal_time_threshold=20,
    subgoal_smoothing_bandwidth=0.02,
    distance_metric="l1",
):
    """
    Load video -> VJEPA embeddings (every frame) -> discover subgoals -> return
    sequence of vectors at [first, subgoal_1, ..., subgoal_k, last].
    """
    frames, fps = get_video_frames(video_path)
    if len(frames) == 0:
        return None
    embs, indices = get_vjepa_embeddings(model, transform, frames, tau=1, stride=1, device=device)
    timestamps = [i / fps for i in indices]
    goals_abs, _, _ = discover_subgoals(
        embs, indices, timestamps,
        threshold=subgoal_threshold,
        time_threshold=subgoal_time_threshold,
        distance_metric=distance_metric,
        smoothing_bandwidth=subgoal_smoothing_bandwidth,
    )
    # With stride=1, indices = [0..T-1], so embedding index = frame index. Include first and last.
    T = len(embs)
    seq_positions = sorted(set([0, T - 1] + [g for g in goals_abs if 0 <= g < T]))
    vectors = embs[seq_positions]
    return vectors.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract DROID subgoal latent vectors to HDF5")
    parser.add_argument("--videos_root", type=str, default="/tmp2/hubertchang/p-jepa/data/droid/droid_repo/videos")
    parser.add_argument("--output_h5", type=str, default="droid_subgoal_vectors.h5")
    parser.add_argument("--output_manifest", type=str, default="droid_subgoal_manifest.json")
    parser.add_argument("--pt_model_path", type=str, default="/tmp2/hubertchang/p-jepa/models/vitg-384.pt")
    parser.add_argument("--chunk_start", type=int, default=0)
    parser.add_argument("--chunk_end", type=int, default=93)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subgoal_threshold", type=float, default=-0.99)
    parser.add_argument("--subgoal_time_threshold", type=int, default=20)
    parser.add_argument("--subgoal_smoothing_bandwidth", type=float, default=0.02)
    parser.add_argument("--distance_metric", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--skip_existing", action="store_true", help="Skip keys already in HDF5")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        print("Install h5py: pip install h5py")
        sys.exit(1)

    model, transform = load_vjepa_model(args.pt_model_path, device=args.device)
    paths = collect_mp4_paths(args.videos_root, chunk_start=args.chunk_start, chunk_end=args.chunk_end)
    print(f"Found {len(paths)} mp4 files")

    manifest = []
    with h5py.File(args.output_h5, "a") as f:
        for i, (key, path) in enumerate(tqdm(paths, desc="Processing videos")):
            if not os.path.exists(path):
                print(f"Skip (missing): {path}")
                continue
            # HDF5 group key: replace / with __ to avoid nested groups if desired; we use flat key
            h5_key = key.replace("/", "__")
            if args.skip_existing and h5_key in f:
                manifest.append({"key": key, "path": path, "h5_key": h5_key})
                continue
            try:
                vectors = process_one_video(
                    path, key, model, transform, args.device,
                    subgoal_threshold=args.subgoal_threshold,
                    subgoal_time_threshold=args.subgoal_time_threshold,
                    subgoal_smoothing_bandwidth=args.subgoal_smoothing_bandwidth,
                    distance_metric=args.distance_metric,
                )
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            if vectors is None:
                continue
            if h5_key in f:
                del f[h5_key]
            f.create_dataset(h5_key, data=vectors, compression="gzip")
            manifest.append({"key": key, "path": path, "h5_key": h5_key, "length": int(vectors.shape[0]), "dim": int(vectors.shape[1])})

    with open(args.output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {args.output_h5} and {args.output_manifest}")


if __name__ == "__main__":
    main()
