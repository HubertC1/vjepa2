# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Read embeddings HDF5 from extract_jepa_vectors.py, run subgoal discovery using
# VJEPA energy (mean of |a-b| over tokens and D), and save a second HDF5 with
# per-video subgoal embeddings [num_subgoals, N_tokens, D] and subgoal frame indices.
# Group key = video id (same as extract_jepa_vectors). Option A: save subgoal frame
# indices only (timestamps = frame_index / fps if fps known elsewhere).

import argparse
import os
import time
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr


def calculate_monotonicity_score(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    indices = np.arange(len(values))
    if np.all(values == values[0]):
        return 0.0
    correlation, _ = spearmanr(indices, values)
    return 0.0 if np.isnan(correlation) else float(correlation)


def kernel_regression_smoothing(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    if len(x) < 2:
        return y
    x_col = x[:, np.newaxis]
    dist_sq = (x_col - x_col.T) ** 2
    weights = np.exp(-dist_sq / (2 * bandwidth**2))
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights < 1e-10] = 1.0
    return (weights @ y) / sum_weights


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
    if len(vectors) == 0:
        return [], []

    vectors_tensor = torch.from_numpy(vectors)
    T = vectors_tensor.shape[0]
    goals_rel = [T - 1]
    current_goal_idx = T - 1

    # Normalized time for smoothing (from frame indices so spacing reflects real time)
    ts_norm = None
    if smoothing_bandwidth is not None and len(frame_indices) > 1:
        ts_arr = np.array(frame_indices, dtype=np.float64)
        duration = ts_arr[-1] - ts_arr[0]
        ts_norm = (ts_arr - ts_arr[0]) / duration if duration > 1e-6 else np.zeros_like(ts_arr)

    for t in range(T - 2, -1, -1):
        segment_vectors = vectors_tensor[t : current_goal_idx + 1]  # [L, N, D]
        target_goal = vectors_tensor[current_goal_idx]  # [N, D]
        diff = segment_vectors - target_goal.unsqueeze(0)  # [L, N, D]
        # VJEPA energy: mean over (N, D)
        dists_raw = torch.mean(torch.abs(diff), dim=(1, 2)).numpy()  # [L]

        dists_for_discovery = dists_raw
        if smoothing_bandwidth is not None and ts_norm is not None:
            seg_ts_norm = ts_norm[t : current_goal_idx + 1]
            dists_for_discovery = kernel_regression_smoothing(seg_ts_norm, dists_raw, smoothing_bandwidth)

        score = calculate_monotonicity_score(dists_for_discovery)
        if len(dists_for_discovery) >= time_threshold and score > threshold:
            current_goal_idx = t
            goals_rel.append(t)

    goals_rel = sorted(goals_rel)
    goals_abs = [int(frame_indices[g]) for g in goals_rel]
    return goals_rel, goals_abs


def main():
    parser = argparse.ArgumentParser(
        description="Run subgoal discovery on embeddings from extract_jepa_vectors.py and save subgoal-only dataset.",
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="HDF5 from extract_jepa_vectors.py (groups keyed by video id; embeddings [T,N,D], frame_indices [T]).",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        required=True,
        help="Output HDF5: one group per video (same key), subgoal_embeddings [num_subgoals, N, D], subgoal_frame_indices [num_subgoals].",
    )
    parser.add_argument("--monotonicity_threshold", type=float, default=-0.99)
    parser.add_argument("--time_threshold", type=int, default=15)
    parser.add_argument("--smoothing_bandwidth", type=float, default=0.02)
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos already present in output HDF5.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_h5) or ".", exist_ok=True)
    start_total = time.perf_counter()
    processed = 0
    skipped = 0

    with h5py.File(args.input_h5, "r") as fin, h5py.File(args.output_h5, "a") as fout:
        for key in fin.keys():
            if args.skip_existing and key in fout:
                skipped += 1
                continue
            grp_in = fin[key]
            embs = np.array(grp_in["embeddings"])  # [T, N_tokens, D]
            fi = np.array(grp_in["frame_indices"])  # [T]
            if embs.shape[0] == 0:
                continue
            goals_rel, goals_abs = discover_subgoals(
                embs,
                fi,
                threshold=args.monotonicity_threshold,
                time_threshold=args.time_threshold,
                smoothing_bandwidth=args.smoothing_bandwidth,
            )
            subgoal_embs = embs[np.array(goals_rel)]  # [num_subgoals, N_tokens, D]
            subgoal_fi = np.array(goals_abs, dtype=np.int64)
            out_grp = fout.create_group(key)
            out_grp.create_dataset("subgoal_embeddings", data=subgoal_embs.astype(np.float32), compression="gzip")
            out_grp.create_dataset("subgoal_frame_indices", data=subgoal_fi, compression="gzip")
            processed += 1

    elapsed = time.perf_counter() - start_total
    print(f"Done. Processed {processed} videos, skipped {skipped}. Output: {args.output_h5} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
