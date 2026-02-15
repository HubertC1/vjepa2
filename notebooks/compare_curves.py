# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import datetime
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.io as io
from PIL import Image
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# VJEPA Imports
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

# Add VIP path (assuming vip is installed or in a known location relative to this script)
# Adjust this path as needed
VIP_PATH = '/home/hubertchang/p-progress/vip'
if os.path.exists(VIP_PATH):
    sys.path.append(VIP_PATH)
    from vip import load_vip
else:
    print(f"Warning: VIP path {VIP_PATH} not found. VIP models will not be available.")

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# --- VJEPA Utils ---

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    print(f"Loading weights from {pretrained_weights}...")
    try:
        # Load directly to GPU if possible to save CPU RAM
        print("Attempting to load weights directly to GPU to save CPU memory...")
        pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cuda:0")
        if "encoder" in pretrained_dict:
            pretrained_dict = pretrained_dict["encoder"]
        
        print("State dict loaded. Processing keys...")
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        
        print("Loading state dict into model...")
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
        
        del pretrained_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e

def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

def load_vjepa_model(model_path, img_size=384, device="cuda"):
    print(f"Initializing VJEPA model (img_size={img_size})...")
    model = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
    model.to(device).eval()
    if os.path.exists(model_path):
        load_pretrained_vjepa_pt_weights(model, model_path)
    else:
        print(f"Warning: VJEPA model not found at {model_path}")
    transform = build_pt_video_transform(img_size)
    return model, transform

# --- VIP Utils ---

def load_vip_model(device='cuda'):
    print("Initializing VIP model...")
    model = load_vip()
    model.to(device).eval()
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    return model, transform

# --- Extraction Logic ---

def get_video_frames(video_path):
    # Read video using torchvision
    # returns: video (Tensor[T, H, W, C]), audio, info
    print(f"Reading video from {video_path}...")
    video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    # video is (T, C, H, W)
    fps = info.get("video_fps", 30.0)
    
    # Convert to list of numpy arrays (H, W, C) for compatibility with transforms
    # T, C, H, W -> T, H, W, C
    video_np = video.permute(0, 2, 3, 1).numpy()
    return video_np, fps

def get_vip_embeddings(model, transform, frames, stride=1, device="cuda"):
    embeddings = []
    indices = range(0, len(frames), stride)
    timestamps = [] # We'll just return indices for now, caller handles timestamps
    
    print(f"Extracting VIP embeddings (stride={stride})...")
    
    # Batch processing
    batch_size = 64
    batch_imgs = []
    batch_indices = []
    
    for i in indices:
        img = frames[i] # H, W, C
        batch_imgs.append(transform(Image.fromarray(img)))
        batch_indices.append(i)
        
        if len(batch_imgs) >= batch_size:
            batch = torch.stack(batch_imgs).to(device)
            batch = batch * 255 # VIP expects 0-255
            with torch.no_grad():
                emb_batch = model(batch)
                embeddings.append(emb_batch.cpu().numpy())
            batch_imgs = []
            
    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        batch = batch * 255
        with torch.no_grad():
            emb_batch = model(batch)
            embeddings.append(emb_batch.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0), list(indices)

def get_vjepa_embeddings(model, transform, frames, tau=1, stride=1, device="cuda"):
    embeddings = []
    
    # Determine indices based on tau and stride
    # If tau=1 (single frame), we start at 0
    # If tau>1 (clip), we start at tau so we have a full clip [t-tau, t)
    
    if tau == 1:
        indices = range(0, len(frames), stride)
    else:
        indices = range(tau, len(frames) + 1, stride)
        
    print(f"Extracting VJEPA embeddings (tau={tau}, stride={stride})...")
    
    valid_indices = []
    
    for i, t in enumerate(indices):
        if tau == 1:
            # Single frame mode
            frame = frames[t] # H, W, C
            # Duplicate for temporal dim (tubelet_size=2 usually)
            clip = [frame, frame]
        else:
            # Clip mode [t-tau, t)
            clip = frames[t-tau : t]
            # Pad if needed (shouldn't happen with range logic above but safe to keep)
            if len(clip) < tau:
                missing = tau - len(clip)
                clip = [clip[0]] * missing + list(clip)
        
        # Transform expects list of numpy arrays (H, W, C)
        x = transform(clip) # (C, T, H, W)
        x = x.unsqueeze(0).to(device) # (1, C, T, H, W)
        
        with torch.inference_mode():
            features = model(x) # (1, N_patches, D)
            embedding = features.mean(dim=1).squeeze(0) # (D,)
            
        embeddings.append(embedding.cpu().numpy())
        valid_indices.append(t if tau == 1 else t) # Use end index as timestamp reference
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(indices)}")
            
    return np.array(embeddings), valid_indices

# --- Subgoal Discovery ---

def calculate_monotonicity_score(values):
    if len(values) < 2:
        return 0.0
    indices = np.arange(len(values))
    if np.all(values == values[0]):
        return 0.0
    correlation, _ = spearmanr(indices, values)
    if np.isnan(correlation):
        return 0.0
    return correlation

def discover_subgoals(vectors, indices, threshold=0.0, time_threshold=15, distance_metric='l2'):
    if len(vectors) == 0:
        return [], np.array([])
        
    vectors_tensor = torch.from_numpy(vectors)
    T = len(vectors)
    
    # Indices of identified goals (relative to the vectors array)
    goals_rel = [T - 1]
    dynamic_distances = np.zeros(T)
    current_goal_idx = T - 1
    
    print(f"Discovering subgoals (Backwards, Threshold={threshold}, Metric={distance_metric})...")
    
    for t in range(T - 2, -1, -1):
        segment_vectors = vectors_tensor[t : current_goal_idx + 1]
        target_goal = vectors_tensor[current_goal_idx]
        
        diff = segment_vectors - target_goal.unsqueeze(0)
        if distance_metric == 'l1':
            dists = torch.norm(diff, p=1, dim=1).numpy()
        else:
            dists = torch.norm(diff, p=2, dim=1).numpy()
        
        score = calculate_monotonicity_score(dists)
        dynamic_distances[t] = dists[0]
        
        if len(dists) >= time_threshold and score > threshold:
             current_goal_idx = t
             goals_rel.append(t)
             dynamic_distances[t] = 0.0
             
    goals_rel = sorted(goals_rel)
    dynamic_distances[T-1] = 0.0
    
    # Map relative indices back to original frame indices
    goals_abs = [indices[g] for g in goals_rel]
    
    return goals_abs, dynamic_distances

# --- Plotting and Saving ---

def plot_subgoal_discovery(timestamps, distances, goals, save_path, title_suffix="", distance_metric='l2'):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, distances, label="Distance to Dynamic Goal", linewidth=1.5)
    
    # Plot goal markers
    # goals contains frame indices, timestamps contains time for each frame in distances
    # We need to find which index in 'timestamps' corresponds to 'goals'
    # But 'timestamps' here are actually time values (float) passed from main loop
    
    # goals are absolute frame indices. 
    # timestamps is a list of time values corresponding to the extracted vectors.
    # We need to map goals (frame indices) to time values.
    # However, 'timestamps' passed to this function are already aligned with 'distances'.
    # So we need to find the index in the extracted sequence that corresponds to the goal frame.
    
    # Let's simplify: pass (extracted_indices, fps) instead of timestamps to map back.
    # Or just pass goal_times directly if computed outside.
    
    # Assuming 'timestamps' is the x-axis data (time in seconds) for the curve 'distances'
    # And 'goals' is a list of time values for the goals.
    
    plt.scatter(goals, [distances[np.argmin(np.abs(np.array(timestamps) - g))] for g in goals], 
                c='red', marker='x', s=100, label="Discovered Subgoals", zorder=5)
    
    for gt in goals:
        plt.axvline(x=gt, color='r', linestyle='--', alpha=0.3)

    plt.xlabel("Time (s)")
    plt.ylabel(f"{distance_metric.upper()} Distance to Current Subgoal")
    plt.title(f"Subgoal Discovery {title_suffix}\nFound {len(goals)} goals")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Subgoal plot saved to {save_path}")
    plt.close()

def save_subgoal_videos(video_path, goals_time, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving {len(goals_time)} subgoal clips to {output_dir}/ ...")
    
    try:
        clip = VideoFileClip(video_path)
        start_time = 0.0
        
        for i, end_time in enumerate(goals_time):
            if end_time > clip.duration:
                end_time = clip.duration
            
            if end_time > start_time:
                if hasattr(clip, "subclipped"):
                    subclip = clip.subclipped(start_time, end_time)
                else:
                    subclip = clip.subclip(start_time, end_time)
                
                output_filename = os.path.join(output_dir, f"subgoal_{i+1:02d}.mp4")
                subclip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
            
            start_time = end_time
            
        clip.close()

    except Exception as e:
        print(f"Error saving video clips: {e}")

def create_comparison_gif(frames, results, output_path, fps=30, distance_metric='l2'):
    # results: dict {name: (embeddings, indices)}
    
    print(f"Creating comparison GIF at {output_path}...")
    
    # Precompute distances for all methods
    all_dists = {}
    for name, (embs, indices) in results.items():
        goal = embs[-1]
        dists = []
        for t in range(len(embs)):
            diff = goal - embs[t]
            if distance_metric == 'l1':
                d = np.linalg.norm(diff, ord=1)
            else:
                d = np.linalg.norm(diff, ord=2)
            dists.append(d)
        all_dists[name] = (np.array(dists), indices)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    
    # Setup static parts
    ax[1].set_title("Input Video", fontsize=15)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    ax[0].set_title(f"Embedding Distance Comparison ({distance_metric.upper()})", fontsize=15)
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    
    # Determine max frame for x-axis
    max_frame = len(frames)
    ax[0].set_xlim(0, max_frame)
    
    # Determine y-axis limits
    max_dist = 0
    for d, _ in all_dists.values():
        if len(d) > 0:
            max_dist = max(max_dist, np.max(d))
    ax[0].set_ylim(0, max_dist * 1.1)

    def animate(i):
        # i is the current frame index of the video
        if i >= len(frames):
            return ax
            
        for ax_subplot in ax:
            ax_subplot.clear()
            
        # 1. Plot curves up to current frame i
        for name, (dists, indices) in all_dists.items():
            # Find which embeddings correspond to frames <= i
            # indices is list of frame indices for the embeddings
            valid_mask = np.array(indices) <= i
            if np.any(valid_mask):
                curr_dists = dists[valid_mask]
                curr_indices = np.array(indices)[valid_mask]
                ax[0].plot(curr_indices, curr_dists, label=name, linewidth=2)
        
        ax[0].set_title(f"Embedding Distance Comparison ({distance_metric.upper()})", fontsize=15)
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel("Frame", fontsize=15)
        ax[0].set_ylabel("Embedding Distance", fontsize=15)
        ax[0].set_xlim(0, max_frame)
        ax[0].set_ylim(0, max_dist * 1.1)

        # 2. Plot video frame
        ax[1].imshow(frames[i])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(f"Input Video (Frame {i})", fontsize=15)

        return ax

    ani = FuncAnimation(fig, animate, interval=1000/fps, repeat=False, frames=len(frames))
    ani.save(output_path, dpi=100, writer=PillowWriter(fps=fps))
    plt.close()
    print("GIF saved.")


# --- Main ---

def run_comparison():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/compare_{timestamp}"
    
    config = {
        "device": "cuda:0",
        # "video_path": "/home/hubertchang/p-progress/vjepa2/videos/S4_Hotdog_C1_trim.mp4",
        # "video_path": "/home/hubertchang/p-progress/vjepa2/videos/fold.mp4",
        "video_path": "/home/hubertchang/p-progress/vjepa2/notebooks/droid_samples/episode_005/exterior_image_2_left.mp4",
        # "video_path": "/home/hubertchang/p-progress/vip/vip/examples/demo_hotdog/subgoal_09.mp4",
        "pt_model_path": "/tmp2/hubertchang/p-progress/models/vitg-384.pt",
        "output_dir": output_dir,
        "distance_metric": "l1", # 'l1' or 'l2'
        "subgoal": {
            "monotonicity_threshold": -0.9,
            "time_threshold": 15
        },
        "configs": {
            "VIP": {
                "type": "vip",
                "stride": 1
            },
            "VJEPA_Single": {
                "type": "vjepa",
                "tau": 1,
                "stride": 1
            },
            "VJEPA_Clip_2": {
                "type": "vjepa",
                "tau": 2,
                "stride": 1
            },
            "VJEPA_Clip_8": {
                "type": "vjepa",
                "tau": 8,
                "stride": 1
            },
            "VJEPA_Clip_16": {
                "type": "vjepa",
                "tau": 16,
                "stride": 1
            }
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    device = config["device"]
    video_path = config["video_path"]
    
    # Load Video
    frames, fps = get_video_frames(video_path)
    print(f"Loaded video: {len(frames)} frames, {fps} fps")
    
    # Initialize Models
    # Only load if needed by config
    use_vip = any(c["type"] == "vip" for c in config["configs"].values())
    use_vjepa = any(c["type"] == "vjepa" for c in config["configs"].values())
    
    models = {}
    if use_vip:
        models["vip"] = load_vip_model(device=device)
    if use_vjepa:
        models["vjepa"] = load_vjepa_model(config["pt_model_path"], device=device)
        
    results = {} # name -> (embeddings, indices)
    
    # Run Extractions
    for name, cfg in config["configs"].items():
        print(f"\n--- Processing Configuration: {name} ---")
        model_type = cfg["type"]
        
        if model_type == "vip":
            model, transform = models["vip"]
            stride = cfg.get("stride", 1)
            embs, indices = get_vip_embeddings(model, transform, frames, stride=stride, device=device)
            
        elif model_type == "vjepa":
            model, transform = models["vjepa"]
            tau = cfg.get("tau", 1)
            stride = cfg.get("stride", 1)
            embs, indices = get_vjepa_embeddings(model, transform, frames, tau=tau, stride=stride, device=device)
            
        results[name] = (embs, indices)
        
        # Subgoal Discovery & Saving
        sub_dir = os.path.join(output_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        
        # Convert indices to timestamps for plotting/saving
        timestamps = [i / fps for i in indices]
        
        goals_abs, dynamic_dists = discover_subgoals(
            embs, indices, 
            threshold=config["subgoal"]["monotonicity_threshold"], 
            time_threshold=config["subgoal"]["time_threshold"],
            distance_metric=config.get("distance_metric", "l2")
        )
        
        goals_time = [g / fps for g in goals_abs]
        
        plot_subgoal_discovery(
            timestamps, dynamic_dists, goals_time, 
            save_path=os.path.join(sub_dir, "subgoal_discovery.png"),
            title_suffix=f"({name})",
            distance_metric=config.get("distance_metric", "l2")
        )
        
        save_subgoal_videos(video_path, goals_time, output_dir=os.path.join(sub_dir, "clips"))
        
    # Create Comparison GIF
    create_comparison_gif(
        frames, results, 
        output_path=os.path.join(output_dir, "comparison.gif"), 
        fps=fps,
        distance_metric=config.get("distance_metric", "l1")
    )
    
    print(f"\nDone! All results saved to {output_dir}")

if __name__ == "__main__":
    run_comparison()

