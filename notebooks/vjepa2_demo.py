# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io
from transformers import AutoModel, AutoVideoProcessor

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform


def get_video(video_path):
    # Read video using torchvision
    # returns: video (Tensor[T, H, W, C]), audio, info
    video, _, _ = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    
    # choosing some frames here, you can define more complex sampling strategy
    # Ensure we don't go out of bounds if video is short
    total_frames = video.shape[0]
    frame_idx = np.arange(0, 128, 2)
    if len(frame_idx) > 0 and frame_idx[-1] >= total_frames:
        frame_idx = frame_idx[frame_idx < total_frames]

    # Convert to numpy and rearrange to T, H, W, C for consistency with previous decord output if needed,
    # BUT forward_vjepa_video expects T x C x H x W immediately after or handles it.
    # Previous decord code:
    # video = vr.get_batch(frame_idx).asnumpy() (T, H, W, C)
    # Then forward_vjepa_video did: torch.from_numpy(video).permute(0, 3, 1, 2)
    
    # torchvision read_video with output_format="TCHW" gives (T, C, H, W).
    # We will return it as T, H, W, C numpy array to minimize changes in forward_vjepa_video
    
    video = video[frame_idx].permute(0, 2, 3, 1).numpy()
    return video


def forward_vjepa_video(model_hf, model_pt, hf_transform, pt_transform, video_path):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video(video_path)  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_pt = model_pt(x_pt)
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf, out_patch_features_pt


def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

    return


def calculate_monotonicity_score(values):
    """
    Calculates the Spearman Rank Correlation between the indices and the values.
    A score of -1.0 implies a perfect monotonic decrease (what we want for distance to goal).
    A score of +1.0 implies a perfect monotonic increase.
    A score of 0.0 implies no monotonic trend.
    """
    if len(values) < 2:
        return 0.0
    
    indices = np.arange(len(values))
    correlation, _ = spearmanr(indices, values)
    return correlation


def calculate_smoothness_score(values):
    """
    Calculates a smoothness score where higher is better (smoother).
    
    We first calculate 'jaggedness' as the mean absolute second difference.
    Then we convert to a smoothness score in [0, 1]:
        Smoothness = 1.0 / (1.0 + Jaggedness)
    
    A perfectly straight line has jaggedness 0 -> Smoothness 1.0.
    Highly erratic curves have high jaggedness -> Smoothness approaches 0.0.
    """
    if len(values) < 3:
        return 1.0

    # First difference (velocity)
    diffs = np.diff(values)
    # Second difference (acceleration/change in slope)
    second_diffs = np.diff(diffs)

    # Jaggedness: mean magnitude of changes in slope
    jaggedness = np.mean(np.abs(second_diffs))
    
    # Convert to smoothness (higher is better)
    return 1.0 / (1.0 + jaggedness)


def plot_distance_to_goal(vectors, timestamps=None, goal_vector=None, save_path="distance_to_goal.png"):
    """
    Plots the L2 distance of a sequence of vectors to a goal vector.

    Args:
        vectors (torch.Tensor or np.ndarray): Sequence of latent vectors (T, D).
        timestamps (list or np.ndarray, optional): Time values for x-axis.
        goal_vector (torch.Tensor or np.ndarray, optional): The goal state vector (D,).
                                                            If None, use the last vector in the sequence.
        save_path (str): Path to save the plot.
    """
    if isinstance(vectors, np.ndarray):
        vectors = torch.from_numpy(vectors)
    if goal_vector is not None and isinstance(goal_vector, np.ndarray):
        goal_vector = torch.from_numpy(goal_vector)

    if goal_vector is None:
        goal_vector = vectors[-1]

    # Calculate L2 distances
    # vectors: (T, D), goal_vector: (D,)
    diff = vectors - goal_vector.unsqueeze(0)  # (T, D)
    distances = torch.norm(diff, p=2, dim=1).cpu().numpy()

    # Calculate metrics
    mono_score = calculate_monotonicity_score(distances)
    smooth_score = calculate_smoothness_score(distances)

    plt.figure(figsize=(10, 6))
    
    if timestamps is not None:
        plt.plot(timestamps, distances, marker="o", linestyle="-")
        plt.xlabel("Time (seconds)")
    else:
        plt.plot(distances, marker="o", linestyle="-")
        plt.xlabel("Time Step")
        
    plt.ylabel("L2 Distance to Goal")
    title = (
        f"Distance to Goal State over Time\n"
        f"Monotonicity (Spearman): {mono_score:.4f} (Target: -1.0) | "
        f"Smoothness: {smooth_score:.4f} (Target: 1.0)"
    )
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Distance plot saved to {save_path} | Monotonicity: {mono_score:.4f} | Smoothness: {smooth_score:.4f}")
    plt.close()


def extract_video_vectors(
    model, transform, video_path, mode="single_frame", tau=16, stride=1, device="cuda"
):
    """
    Extracts latent vectors from a video using the VJEPA model.

    Args:
        model (torch.nn.Module): The VJEPA model.
        transform (callable): Video transform function.
        video_path (str): Path to the video file.
        mode (str): 'single_frame' or 'clip'.
                    'single_frame': Encodes each frame as a state (duplicating to meet model input reqs).
                    'clip': Encodes a clip of length tau ending at t as the state at t.
        tau (int): Window size for 'clip' mode.
        stride (int): Stride for sampling frames.
        device (str): Device to run inference on.

    Returns:
        tuple: (vectors, timestamps)
            vectors (torch.Tensor): Sequence of state vectors (T, D).
            timestamps (list): Time in seconds corresponding to each vector.
    """
    # Read video using torchvision
    # returns: video (Tensor[T, H, W, C]), audio, info
    # We load the full video first. For very long videos, this might be memory intensive.
    video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    # video is (T, C, H, W)
    
    fps = info.get("video_fps", 30.0) # Default to 30 if not found
    
    total_frames = video.shape[0]
    vectors = []
    timestamps = []

    print(f"Extracting vectors from {video_path} (frames: {total_frames}, fps: {fps:.2f}, mode: {mode})")

    indices = []
    if mode == "single_frame":
        indices = range(0, total_frames, stride)
    elif mode == "clip":
        # Start from tau so we have a full clip [t-tau, t)
        indices = range(tau, total_frames + 1, stride)

    for i, t in enumerate(indices):
        if mode == "single_frame":
            # Extract single frame
            # frame is (C, H, W) -> need (H, W, C) for transform compatibility
            frame = video[t].permute(1, 2, 0).numpy() 
            # Duplicate to satisfy temporal dimension if needed (e.g. tubelet_size=2)
            clip = [frame, frame]
        elif mode == "clip":
            # Extract clip [t-tau, t)
            # clip_indices = range(t - tau, t)
            # video[start:end] works
            clip_tensor = video[t - tau : t] # (tau, C, H, W)
            # Convert to list of numpy arrays (H, W, C)
            clip = [c.permute(1, 2, 0).numpy() for c in clip_tensor]

        # Transform
        # Transform expects list of numpy arrays (H, W, C)
        # Returns (C, T, H, W) tensor
        x = transform(clip)
        x = x.unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.inference_mode():
            # Model output: (1, N_patches, D)
            features = model(x)
            # Average pool patches to get single state vector
            embedding = features.mean(dim=1).squeeze(0)  # (D,)

        vectors.append(embedding.cpu())
        # Calculate timestamp for current index t
        timestamps.append(t / fps)

        if i % 50 == 0:
            print(f"Processed {i}/{len(indices)}")

    if not vectors:
        return torch.empty(0), []

    return torch.stack(vectors), timestamps


def run_sample_inference():
    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitg-fpc64-384"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    # Path to local PyTorch weights
    pt_model_path = "/tmp2/hubertchang/p-progress/models/vitg-384.pt"

    # --- USER CONFIGURATION ---
    # Change this variable to choose another video path
    # target_video_path = "sample_video.mp4"
    target_video_path = "/home/hubertchang/p-progress/vjepa2/videos/S4_Hotdog_C1_trim.mp4"
    # --------------------------

    # Download the video if not yet downloaded to local path
    if target_video_path == "sample_video.mp4" and not os.path.exists(target_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        command = ["wget", video_url, "-O", target_video_path]
        subprocess.run(command)
        print("Downloading video")

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Initialize the PyTorch model, load pretrained weights
    model_pt = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
    model_pt.cuda().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # Inference on video
    out_patch_features_hf, out_patch_features_pt = forward_vjepa_video(
        model_hf, model_pt, hf_transform, pt_video_transform, target_video_path
    )

    print(
        f"""
        Inference results on video:
        HuggingFace output shape: {out_patch_features_hf.shape}
        PyTorch output shape:     {out_patch_features_pt.shape}
        Absolute difference sum:  {torch.abs(out_patch_features_pt - out_patch_features_hf).sum():.6f}
        Close: {torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3)}
        """
    )

    # Initialize the classifier
    classifier_model_path = "/tmp2/hubertchang/p-progress/models/ssv2-vitg-384-64x2x3.pt"
    classifier = (
        AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Download SSV2 classes if not already present
    ssv2_classes_path = "ssv2_classes.json"
    if not os.path.exists(ssv2_classes_path):
        command = [
            "wget",
            "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/"
            "something-something-v2-id2label.json",
            "-O",
            "ssv2_classes.json",
        ]
        subprocess.run(command)
        print("Downloading SSV2 classes")

    get_vjepa_video_classification_results(classifier, out_patch_features_pt)

    # --- New Demo: Extract Vectors and Plot Progress ---
    print("\n--- Running Video Vector Extraction Demo ---")
    # Use PyTorch model and transform
    # Extract using single frame mode
    # Note: Processing all frames might take time.
    vectors_single, timestamps_single = extract_video_vectors(
        model_pt,
        pt_video_transform,
        target_video_path,
        mode="single_frame",
        stride=2,
        device="cuda",
    )
    plot_distance_to_goal(vectors_single, timestamps=timestamps_single, save_path="distance_single_frame.png")

    # Extract using clip mode
    vectors_clip, timestamps_clip = extract_video_vectors(
        model_pt,
        pt_video_transform,
        target_video_path,
        mode="clip",
        tau=16,
        stride=2,  # Match tubelet_size (2) to avoid aliasing artifacts
        device="cuda",
    )
    plot_distance_to_goal(vectors_clip, timestamps=timestamps_clip, save_path="distance_clip.png")


if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    run_sample_inference()
