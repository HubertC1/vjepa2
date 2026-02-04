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
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

import matplotlib.pyplot as plt

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


def get_video():
    vr = VideoReader("sample_video.mp4")
    # choosing some frames here, you can define more complex sampling strategy
    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video


def forward_vjepa_video(model_hf, model_pt, hf_transform, pt_transform):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video()  # T x H x W x C
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


def plot_distance_to_goal(vectors, goal_vector=None, save_path="distance_to_goal.png"):
    """
    Plots the L2 distance of a sequence of vectors to a goal vector.

    Args:
        vectors (torch.Tensor or np.ndarray): Sequence of latent vectors (T, D).
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

    plt.figure(figsize=(10, 6))
    plt.plot(distances, marker="o", linestyle="-")
    plt.xlabel("Time Step")
    plt.ylabel("L2 Distance to Goal")
    plt.title("Distance to Goal State over Time")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Distance plot saved to {save_path}")
    plt.close()


def extract_video_vectors(model, transform, video_path, mode="single_frame", tau=16, device="cuda"):
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
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Sequence of state vectors (T, D).
    """
    vr = VideoReader(video_path)
    total_frames = len(vr)
    vectors = []

    print(f"Extracting vectors from {video_path} (frames: {total_frames}, mode: {mode})")

    indices = []
    if mode == "single_frame":
        indices = range(total_frames)
    elif mode == "clip":
        # Start from tau so we have a full clip [t-tau, t)
        indices = range(tau, total_frames + 1)

    for t in indices:
        if mode == "single_frame":
            # Extract single frame
            frame = vr[t].asnumpy()  # H, W, C
            # Duplicate to satisfy temporal dimension if needed (e.g. tubelet_size=2)
            clip = [frame, frame]
        elif mode == "clip":
            # Extract clip [t-tau, t)
            clip_indices = range(t - tau, t)
            clip = vr.get_batch(clip_indices).asnumpy()  # (tau, H, W, C)
            clip = list(clip)

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

        if t % 50 == 0:
            print(f"Processed {t}/{len(indices)}")

    if not vectors:
        return torch.empty(0)

    return torch.stack(vectors)  # (T, D)


def run_sample_inference():
    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitg-fpc64-384"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    # Path to local PyTorch weights
    pt_model_path = "/tmp2/hubertchang/p-progress/models/vitg-384.pt"

    sample_video_path = "sample_video.mp4"
    # Download the video if not yet downloaded to local path
    if not os.path.exists(sample_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        command = ["wget", video_url, "-O", sample_video_path]
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
        model_hf, model_pt, hf_transform, pt_video_transform
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
    vectors_single = extract_video_vectors(
        model_pt,
        pt_video_transform,
        sample_video_path,
        mode="single_frame",
        device="cuda",
    )
    plot_distance_to_goal(vectors_single, save_path="distance_single_frame.png")

    # Extract using clip mode
    vectors_clip = extract_video_vectors(
        model_pt,
        pt_video_transform,
        sample_video_path,
        mode="clip",
        tau=16,
        device="cuda",
    )
    plot_distance_to_goal(vectors_clip, save_path="distance_clip.png")


if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    run_sample_inference()
