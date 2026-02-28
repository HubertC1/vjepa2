import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io

from app.vjepa_droid.transforms import make_transforms
from src.models.vision_transformer import vit_giant_xformers_rope


def find_mp4s(root: str) -> List[str]:
    mp4s = []
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

    if target_fps is None:
        indices = np.arange(0, vlen, 1, dtype=np.int64)
    else:
        video_fps = info.get("video_fps", None)
        if video_fps is None or video_fps <= 0:
            video_fps = target_fps
        frame_step = max(int(round(video_fps / target_fps)), 1)
        indices = np.arange(0, vlen, frame_step, dtype=np.int64)

    video = video[indices]  # [T', H, W, C]
    clip = transform(video)  # [C, T', H, W]
    return clip, indices


class Mp4Dataset(Dataset):
    def __init__(self, videos_root: str, transform, target_fps: float = None):
        self.videos_root = videos_root
        self.transform = transform
        self.target_fps = target_fps
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
            path, transform=self.transform, target_fps=self.target_fps
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract frozen VJEPA frame embeddings from MP4 videos using the DROID AC-predictor encoder."
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
    parser.add_argument("--batch_size", type=int, default=8, help="Number of videos per encoder forward.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for Decord loading.")
    parser.add_argument("--target_fps", type=float, default=None, help="If set, downsample videos to this fps.")
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
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Inference dtype for the encoder.",
    )
    args = parser.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    # Dataset + DataLoader over all MP4s
    dataset = Mp4Dataset(args.videos_root, transform=transform, target_fps=args.target_fps)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_padded,
    )

    # HDF5 output
    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)

    start_total = time.perf_counter()
    with h5py.File(args.output_h5, "a") as f:
        processed = 0
        for batch_idx, (clips, lengths, frame_indices, keys) in enumerate(loader):
            t0 = time.perf_counter()
            B = clips.shape[0]

            # Skip videos already in HDF5
            mask_keep = []
            keep_indices = []
            for i, key in enumerate(keys):
                if key in f:
                    mask_keep.append(False)
                else:
                    mask_keep.append(True)
                    keep_indices.append(i)

            if not any(mask_keep):
                continue

            # Run encoder on full batch [B, C, T, H, W]
            try:
                embs_bt = forward_target_batched(
                    clips, target_encoder=target_encoder, device=device,
                    crop_size=args.crop_size, patch_size=args.patch_size, dtype=dtype
                )  # [B, T, N_tokens, D]
            except Exception as e:
                print(f"Failed to encode batch {batch_idx}: {e}")
                continue

            embs_bt = embs_bt.cpu().numpy()  # [B, T, N_tokens, D]
            lengths_np = lengths

            # Save per-video
            for local_idx, key in enumerate(keys):
                if key in f:
                    continue
                T_i = int(lengths_np[local_idx])
                embs_i = embs_bt[local_idx, :T_i]  # [T_i, N_tokens, D]
                fi_i = frame_indices[local_idx, :T_i]

                grp = f.create_group(key)
                grp.create_dataset("embeddings", data=embs_i.astype(np.float32), compression="gzip")
                # When target_fps is set we downsample; embedding position i is from original frame fi_i[i].
                grp.create_dataset("frame_indices", data=fi_i.astype(np.int64), compression="gzip")
                processed += 1

            dt = time.perf_counter() - t0
            print(
                f"[batch {batch_idx+1}] saved {sum(mask_keep)} / {B} videos, "
                f"time={dt:.2f}s"
            )

    total_dt = time.perf_counter() - start_total
    print(f"Done. Processed {processed} videos in {total_dt/60.0:.1f} min. Output: {args.output_h5}")


if __name__ == "__main__":
    main()

