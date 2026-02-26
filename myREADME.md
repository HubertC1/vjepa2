cd /home/hubertchang/p-jepa/vjepa2

conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312

# Install torch/torchvision for H100 from PyTorch site (example, adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements2.txt

# Finally, install the repo package itself
pip install -e .

download jepa model:
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P YOUR_DIR

python download_then_extract_droid.py \
  --cache-dir /tmp2/hubertchang/p-progress/data/droid \
  --output-dir /tmp2/hubertchang/p-progress/data/droid/subgoal \
  --pt_model_path /tmp2/hubertchang/p-progress/models/vitg-384.pt \
  --device cuda:0 \
  --video_batch_size 16 \
  --skip_chunk_if_exists



