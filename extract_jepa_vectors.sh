python -m app.vjepa_droid.extract_jepa_vectors \
    --videos_root /tmp2/hubertchang/p-jepa/data/droid/droid_repo/videos \
    --output_h5 /tmp2/hubertchang/p-jepa/data/droid/jepa_rep/jepa_vectors_test.h5 \
    --vjepa_checkpoint /tmp2/hubertchang/p-jepa/models/vitg-384.pt \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers 1 \
    --dtype float32 \
    --videos_per_file 5