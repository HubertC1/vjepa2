python -m app.vjepa_droid.extract_jepa_vectors \
    --videos_root /home/hubertchang/p-progress/lerobot-libero/notebooks/videos/libero_spatial \
    --output_h5 /home/hubertchang/p-progress/lerobot-libero/notebooks/videos/libero_spatial/output_256.h5 \
    --vjepa_checkpoint /tmp2/hubertchang/p-jepa/models/vitg-384.pt \
    --device cuda:0 \
    --batch_size 1 \
    --num_workers 1 \
    --dtype float32 \
    --target_fps 15 \
    --smoothing_bandwidth 0.02 \
    --time_threshold_seconds 1 \
    --monotonicity_threshold -0.99 \
    --videos_per_file 500 \
    --min_video_seconds 4 \
    --max_video_seconds 40 \
    --debug \
    --debug_num_videos 10 \
    --debug_start_video_index 0 \
    --debug_output_dir /home/hubertchang/p-progress/lerobot-libero/notebooks/videos/libero_spatial/debug \


    # --videos_root /tmp2/hubertchang/p-jepa/data/droid/droid_repo/videos \
    # --output_h5 /tmp2/hubertchang/p-jepa/data/droid/jepa_rep/jepa_vectors_smooth02_test.h5 \