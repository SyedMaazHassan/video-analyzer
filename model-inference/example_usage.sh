#!/bin/bash
# Example usage scripts for inference

# Basic inference
python inference_video.py \
    --video_path "./example_video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --output_dir "./results"

# With labels CSV (if not in checkpoint)
python inference_video.py \
    --video_path "./example_video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --labels_csv "./data/training/phases.csv" \
    --output_dir "./results"

# Custom settings
python inference_video.py \
    --video_path "./example_video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --output_dir "./results" \
    --batch_size 1 \
    --smooth
