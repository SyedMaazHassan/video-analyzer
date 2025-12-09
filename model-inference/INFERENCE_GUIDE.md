# Inference Guide: Running Phase Recognition on New Videos

This guide explains how to use the trained BiLSTM-CRF model to predict surgical phases on new surgical videos.

## Quick Start

```bash
python inference_video.py \
    --video_path "path/to/your/video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --labels_csv "./data/training/phases.csv" \
    --output_dir "./inference_results"
```

## Requirements

- Trained model checkpoint (`best_model.pt`)
- Video file (`.mp4`, `.avi`, etc.)
- CUDA-enabled GPU (recommended) or CPU

## Arguments

### Required
- `--video_path`: Path to the input surgical video file
- `--model_path`: Path to the trained model checkpoint (e.g., `./models/bilstm_crf/best_model.pt`)

### Optional
- `--labels_csv`: Path to `phases.csv` (default: tries to load from checkpoint)
- `--output_dir`: Output directory for results (default: `./inference_results`)
- `--input_dim`: Feature dimension (default: 512 for r21d)
- `--hidden`: LSTM hidden dimension (default: 256)
- `--layers`: Number of LSTM layers (default: 2)
- `--batch_size`: Batch size for inference (default: 1)
- `--extraction_fps`: FPS for feature extraction (default: 1.0)
- `--smooth`: Apply smoothing to predictions (default: True)

## Output Files

The script generates three CSV files:

1. **`{video_name}_frame_predictions.csv`**
   - Frame-level predictions with timestamps
   - Columns: `frame`, `timestamp_s`, `phase_idx`, `phase`

2. **`{video_name}_phase_segments.csv`**
   - Phase segments with start/end times
   - Columns: `phase`, `start_time_s`, `end_time_s`, `duration_s`

3. **`{video_name}_summary.csv`**
   - Summary statistics per phase
   - Columns: `phase`, `frame_count`, `total_duration_s`, `percentage`

## Example Usage

### Basic inference:
```bash
python inference_video.py \
    --video_path "./new_surgery.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt"
```

### With custom settings:
```bash
python inference_video.py \
    --video_path "./new_surgery.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --labels_csv "./data/training/phases.csv" \
    --output_dir "./results/new_surgery" \
    --extraction_fps 2.0 \
    --smooth
```

## How It Works

1. **Feature Extraction**: 
   - Extracts r21d features from the video at specified FPS
   - Features are 512-dimensional vectors per frame

2. **Model Inference**:
   - Loads the trained BiLSTM-CRF model
   - Processes features through the model
   - Uses CRF decoding for sequence smoothing

3. **Post-processing**:
   - Applies optional smoothing (majority voting)
   - Converts frame-level predictions to phase segments
   - Generates summary statistics

4. **Output**:
   - Saves predictions to CSV files
   - Prints summary to console

## Expected Phases

The model predicts the following surgical phases:

1. Anchor Placement
2. Diagnostic Arthroscopy
3. Final Inspection
4. Glenoid Preparation
5. Instruments
6. Labral Mobilization
7. Portal Placement
8. Suture Passage
9. Suture Tensioning
10. background (unlabeled frames)

## Tips

- **Extraction FPS**: Lower FPS (0.5-1.0) = faster processing, less temporal resolution
- **Smoothing**: Helps reduce noise in predictions, especially for short transitions
- **GPU**: Use CUDA for faster feature extraction and inference
- **Video Format**: Supports common formats (MP4, AVI, MOV, etc.)

## Troubleshooting

### Out of Memory
- Reduce `--extraction_fps` (e.g., 0.5)
- Process shorter video segments

### Feature Dimension Mismatch
- Ensure `--input_dim` matches training (default: 512 for r21d)

### Model Not Found
- Check `--model_path` points to `best_model.pt`
- Verify checkpoint file exists

### No Labels CSV
- If `label_to_idx` is saved in checkpoint, `--labels_csv` is optional
- Otherwise, provide path to `phases.csv`

