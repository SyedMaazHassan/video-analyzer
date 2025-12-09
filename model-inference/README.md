# Surgical Phase Recognition - Inference Package

This package contains everything needed to run inference on new surgical videos using the trained BiLSTM-CRF model.

## 📦 Package Contents

```
nCight_inference/
├── inference_video.py          # Main inference script
├── model_utils.py              # Model class definitions (extracted from train_bilstm_crf.py)
├── models/
│   └── bilstm_crf/
│       └── best_model.pt       # Trained model checkpoint
├── video_features/             # Feature extraction library
│   ├── models/
│   ├── configs/
│   └── utils/
├── requirements_inference.txt  # Python dependencies
├── README.md                   # This file
└── example_usage.sh           # Example commands
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_inference.txt
```

**Note:** For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Inference

```bash
python inference_video.py \
    --video_path "path/to/your/video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --output_dir "./results"
```

## 📋 Required Files

### Minimum Required:
- ✅ `inference_video.py` - Main inference script
- ✅ `model_utils.py` - Model architecture definitions
- ✅ `models/bilstm_crf/best_model.pt` - Trained model checkpoint
- ✅ `video_features/` directory - Feature extraction library
- ✅ `requirements_inference.txt` - Dependencies

### Optional but Recommended:
- `data/training/phases.csv` - Label mapping (if not in checkpoint)
- `INFERENCE_GUIDE.md` - Detailed usage guide

## 🎯 Usage Examples

### Basic Inference
```bash
python inference_video.py \
    --video_path "./surgery_video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt"
```

### With Custom Output Directory
```bash
python inference_video.py \
    --video_path "./surgery_video.mp4" \
    --model_path "./models/bilstm_crf/best_model.pt" \
    --output_dir "./my_results"
```

### CPU-only Inference
The script automatically detects GPU availability. For CPU-only:
- Ensure CUDA is not available, or
- Modify device in script if needed

## 📊 Output Files

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

## 🔧 Troubleshooting

### Issue: "No frames read from video"
- **Solution**: Ensure video file path is correct and video is not corrupted
- Check video codec compatibility (MP4, AVI, MOV should work)

### Issue: "CUDA out of memory"
- **Solution**: Reduce batch size or use CPU
- Add `--batch_size 1` to command

### Issue: "Model checkpoint not found"
- **Solution**: Verify `--model_path` points to `best_model.pt`
- Check file permissions

### Issue: Import errors
- **Solution**: Ensure all dependencies are installed
- Try: `pip install -r requirements_inference.txt --upgrade`

## 📝 Model Information

- **Architecture**: BiLSTM-CRF
- **Input**: r21d features (512-dim)
- **Output**: 10 classes (9 surgical phases + background)
- **Best Validation Accuracy**: 77.88%

### Supported Phases:
1. Anchor Placement
2. Diagnostic Arthroscopy
3. Final Inspection
4. Glenoid Preparation
5. Instruments
6. Labral Mobilization
7. Portal Placement
8. Suture Passage
9. Suture Tensioning
10. background

## 💡 Tips

- **GPU Recommended**: Inference is ~10x faster on GPU
- **Video Format**: MP4 (H.264) works best
- **Processing Time**: ~1-2 minutes per minute of video (GPU)
- **Memory**: ~2-4 GB GPU memory for typical videos

## 📞 Support

For issues or questions:
1. Check `INFERENCE_GUIDE.md` for detailed documentation
2. Verify all dependencies are installed correctly
3. Ensure video file is accessible and not corrupted

## 📄 License

[Specify your license here]

