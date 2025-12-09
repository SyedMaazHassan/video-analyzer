"""
inference_video.py

Run inference on a new surgical video to predict surgical phases.

Usage:
  python inference_video.py --video_path "path/to/video.mp4" --model_path "./models/bilstm_crf/best_model.pt" --output_dir "./inference_results"
"""
import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
from pathlib import Path
import cv2

# Add video_features to path
VIDEO_FEATURES_DIR = Path(__file__).parent / "video_features"
sys.path.insert(0, str(VIDEO_FEATURES_DIR))

# Change to video_features directory for relative paths to work
original_cwd = os.getcwd()
os.chdir(VIDEO_FEATURES_DIR)

from utils.utils import build_cfg_path
from models.r21d.extract_r21d import ExtractR21D
from omegaconf import OmegaConf

# Change back to original directory
os.chdir(original_cwd)

# Try to import from model_utils first (for standalone package), fallback to train_bilstm_crf
try:
    from model_utils import BiLSTMCRF, get_label_mapping
except ImportError:
    from train_bilstm_crf import BiLSTMCRF, get_label_mapping

def extract_video_features(video_path, device="cuda:0", extraction_fps=1.0):
    """
    Extract r21d features from a video file.
    
    Returns:
        features: (T, 512) numpy array
        timestamps: (T,) numpy array of timestamps in seconds
        fps: video FPS
    """
    # Convert to absolute path to avoid issues when changing directories
    video_path_abs = Path(video_path).resolve()
    print(f"\n🎬 Extracting r21d features from: {video_path_abs}")
    
    # Get video FPS first (before any re-encoding or directory changes)
    cap = cv2.VideoCapture(str(video_path_abs))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path_abs}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames == 0:
        raise ValueError(f"Video file appears to be empty or corrupted: {video_path_abs}")
    
    print(f"   Video info: {total_frames} frames @ {video_fps:.2f} FPS")
    
    # Change to video_features directory
    os.chdir(VIDEO_FEATURES_DIR)
    
    try:
        # Load r21d config
        config_path = build_cfg_path('r21d')
        args = OmegaConf.load(config_path)
        
        # Override settings
        args.device = device
        args.on_extraction = 'print'  # We'll capture the output
        # Set extraction_fps to None to use original video FPS (avoid re-encoding issues)
        # r21d processes frames in stacks (16 frames per stack), so it naturally samples
        args.extraction_fps = None  # Use original FPS to avoid re-encoding problems
        args.keep_tmp_files = False
        args.show_pred = False
        
        # Initialize extractor
        extractor = ExtractR21D(args)
        
        # Extract features - use absolute path
        feats_dict = extractor.extract(str(video_path_abs))
        
        # r21d returns features under 'r21d' key
        if 'r21d' in feats_dict:
            features = feats_dict['r21d']
        else:
            raise ValueError("No r21d features found in extraction output")
        
        # Calculate timestamps based on r21d's stack processing
        # r21d processes stacks of 16 frames with step_size 16
        # Each feature corresponds to a stack, so timestamps are at stack centers
        stack_size = extractor.stack_size  # Usually 16
        step_size = extractor.step_size    # Usually 16
        
        # Calculate effective feature FPS
        feature_fps = video_fps / step_size  # Features per second
        
        # Generate timestamps: each feature represents a stack centered at its time
        # First feature is at time (stack_size/2) / video_fps
        # Subsequent features are spaced by step_size / video_fps
        timestamps = np.arange(len(features)) * (step_size / video_fps) + (stack_size / 2 / video_fps)
        
        print(f"✅ Extracted {len(features)} feature vectors")
        print(f"   Feature shape: {features.shape}")
        print(f"   Video FPS: {video_fps:.2f}")
        print(f"   Extraction FPS: {extraction_fps}")
        print(f"   Video duration: {timestamps[-1]:.2f}s")
        
        return features, timestamps, video_fps
        
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

def load_model(model_path, device, input_dim=512, hidden_dim=256, n_layers=2, n_classes=10):
    """Load the trained model from checkpoint."""
    print(f"\n📥 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = BiLSTMCRF(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_classes=n_classes
    )

    # Map state dict keys from pytorch-crf to TorchCRF format
    state_dict = checkpoint['model_state_dict']
    key_mapping = {
        'crf.transitions': 'crf.trans_matrix',
        'crf.start_transitions': 'crf.start_trans',
        'crf.end_transitions': 'crf.end_trans',
    }

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key_mapping.get(key, key)
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    if 'label_to_idx' in checkpoint:
        label_to_idx = checkpoint['label_to_idx']
    else:
        # Fallback: need to load from CSV
        print("⚠️  label_to_idx not found in checkpoint, will need labels_csv")
        label_to_idx = None

    print("✅ Model loaded successfully")
    return model, label_to_idx, checkpoint

def predict_phases(model, features, device, batch_size=1):
    """
    Predict phases for a sequence of features.
    
    Args:
        model: Trained BiLSTM-CRF model
        features: (T, D) numpy array of features
        device: torch device
        batch_size: batch size for inference
    
    Returns:
        predictions: (T,) numpy array of predicted phase indices
    """
    print(f"\n🔮 Running inference on {len(features)} frames...")
    
    model.eval()
    all_predictions = []
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, T, D)
    
    with torch.no_grad():
        # Create mask (all True since we have full sequence)
        mask = torch.ones(1, features_tensor.shape[1], dtype=torch.bool)
        
        features_tensor = features_tensor.to(device)
        mask = mask.to(device)
        
        # Forward pass
        emissions = model(features_tensor, mask)
        mask_t = mask.transpose(0, 1)
        
        # Decode with CRF
        if hasattr(model.crf, 'decode'):
            predictions = model.crf.decode(emissions, mask=mask_t)
            predictions = predictions[0]  # Get first (and only) batch item
        else:
            # Fallback: use argmax
            emissions_batch = emissions.transpose(0, 1)
            predictions = emissions_batch.argmax(dim=-1)[0].cpu().numpy()

    print(f"Predicted phases for {len(predictions)} frames")
    return np.array(predictions)

def smooth_predictions(predictions, window_size=5):
    """
    Apply simple smoothing to predictions using majority voting in sliding window.
    """
    if window_size <= 1:
        return predictions
    
    smoothed = np.zeros_like(predictions)
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        window = predictions[start:end]
        smoothed[i] = np.bincount(window).argmax()
    
    return smoothed

def create_phase_segments(predictions, timestamps, idx_to_label):
    """
    Convert frame-level predictions to phase segments.
    
    Returns:
        segments: List of dicts with 'phase', 'start_time', 'end_time', 'duration'
    """
    segments = []
    current_phase = predictions[0]
    start_time = timestamps[0]
    
    for i in range(1, len(predictions)):
        if predictions[i] != current_phase:
            # Phase changed
            segments.append({
                'phase': idx_to_label[current_phase],
                'start_time_s': start_time,
                'end_time_s': timestamps[i-1],
                'duration_s': timestamps[i-1] - start_time
            })
            current_phase = predictions[i]
            start_time = timestamps[i]
    
    # Add final segment
    segments.append({
        'phase': idx_to_label[current_phase],
        'start_time_s': start_time,
        'end_time_s': timestamps[-1],
        'duration_s': timestamps[-1] - start_time
    })
    
    return segments

def save_results(predictions, timestamps, idx_to_label, output_dir, video_name, smooth=True):
    """Save inference results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply smoothing if requested
    if smooth:
        predictions_smooth = smooth_predictions(predictions, window_size=5)
    else:
        predictions_smooth = predictions
    
    # Frame-level predictions
    frame_df = pd.DataFrame({
        'frame': np.arange(len(predictions)),
        'timestamp_s': timestamps,
        'phase_idx': predictions_smooth,
        'phase': [idx_to_label[p] for p in predictions_smooth]
    })
    
    frame_csv = os.path.join(output_dir, f"{video_name}_frame_predictions.csv")
    frame_df.to_csv(frame_csv, index=False)
    print(f"💾 Frame-level predictions saved to: {frame_csv}")
    
    # Phase segments
    segments = create_phase_segments(predictions_smooth, timestamps, idx_to_label)
    segments_df = pd.DataFrame(segments)
    
    segments_csv = os.path.join(output_dir, f"{video_name}_phase_segments.csv")
    segments_df.to_csv(segments_csv, index=False)
    print(f"💾 Phase segments saved to: {segments_csv}")
    
    # Summary statistics
    phase_counts = pd.Series([idx_to_label[p] for p in predictions_smooth]).value_counts()
    phase_durations = segments_df.groupby('phase')['duration_s'].sum()
    
    summary_df = pd.DataFrame({
        'phase': phase_counts.index,
        'frame_count': phase_counts.values,
        'total_duration_s': [phase_durations.get(p, 0) for p in phase_counts.index],
        'percentage': (phase_counts.values / len(predictions_smooth) * 100).round(2)
    })
    
    summary_csv = os.path.join(output_dir, f"{video_name}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"💾 Summary statistics saved to: {summary_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 PREDICTION SUMMARY")
    print("="*80)
    print(f"Total duration: {timestamps[-1]:.2f}s ({timestamps[-1]/60:.2f} minutes)")
    print(f"Total frames: {len(predictions)}")
    print(f"\nPhase distribution:")
    for _, row in summary_df.iterrows():
        print(f"  {row['phase']:<30} {row['total_duration_s']:>8.2f}s ({row['percentage']:>5.2f}%)")
    print("="*80)
    
    return frame_csv, segments_csv, summary_csv

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get video name
    video_path = Path(args.video_path)
    video_name = video_path.stem
    
    # Load label mapping
    if args.labels_csv:
        label_to_idx = get_label_mapping(args.labels_csv)
    else:
        # Try to get from checkpoint
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'label_to_idx' in checkpoint:
            label_to_idx = checkpoint['label_to_idx']
        else:
            raise ValueError("labels_csv required if not in checkpoint")
    
    n_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    print(f"\n📋 Label mapping ({n_classes} classes):")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    
    # Extract features from video
    features, timestamps, video_fps = extract_video_features(
        args.video_path, 
        device=str(device),
        extraction_fps=args.extraction_fps
    )
    
    # Load model
    model, _, _ = load_model(
        args.model_path,
        device,
        input_dim=args.input_dim,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_classes=n_classes
    )
    
    # Predict phases
    predictions = predict_phases(model, features, device, batch_size=args.batch_size)
    
    # Save results
    save_results(
        predictions, 
        timestamps, 
        idx_to_label, 
        args.output_dir, 
        video_name,
        smooth=args.smooth
    )
    
    print(f"\n✅ Inference complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a surgical video")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--labels_csv", default=None, help="Path to phases.csv (optional if in checkpoint)")
    parser.add_argument("--output_dir", default="./inference_results", help="Output directory for results")
    parser.add_argument("--input_dim", type=int, default=512, help="Input feature dimension")
    parser.add_argument("--hidden", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--extraction_fps", type=float, default=1.0, help="FPS for feature extraction")
    parser.add_argument("--smooth", action="store_true", default=True, help="Apply smoothing to predictions")
    args = parser.parse_args()
    
    main(args)

