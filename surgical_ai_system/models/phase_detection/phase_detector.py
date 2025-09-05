#!/usr/bin/env python3
"""
SURGICAL PHASE DETECTION MODEL
=============================
Advanced temporal-aware model for precise surgical phase detection.

Features:
- EfficientNet backbone for visual features
- LSTM for temporal consistency
- Multi-scale attention mechanism
- Phase transition smoothing
- Confidence calibration

Author: AI Surgical Analysis Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing import Dict, List, Optional, Tuple
import numpy as np

class TemporalAttention(nn.Module):
    """Multi-head attention for temporal feature modeling."""
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, sequence_length: int) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Reshape for multi-head attention
        Q = self.query(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.feature_dim
        )
        
        return self.output(attended)

class PhaseTransitionSmoother(nn.Module):
    """Smooth phase transitions using temporal consistency."""
    
    def __init__(self, num_phases: int, smoothing_window: int = 5):
        super().__init__()
        self.num_phases = num_phases
        self.smoothing_window = smoothing_window
        
        # Learnable transition matrix
        self.transition_matrix = nn.Parameter(torch.eye(num_phases) * 0.8 + 0.02)
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to predictions."""
        batch_size, seq_len, num_classes = predictions.shape
        
        # Apply exponential smoothing
        smoothed = torch.zeros_like(predictions)
        smoothed[:, 0] = predictions[:, 0]
        
        alpha = 0.7  # Smoothing factor
        for t in range(1, seq_len):
            smoothed[:, t] = alpha * predictions[:, t] + (1 - alpha) * smoothed[:, t-1]
        
        return smoothed

class AdvancedPhaseDetector(nn.Module):
    """Advanced surgical phase detection with temporal modeling."""
    
    def __init__(self, 
                 num_phases: int = 8,
                 feature_dim: int = 512,
                 lstm_hidden: int = 256,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_phases = num_phases
        self.feature_dim = feature_dim
        
        # Visual feature extractor - ResNet50 to match trained model
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_phases)
        )
        
        # Temporal modeling layers
        self.temporal_attention = TemporalAttention(feature_dim)
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        lstm_output_dim = lstm_hidden * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_phases)
        )
        
        # Phase transition smoother
        self.transition_smoother = PhaseTransitionSmoother(num_phases)
        
        # Confidence calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(num_phases, num_phases * 2),
            nn.ReLU(),
            nn.Linear(num_phases * 2, num_phases)
        )
        
    def forward(self, 
                x: torch.Tensor, 
                sequence_mode: bool = False,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple output modes.
        
        Args:
            x: Input tensor [batch_size, channels, height, width] or 
               [batch_size, sequence_length, channels, height, width]
            sequence_mode: Whether input is a sequence of frames
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and optional features
        """
        
        if sequence_mode:
            batch_size, seq_len = x.shape[:2]
            
            # Reshape for backbone processing
            x_flat = x.view(-1, *x.shape[2:])
            visual_features = self.backbone(x_flat)
            visual_features = self.feature_adapter(visual_features)
            
            # Reshape back to sequence
            visual_features = visual_features.view(batch_size, seq_len, self.feature_dim)
            
            # Apply temporal attention
            attended_features = self.temporal_attention(visual_features, seq_len)
            
            # LSTM processing
            lstm_output, (hidden, cell) = self.lstm(attended_features)
            
            # Classification
            predictions = self.classifier(lstm_output)
            
            # Apply phase transition smoothing
            smoothed_predictions = self.transition_smoother(predictions)
            
            # Confidence calibration
            calibrated_predictions = self.confidence_calibrator(smoothed_predictions)
            
            # Convert to probabilities
            probabilities = F.softmax(calibrated_predictions, dim=-1)
            
            output = {
                'predictions': calibrated_predictions,
                'probabilities': probabilities,
                'smoothed_predictions': smoothed_predictions
            }
            
            if return_features:
                output['visual_features'] = visual_features
                output['attended_features'] = attended_features
                output['lstm_features'] = lstm_output
            
        else:
            # Single frame processing
            visual_features = self.backbone(x)
            adapted_features = self.feature_adapter(visual_features)
            
            # Simple classification for single frame
            # Add temporal dimension for LSTM
            lstm_input = adapted_features.unsqueeze(1)
            lstm_output, _ = self.lstm(lstm_input)
            predictions = self.classifier(lstm_output.squeeze(1))
            
            probabilities = F.softmax(predictions, dim=-1)
            
            output = {
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            if return_features:
                output['visual_features'] = visual_features
                output['adapted_features'] = adapted_features
        
        return output
    
    def predict_phase_sequence(self, 
                             video_frames: torch.Tensor,
                             confidence_threshold: float = 0.7) -> List[Dict]:
        """
        Predict phase sequence for entire video.
        
        Args:
            video_frames: Tensor of shape [num_frames, channels, height, width]
            confidence_threshold: Minimum confidence for phase detection
            
        Returns:
            List of phase predictions with metadata
        """
        self.eval()
        
        with torch.no_grad():
            # Process in sequence mode
            frames_batch = video_frames.unsqueeze(0)  # Add batch dimension
            
            output = self.forward(frames_batch, sequence_mode=True, return_features=True)
            probabilities = output['probabilities'].squeeze(0)  # Remove batch dimension
            
            phase_sequence = []
            
            for frame_idx, frame_probs in enumerate(probabilities):
                max_prob, predicted_class = torch.max(frame_probs, dim=0)
                
                if max_prob >= confidence_threshold:
                    phase_sequence.append({
                        'frame': frame_idx,
                        'phase_id': predicted_class.item(),
                        'confidence': max_prob.item(),
                        'all_probabilities': frame_probs.cpu().numpy().tolist()
                    })
            
            return phase_sequence
    
    def detect_phase_transitions(self, 
                               phase_sequence: List[Dict],
                               min_phase_duration: float = 5.0,
                               fps: float = 30.0) -> List[Dict]:
        """
        Detect and validate phase transitions.
        
        Args:
            phase_sequence: Output from predict_phase_sequence
            min_phase_duration: Minimum duration for a valid phase (seconds)
            fps: Video frame rate
            
        Returns:
            List of validated phase segments
        """
        if not phase_sequence:
            return []
        
        min_frames = int(min_phase_duration * fps)
        phase_segments = []
        current_phase = phase_sequence[0]['phase_id']
        current_start = 0
        
        for i, prediction in enumerate(phase_sequence[1:], 1):
            if prediction['phase_id'] != current_phase:
                # Phase transition detected
                duration_frames = i - current_start
                
                if duration_frames >= min_frames:
                    # Valid phase duration
                    phase_segments.append({
                        'phase_id': current_phase,
                        'start_frame': current_start,
                        'end_frame': i - 1,
                        'duration_frames': duration_frames,
                        'duration_seconds': duration_frames / fps,
                        'avg_confidence': np.mean([
                            p['confidence'] for p in phase_sequence[current_start:i]
                        ])
                    })
                
                current_phase = prediction['phase_id']
                current_start = i
        
        # Handle last phase
        final_duration = len(phase_sequence) - current_start
        if final_duration >= min_frames:
            phase_segments.append({
                'phase_id': current_phase,
                'start_frame': current_start,
                'end_frame': len(phase_sequence) - 1,
                'duration_frames': final_duration,
                'duration_seconds': final_duration / fps,
                'avg_confidence': np.mean([
                    p['confidence'] for p in phase_sequence[current_start:]
                ])
            })
        
        return phase_segments

class PhaseDetectorLoss(nn.Module):
    """Specialized loss function for phase detection."""
    
    def __init__(self, 
                 num_phases: int,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 temporal_weight: float = 0.1):
        super().__init__()
        self.num_phases = num_phases
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.temporal_weight = temporal_weight
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                sequence_mode: bool = False) -> torch.Tensor:
        """
        Compute specialized loss for phase detection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            sequence_mode: Whether processing sequences
            
        Returns:
            Combined loss value
        """
        # Focal loss for class imbalance
        ce_loss = F.cross_entropy(predictions.view(-1, self.num_phases), 
                                 targets.view(-1), reduction='none')
        
        # Compute focal loss weights
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        primary_loss = focal_loss.mean()
        
        # Add temporal consistency loss for sequences
        if sequence_mode and predictions.dim() == 3:
            temporal_loss = self._compute_temporal_loss(predictions)
            total_loss = primary_loss + self.temporal_weight * temporal_loss
        else:
            total_loss = primary_loss
        
        return total_loss
    
    def _compute_temporal_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Penalize rapid phase changes
        diff = predictions[:, 1:] - predictions[:, :-1]
        temporal_loss = torch.mean(torch.norm(diff, p=2, dim=-1))
        return temporal_loss

# ==================== MODEL FACTORY ====================

def create_phase_detector(config: Dict) -> AdvancedPhaseDetector:
    """Factory function to create phase detector from config."""
    
    model_config = config.get('phase_detector', {})
    
    return AdvancedPhaseDetector(
        num_phases=model_config.get('num_phases', 8),
        feature_dim=model_config.get('feature_dim', 512),
        lstm_hidden=model_config.get('lstm_hidden', 256),
        num_lstm_layers=model_config.get('num_lstm_layers', 2),
        dropout=model_config.get('dropout', 0.3)
    )

if __name__ == "__main__":
    # Test the model
    model = AdvancedPhaseDetector()
    
    # Single frame test
    single_frame = torch.randn(1, 3, 224, 224)
    single_output = model(single_frame, sequence_mode=False)
    print(f"Single frame output shape: {single_output['predictions'].shape}")
    
    # Sequence test
    sequence_frames = torch.randn(1, 10, 3, 224, 224)
    sequence_output = model(sequence_frames, sequence_mode=True)
    print(f"Sequence output shape: {sequence_output['predictions'].shape}")
    
    print("Phase detector model test completed successfully!")