#!/usr/bin/env python3
"""
SURGICAL PHASE DETECTION MODEL
=============================
Simple ResNet50-based model for surgical phase detection.
Architecture matches training script exactly.

Author: AI Surgical Analysis Team
Version: 2.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

class AdvancedPhaseDetector(nn.Module):
    """Simple phase detection model matching training script architecture."""
    
    def __init__(self, num_phases=8):
        super().__init__()
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_phases)
        )
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x):
        # x shape: [B, C, T, H, W] or [B, C, H, W]
        if len(x.shape) == 5:
            # Temporal input
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
            x = x.view(B * T, C, H, W)  # Flatten temporal dimension
            
            features = self.backbone(x)  # [B*T, num_classes]
            features = features.view(B, T, -1)  # [B, T, num_classes]
            features = features.mean(dim=1)  # Average over time
        else:
            # Single frame input
            features = self.backbone(x)
        
        return features
    
    def analyze_video_phases(self, video_path: str, fps: float, progress_callback=None) -> List[Dict]:
        """Analyze video to detect surgical phases."""
        # Phase labels matching the training data
        phase_labels = [
            "Portal Placement", "Diagnostic Arthroscopy", "Labral Mobilization",
            "Glenoid Preparation", "Anchor Placement", "Suture Passage", 
            "Suture Tensioning", "Final Inspection"
        ]
        
        # Transform for model input
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 300 frames (10 seconds at 30fps) - OPTIMIZED FOR DEMO
        sample_rate = 300
        predictions = []
        frame_count = 0
        processed_frames = 0
        
        print(f"Starting phase analysis of {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                processed_frames += 1
                if processed_frames % 10 == 0:  # Progress every 10 samples
                    progress = (frame_count / total_frames) * 100
                    print(f"Phase analysis progress: {progress:.1f}% ({processed_frames} samples processed)")
                    if progress_callback:
                        progress_callback(f"🔍 Phase Detection: {progress:.1f}%")
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.forward(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    predicted_phase = phase_labels[predicted_class]
                    timestamp = frame_count / fps
                    
                    predictions.append({
                        'frame': frame_count,
                        'timestamp_seconds': timestamp,
                        'timestamp_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        'predicted_phase': predicted_phase,
                        'confidence': confidence
                    })
            
            frame_count += 1
        
        cap.release()
        return predictions

def create_phase_detector(num_phases: int = 8) -> AdvancedPhaseDetector:
    """Factory function to create phase detector."""
    return AdvancedPhaseDetector(num_phases)