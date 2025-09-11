#!/usr/bin/env python3
"""
SURGICAL MOTION ANALYSIS MODEL
=============================
Simple ResNet50-based model for surgical motion analysis.
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

class SurgicalMotionAnalyzer(nn.Module):
    """Simple motion analysis model matching training script architecture."""
    
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)  # Motion score
        )
        
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        if len(x.shape) == 5:
            # Compute motion between first and last frame
            B, C, T, H, W = x.shape
            first_frame = x[:, :, 0, :, :]
            last_frame = x[:, :, -1, :, :]
            
            # Simple motion estimation using frame difference
            motion_input = torch.abs(last_frame - first_frame)
        else:
            # Single frame input
            motion_input = x
        
        return self.backbone(motion_input)
    
    def analyze_video_motion(self, video_path: str, fps: float) -> List[Dict]:
        """Analyze motion patterns throughout the video."""
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
        
        # Sample every 450 frames (15 seconds at 30fps) - OPTIMIZED FOR DEMO
        sample_rate = 450
        motion_data = []
        frame_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert to grayscale for motion analysis
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference for motion estimation
                    frame_diff = cv2.absdiff(prev_frame, gray_frame)
                    motion_score = np.mean(frame_diff) / 255.0  # Normalize to 0-1
                    
                    # Convert back to RGB for model input
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(frame_rgb).unsqueeze(0)
                    
                    with torch.no_grad():
                        model_output = self.forward(input_tensor)
                        # Convert model output to motion score
                        model_motion_score = torch.sigmoid(model_output).item()
                    
                    # Combine simple motion detection with model output
                    combined_score = (motion_score + model_motion_score) / 2
                    
                    timestamp = frame_count / fps
                    
                    motion_data.append({
                        'frame': frame_count,
                        'timestamp_seconds': timestamp,
                        'timestamp_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        'motion_score': combined_score,
                        'activity_level': 'High' if combined_score > 0.5 else 'Medium' if combined_score > 0.2 else 'Low'
                    })
                
                prev_frame = gray_frame
            
            frame_count += 1
        
        cap.release()
        return motion_data

def create_motion_analyzer() -> SurgicalMotionAnalyzer:
    """Factory function to create motion analyzer."""
    return SurgicalMotionAnalyzer()