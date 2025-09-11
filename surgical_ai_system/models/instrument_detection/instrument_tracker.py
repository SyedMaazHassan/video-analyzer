#!/usr/bin/env python3
"""
SURGICAL INSTRUMENT DETECTION MODEL
==================================
Simple ResNet50-based model for surgical instrument detection.
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

class InstrumentTracker(nn.Module):
    """Simple instrument detection model matching training script architecture."""
    
    def __init__(self, num_instruments=14):
        super().__init__()
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_instruments)
        )
        
    def forward(self, x):
        # x shape: [B, C, T, H, W] or [B, C, H, W]
        if len(x.shape) == 5:
            # Use only the last frame for instrument detection
            B, C, T, H, W = x.shape
            x = x[:, :, -1, :, :]  # [B, C, H, W]
        
        return self.backbone(x)
    
    def track_video_instruments(self, video_path: str, fps: float) -> List[Dict]:
        """Track instruments throughout the video."""
        # Instrument labels matching the training data
        instrument_labels = [
            "Arthroscopic Camera", "Trocar", "Cannula", "Shaver", 
            "Electrocautery Probe", "Probe", "Grasper", "Burr", 
            "Rasp", "Drill Guide", "Suture Anchor", "Suture Passer", 
            "Knot Pusher", "Suture Cutter"
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
        
        # Sample every 450 frames (15 seconds at 30fps) - OPTIMIZED FOR DEMO
        sample_rate = 450
        detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.forward(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    # Only include detections with reasonable confidence
                    if confidence > 0.3:
                        predicted_instrument = instrument_labels[predicted_class]
                        timestamp = frame_count / fps
                        
                        detections.append({
                            'frame': frame_count,
                            'timestamp_seconds': timestamp,
                            'timestamp_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                            'detected_instrument': predicted_instrument,
                            'confidence': confidence
                        })
            
            frame_count += 1
        
        cap.release()
        return detections

def create_instrument_tracker(num_instruments: int = 14) -> InstrumentTracker:
    """Factory function to create instrument tracker."""
    return InstrumentTracker(num_instruments)