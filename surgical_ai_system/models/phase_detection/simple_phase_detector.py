#!/usr/bin/env python3
"""
Simple Phase Detection Model
============================
Matches the exact architecture used in training.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50

class AdvancedPhaseDetector(nn.Module):
    """Simple phase detector matching training script architecture."""
    
    def __init__(self, num_phases: int = 8):
        super().__init__()
        
        self.num_phases = num_phases
        
        # Exact same architecture as training script
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_phases)
        )
    
    def forward(self, x):
        """Simple forward pass matching training script."""
        if len(x.shape) == 5:  # [B, C, T, H, W]
            # Handle temporal input like training script
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
            x = x.view(B * T, C, H, W)  # Flatten temporal dimension
            
            features = self.backbone(x)  # [B*T, num_classes]
            features = features.view(B, T, -1)  # [B, T, num_classes]
            features = features.mean(dim=1)  # Average over time
            
            return features
        else:
            # Handle single frame input
            return self.backbone(x)
    
    def analyze_video_phases(self, video_path, fps):
        """Placeholder for video analysis - matches expected interface."""
        # This would be implemented for actual video analysis
        return []