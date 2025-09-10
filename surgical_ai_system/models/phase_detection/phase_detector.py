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
from typing import Dict, List, Optional, Tuple
import numpy as np

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

def create_phase_detector(num_phases: int = 8) -> AdvancedPhaseDetector:
    """Factory function to create phase detector."""
    return AdvancedPhaseDetector(num_phases)