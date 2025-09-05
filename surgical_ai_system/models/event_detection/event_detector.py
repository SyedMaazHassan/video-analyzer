#!/usr/bin/env python3
"""
SURGICAL EVENT DETECTION MODEL
=============================
Specialized model for detecting critical surgical events:
- Bleeding (Mild/Moderate/Severe + Controlled status)
- Suture Attempts (Success/Fail + Reason + Anchor Number)
- Anchor Events (Reposition/Pullout + Reason)
- Custom Events (Device Failure, Workflow Deviation, etc.)

Author: AI Surgical Analysis Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b0
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum

class EventType(Enum):
    """Surgical event types."""
    BLEEDING = "bleeding"
    SUTURE_ATTEMPT = "suture_attempt"
    ANCHOR_REPOSITION = "anchor_reposition"
    ANCHOR_PULLOUT = "anchor_pullout"
    CUSTOM_EVENT = "custom_event"

class BleedingSeverity(Enum):
    """Bleeding severity levels."""
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"

class SutureOutcome(Enum):
    """Suture attempt outcomes."""
    SUCCESS = "Success"
    FAIL = "Fail"

class SutureFailureReason(Enum):
    """Reasons for suture failure."""
    LOOP_MISSED = "Loop Missed"
    TORN_SUTURE = "Torn Suture"
    DEVICE_MISFIRE = "Device Misfire"
    OTHER = "Other"
    NONE = "None"

class RepositionReason(Enum):
    """Anchor reposition reasons."""
    ANGLE = "Angle"
    DEPTH = "Depth"
    POSITION = "Position"
    UNKNOWN = "Unknown"

class CustomEventCategory(Enum):
    """Custom event categories."""
    DEVICE_FAILURE = "Device Failure"
    ANCHOR_PULLOUT = "Anchor Pullout"
    ANCHOR_REPOSITION_MULTIPLE = "Anchor Reposition (Multiple)"
    BAILOUT_STRATEGY = "Bailout Strategy"
    CUSTOM_INSTRUMENT_USED = "Custom Instrument Used"
    OFF_AXIS_ANCHOR = "Off-Axis Anchor"
    PROLONGED_SUTURE_PASSAGE = "Prolonged Suture Passage"
    RETRIEVAL_ATTEMPT = "Retrieval Attempt"
    WORKFLOW_DEVIATION = "Workflow Deviation"
    UNEXPECTED_BLEEDING = "Unexpected Bleeding"
    OTHER = "Other"

# ==================== MULTI-BRANCH EVENT DETECTOR ====================

class MultiBranchEventDetector(nn.Module):
    """Multi-branch model for detecting different types of surgical events."""
    
    def __init__(self, 
                 backbone: str = "resnet50",
                 feature_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Backbone feature extractor
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            backbone_features = 1000
        elif backbone == "efficientnet":
            self.backbone = efficientnet_b0(pretrained=True)
            backbone_features = 1000
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature adapter
        self.feature_adapter = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Bleeding Detection Branch
        self.bleeding_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(BleedingSeverity) + 1)  # +1 for no bleeding
        )
        
        # Bleeding Control Status Branch
        self.bleeding_control = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Controlled / Not Controlled
        )
        
        # Suture Attempt Detection Branch
        self.suture_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Suture attempt / No suture attempt
        )
        
        # Suture Outcome Branch
        self.suture_outcome = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(SutureOutcome))
        )
        
        # Suture Failure Reason Branch
        self.suture_reason = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(SutureFailureReason))
        )
        
        # Anchor Event Detection Branch
        self.anchor_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Normal, Reposition, Pullout
        )
        
        # Anchor Reposition Reason Branch
        self.anchor_reason = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(RepositionReason))
        )
        
        # Custom Event Detection Branch
        self.custom_event_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(CustomEventCategory))
        )
        
        # Attention mechanism for feature weighting
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-branch event detection.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing all branch predictions
        """
        batch_size = x.size(0)
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Adapt features
        adapted_features = self.feature_adapter(backbone_features)
        
        # Apply self-attention for better feature representation
        attended_features, attention_weights = self.attention(
            adapted_features.unsqueeze(1), 
            adapted_features.unsqueeze(1), 
            adapted_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Branch predictions
        predictions = {}
        
        # Bleeding detection and analysis
        bleeding_pred = self.bleeding_detector(attended_features)
        bleeding_control_pred = self.bleeding_control(attended_features)
        
        predictions['bleeding'] = {
            'severity': bleeding_pred,
            'control_status': bleeding_control_pred
        }
        
        # Suture attempt detection and analysis
        suture_detection_pred = self.suture_detector(attended_features)
        suture_outcome_pred = self.suture_outcome(attended_features)
        suture_reason_pred = self.suture_reason(attended_features)
        
        predictions['suture'] = {
            'detection': suture_detection_pred,
            'outcome': suture_outcome_pred,
            'failure_reason': suture_reason_pred
        }
        
        # Anchor event detection and analysis
        anchor_detection_pred = self.anchor_detector(attended_features)
        anchor_reason_pred = self.anchor_reason(attended_features)
        
        predictions['anchor'] = {
            'event_type': anchor_detection_pred,
            'reposition_reason': anchor_reason_pred
        }
        
        # Custom event detection
        custom_event_pred = self.custom_event_detector(attended_features)
        predictions['custom_event'] = {
            'category': custom_event_pred
        }
        
        if return_features:
            predictions['features'] = {
                'backbone': backbone_features,
                'adapted': adapted_features,
                'attended': attended_features,
                'attention_weights': attention_weights
            }
        
        return predictions

    def predict_events(self, 
                      x: torch.Tensor, 
                      confidence_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Predict events with confidence filtering.
        
        Args:
            x: Input tensor
            confidence_thresholds: Confidence thresholds for each event type
            
        Returns:
            Filtered predictions with metadata
        """
        if confidence_thresholds is None:
            confidence_thresholds = {
                'bleeding': 0.7,
                'suture': 0.8,
                'anchor': 0.75,
                'custom_event': 0.6
            }
        
        self.eval()
        with torch.no_grad():
            raw_predictions = self.forward(x)
            
            filtered_predictions = {}
            
            # Process bleeding predictions
            bleeding_probs = F.softmax(raw_predictions['bleeding']['severity'], dim=1)
            bleeding_max_prob, bleeding_class = torch.max(bleeding_probs, dim=1)
            
            control_probs = F.softmax(raw_predictions['bleeding']['control_status'], dim=1)
            control_pred = torch.argmax(control_probs, dim=1)
            
            bleeding_detected = bleeding_max_prob >= confidence_thresholds['bleeding']
            filtered_predictions['bleeding'] = []
            
            for i in range(len(bleeding_detected)):
                if bleeding_detected[i] and bleeding_class[i] > 0:  # 0 is no bleeding
                    severity_idx = bleeding_class[i].item() - 1
                    if 0 <= severity_idx < len(BleedingSeverity):
                        filtered_predictions['bleeding'].append({
                            'severity': list(BleedingSeverity)[severity_idx].value,
                            'controlled': bool(control_pred[i].item()),
                            'confidence': bleeding_max_prob[i].item()
                        })
            
            # Process suture predictions
            suture_probs = F.softmax(raw_predictions['suture']['detection'], dim=1)
            suture_max_prob, suture_detected = torch.max(suture_probs, dim=1)
            
            outcome_probs = F.softmax(raw_predictions['suture']['outcome'], dim=1)
            outcome_pred = torch.argmax(outcome_probs, dim=1)
            
            reason_probs = F.softmax(raw_predictions['suture']['failure_reason'], dim=1)
            reason_pred = torch.argmax(reason_probs, dim=1)
            
            filtered_predictions['suture_attempts'] = []
            
            for i in range(len(suture_detected)):
                if suture_max_prob[i] >= confidence_thresholds['suture'] and suture_detected[i] > 0:
                    outcome = list(SutureOutcome)[outcome_pred[i].item()]
                    reason = list(SutureFailureReason)[reason_pred[i].item()] if outcome == SutureOutcome.FAIL else None
                    
                    filtered_predictions['suture_attempts'].append({
                        'outcome': outcome.value,
                        'failure_reason': reason.value if reason else None,
                        'confidence': suture_max_prob[i].item()
                    })
            
            # Process anchor predictions
            anchor_probs = F.softmax(raw_predictions['anchor']['event_type'], dim=1)
            anchor_max_prob, anchor_event = torch.max(anchor_probs, dim=1)
            
            reason_probs = F.softmax(raw_predictions['anchor']['reposition_reason'], dim=1)
            reposition_reason_pred = torch.argmax(reason_probs, dim=1)
            
            filtered_predictions['anchor_events'] = []
            
            for i in range(len(anchor_event)):
                if anchor_max_prob[i] >= confidence_thresholds['anchor'] and anchor_event[i] > 0:
                    event_type = ["normal", "reposition", "pullout"][anchor_event[i].item()]
                    
                    if event_type in ["reposition", "pullout"]:
                        reason = list(RepositionReason)[reposition_reason_pred[i].item()]
                        filtered_predictions['anchor_events'].append({
                            'event_type': event_type,
                            'reason': reason.value if event_type == "reposition" else None,
                            'confidence': anchor_max_prob[i].item()
                        })
            
            # Process custom event predictions
            custom_probs = F.softmax(raw_predictions['custom_event']['category'], dim=1)
            custom_max_prob, custom_category = torch.max(custom_probs, dim=1)
            
            filtered_predictions['custom_events'] = []
            
            for i in range(len(custom_category)):
                if custom_max_prob[i] >= confidence_thresholds['custom_event']:
                    category = list(CustomEventCategory)[custom_category[i].item()]
                    filtered_predictions['custom_events'].append({
                        'category': category.value,
                        'confidence': custom_max_prob[i].item()
                    })
            
            return filtered_predictions

# ==================== EVENT DETECTION LOSS ====================

class EventDetectionLoss(nn.Module):
    """Comprehensive loss function for multi-branch event detection."""
    
    def __init__(self, 
                 bleeding_weight: float = 1.0,
                 suture_weight: float = 1.5,
                 anchor_weight: float = 1.2,
                 custom_weight: float = 0.8,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.bleeding_weight = bleeding_weight
        self.suture_weight = suture_weight
        self.anchor_weight = anchor_weight
        self.custom_weight = custom_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        Compute multi-branch loss.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Bleeding detection loss
        if 'bleeding' in predictions and 'bleeding' in targets:
            bleeding_severity_loss = self._focal_loss(
                predictions['bleeding']['severity'], 
                targets['bleeding']['severity']
            )
            
            bleeding_control_loss = F.cross_entropy(
                predictions['bleeding']['control_status'],
                targets['bleeding']['control_status']
            )
            
            bleeding_loss = bleeding_severity_loss + 0.5 * bleeding_control_loss
            total_loss += self.bleeding_weight * bleeding_loss
        
        # Suture attempt loss
        if 'suture' in predictions and 'suture' in targets:
            suture_detection_loss = self._focal_loss(
                predictions['suture']['detection'],
                targets['suture']['detection']
            )
            
            suture_outcome_loss = F.cross_entropy(
                predictions['suture']['outcome'],
                targets['suture']['outcome']
            )
            
            suture_reason_loss = F.cross_entropy(
                predictions['suture']['failure_reason'],
                targets['suture']['failure_reason']
            )
            
            suture_loss = suture_detection_loss + 0.8 * suture_outcome_loss + 0.6 * suture_reason_loss
            total_loss += self.suture_weight * suture_loss
        
        # Anchor event loss
        if 'anchor' in predictions and 'anchor' in targets:
            anchor_detection_loss = self._focal_loss(
                predictions['anchor']['event_type'],
                targets['anchor']['event_type']
            )
            
            anchor_reason_loss = F.cross_entropy(
                predictions['anchor']['reposition_reason'],
                targets['anchor']['reposition_reason']
            )
            
            anchor_loss = anchor_detection_loss + 0.7 * anchor_reason_loss
            total_loss += self.anchor_weight * anchor_loss
        
        # Custom event loss
        if 'custom_event' in predictions and 'custom_event' in targets:
            custom_loss = self._focal_loss(
                predictions['custom_event']['category'],
                targets['custom_event']['category']
            )
            total_loss += self.custom_weight * custom_loss
        
        return total_loss
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

# ==================== FACTORY FUNCTIONS ====================

def create_event_detector(config: Dict) -> MultiBranchEventDetector:
    """Factory function to create event detector from configuration."""
    
    model_config = config.get('event_detector', {})
    
    return MultiBranchEventDetector(
        backbone=model_config.get('backbone', 'resnet50'),
        feature_dim=model_config.get('feature_dim', 512),
        dropout=model_config.get('dropout', 0.3)
    )

if __name__ == "__main__":
    # Test the event detection model
    model = MultiBranchEventDetector()
    
    # Test input
    test_input = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    predictions = model(test_input, return_features=True)
    
    print("Event Detection Model Test Results:")
    print(f"Bleeding severity shape: {predictions['bleeding']['severity'].shape}")
    print(f"Suture detection shape: {predictions['suture']['detection'].shape}")
    print(f"Anchor event shape: {predictions['anchor']['event_type'].shape}")
    print(f"Custom event shape: {predictions['custom_event']['category'].shape}")
    
    # Test event prediction
    events = model.predict_events(test_input)
    print(f"Predicted events: {events}")
    
    print("✅ Event detection model test completed successfully!")