"""
Event Detection Module
======================
Bleeding and surgical event detection.
"""

from .event_detector import MultiBranchEventDetector, EventDetectionLoss, create_event_detector

__all__ = ['MultiBranchEventDetector', 'EventDetectionLoss', 'create_event_detector']