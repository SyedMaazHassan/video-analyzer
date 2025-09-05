"""
Surgical AI Models Module
=========================
Contains all specialized AI models for surgical video analysis.
"""

# Import main model classes for easy access
try:
    from .phase_detection.phase_detector import AdvancedPhaseDetector, create_phase_detector
    from .instrument_detection.instrument_tracker import InstrumentTracker
    from .event_detection.event_detector import MultiBranchEventDetector, create_event_detector
    from .motion_analysis.motion_analyzer import SurgicalMotionAnalyzer
except ImportError:
    # Fallback if dependencies not available
    pass