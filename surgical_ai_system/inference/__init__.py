"""
Inference Module
================
Master inference engine and utilities.
"""

try:
    from .master_inference_engine import MasterInferenceEngine
    __all__ = ['MasterInferenceEngine']
except ImportError:
    __all__ = []