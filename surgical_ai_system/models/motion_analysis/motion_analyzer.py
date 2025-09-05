import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

class ActivityLevel(Enum):
    ACTIVE = "active"
    IDLE_SHORT = "idle_short"  # 3-10 seconds
    IDLE_LONG = "idle_long"    # >10 seconds
    NO_INSTRUMENTS = "no_instruments"

@dataclass
class MotionEvent:
    timestamp: float
    duration: float
    activity_level: ActivityLevel
    motion_score: float
    instrument_present: bool
    confidence: float

class OpticalFlowMotionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_calculator = cv2.FarnebackOpticalFlow_create(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def calculate_motion_magnitude(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        flow = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, None, None)
        if flow is not None:
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            return np.mean(magnitude)
        return 0.0

class InstrumentPresenceDetector(nn.Module):
    def __init__(self, num_instrument_classes: int = 14):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_instrument_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class TemporalSmoothingFilter(nn.Module):
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.history = []
        
    def smooth_predictions(self, prediction: float) -> float:
        self.history.append(prediction)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return sum(self.history) / len(self.history)

class MotionAnalysisModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.motion_detector = OpticalFlowMotionDetector()
        self.instrument_detector = InstrumentPresenceDetector()
        self.temporal_smoother = TemporalSmoothingFilter(
            window_size=config.get('smoothing_window', 5)
        )
        
        # Thresholds from client requirements
        self.motion_threshold_low = config.get('motion_threshold_low', 0.1)
        self.motion_threshold_high = config.get('motion_threshold_high', 0.3)
        self.idle_threshold_short = config.get('idle_threshold_short', 3.0)  # seconds
        self.idle_threshold_long = config.get('idle_threshold_long', 10.0)   # seconds
        self.instrument_confidence_threshold = config.get('instrument_confidence_threshold', 0.3)
        
        # State tracking
        self.previous_frame = None
        self.current_idle_duration = 0.0
        self.last_motion_time = 0.0
        
    def analyze_frame_motion(self, frame: np.ndarray, timestamp: float) -> Tuple[float, bool]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        motion_score = 0.0
        if self.previous_frame is not None:
            motion_score = self.motion_detector.calculate_motion_magnitude(
                self.previous_frame, gray_frame
            )
            motion_score = self.temporal_smoother.smooth_predictions(motion_score)
        
        self.previous_frame = gray_frame.copy()
        
        # Detect instrument presence
        frame_tensor = self.preprocess_frame(frame)
        with torch.no_grad():
            instrument_scores = self.instrument_detector(frame_tensor.unsqueeze(0))
            instruments_present = torch.any(instrument_scores > self.instrument_confidence_threshold).item()
        
        return motion_score, instruments_present
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(frame)
    
    def classify_activity(self, motion_score: float, instruments_present: bool, 
                         idle_duration: float) -> ActivityLevel:
        # No instruments visible for >10 seconds
        if not instruments_present and idle_duration >= self.idle_threshold_long:
            return ActivityLevel.NO_INSTRUMENTS
        
        # Low motion for extended periods
        if motion_score < self.motion_threshold_low:
            if idle_duration >= self.idle_threshold_long:
                return ActivityLevel.IDLE_LONG
            elif idle_duration >= self.idle_threshold_short:
                return ActivityLevel.IDLE_SHORT
        
        # Active motion detected
        if motion_score >= self.motion_threshold_high:
            return ActivityLevel.ACTIVE
        
        # Default to short idle if motion is low but not extended
        return ActivityLevel.IDLE_SHORT if motion_score < self.motion_threshold_low else ActivityLevel.ACTIVE
    
    def calculate_confidence(self, motion_score: float, instruments_present: bool, 
                           activity_level: ActivityLevel) -> float:
        base_confidence = 0.5
        
        # High confidence for clear motion patterns
        if activity_level == ActivityLevel.ACTIVE and motion_score > self.motion_threshold_high:
            base_confidence = 0.9
        elif activity_level == ActivityLevel.NO_INSTRUMENTS and not instruments_present:
            base_confidence = 0.85
        elif activity_level in [ActivityLevel.IDLE_SHORT, ActivityLevel.IDLE_LONG]:
            base_confidence = min(0.8, 0.5 + (1.0 - motion_score) * 0.4)
        
        return base_confidence

class SurgicalMotionAnalyzer:
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.model = MotionAnalysisModel(self.config)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.motion_events: List[MotionEvent] = []
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        return {
            'motion_threshold_low': 0.1,
            'motion_threshold_high': 0.3,
            'idle_threshold_short': 3.0,
            'idle_threshold_long': 10.0,
            'instrument_confidence_threshold': 0.3,
            'smoothing_window': 5,
            'frame_skip': 1
        }
    
    def load_model(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.logger.info(f"Motion analysis model loaded from {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load model from {model_path}: {e}")
            self.logger.info("Using untrained model - suitable for basic motion detection")
    
    def analyze_video_motion(self, video_path: str, fps: float = 30.0) -> List[MotionEvent]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.motion_events = []
        frame_count = 0
        current_activity = None
        activity_start_time = 0.0
        last_motion_time = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.config['frame_skip'] != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps
                
                # Analyze motion and instrument presence
                motion_score, instruments_present = self.model.analyze_frame_motion(frame, timestamp)
                
                # Calculate idle duration
                if motion_score >= self.model.motion_threshold_high:
                    last_motion_time = timestamp
                    idle_duration = 0.0
                else:
                    idle_duration = timestamp - last_motion_time
                
                # Classify activity
                activity_level = self.model.classify_activity(motion_score, instruments_present, idle_duration)
                confidence = self.model.calculate_confidence(motion_score, instruments_present, activity_level)
                
                # Track activity transitions
                if current_activity != activity_level:
                    # End previous activity
                    if current_activity is not None:
                        duration = timestamp - activity_start_time
                        if duration >= 0.5:  # Minimum event duration
                            event = MotionEvent(
                                timestamp=activity_start_time,
                                duration=duration,
                                activity_level=current_activity,
                                motion_score=motion_score,
                                instrument_present=instruments_present,
                                confidence=confidence
                            )
                            self.motion_events.append(event)
                    
                    # Start new activity
                    current_activity = activity_level
                    activity_start_time = timestamp
                
                frame_count += 1
                
                if frame_count % 300 == 0:  # Progress logging every 10 seconds at 30fps
                    self.logger.info(f"Processed {frame_count} frames ({timestamp:.1f}s)")
            
            # Handle final activity
            if current_activity is not None:
                final_timestamp = frame_count / fps
                duration = final_timestamp - activity_start_time
                if duration >= 0.5:
                    event = MotionEvent(
                        timestamp=activity_start_time,
                        duration=duration,
                        activity_level=current_activity,
                        motion_score=motion_score,
                        instrument_present=instruments_present,
                        confidence=confidence
                    )
                    self.motion_events.append(event)
        
        finally:
            cap.release()
        
        self.logger.info(f"Motion analysis complete: {len(self.motion_events)} events detected")
        return self.motion_events
    
    def get_idle_time_statistics(self) -> Dict:
        total_idle_short = sum(event.duration for event in self.motion_events 
                              if event.activity_level == ActivityLevel.IDLE_SHORT)
        total_idle_long = sum(event.duration for event in self.motion_events 
                             if event.activity_level == ActivityLevel.IDLE_LONG)
        total_no_instruments = sum(event.duration for event in self.motion_events 
                                  if event.activity_level == ActivityLevel.NO_INSTRUMENTS)
        
        total_idle = total_idle_short + total_idle_long + total_no_instruments
        total_duration = sum(event.duration for event in self.motion_events)
        
        return {
            'total_idle_time': total_idle,
            'idle_percentage': (total_idle / total_duration * 100) if total_duration > 0 else 0,
            'idle_short_time': total_idle_short,
            'idle_long_time': total_idle_long,
            'no_instruments_time': total_no_instruments,
            'total_events': len(self.motion_events),
            'idle_events_count': len([e for e in self.motion_events 
                                    if e.activity_level != ActivityLevel.ACTIVE])
        }
    
    def export_results(self) -> Dict:
        return {
            'motion_events': [
                {
                    'timestamp': event.timestamp,
                    'duration': event.duration,
                    'activity_level': event.activity_level.value,
                    'motion_score': event.motion_score,
                    'instrument_present': event.instrument_present,
                    'confidence': event.confidence
                }
                for event in self.motion_events
            ],
            'statistics': self.get_idle_time_statistics()
        }

def create_motion_analyzer(model_path: Optional[str] = None, 
                          config: Optional[Dict] = None) -> SurgicalMotionAnalyzer:
    return SurgicalMotionAnalyzer(model_path, config)

if __name__ == "__main__":
    # Demo usage
    analyzer = create_motion_analyzer()
    
    # Example analysis
    video_path = "sample_surgical_video.mp4"
    if Path(video_path).exists():
        events = analyzer.analyze_video_motion(video_path)
        stats = analyzer.get_idle_time_statistics()
        print(f"Detected {len(events)} motion events")
        print(f"Total idle time: {stats['total_idle_time']:.2f}s ({stats['idle_percentage']:.1f}%)")