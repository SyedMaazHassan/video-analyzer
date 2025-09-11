#!/usr/bin/env python3
"""
SURGICAL AI SYSTEM - CORE DATA STRUCTURES
========================================
Enterprise-grade data structures for comprehensive surgical analysis.

Author: AI Surgical Analysis Team
Version: 1.0.0
License: Proprietary
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json
import uuid

# ==================== ENUMS ====================

class SurgicalPhaseType(Enum):
    """Enumeration of surgical phases."""
    PORTAL_PLACEMENT = "Portal Placement"
    DIAGNOSTIC_ARTHROSCOPY = "Diagnostic Arthroscopy"
    LABRAL_MOBILIZATION = "Labral Mobilization"
    GLENOID_PREPARATION = "Glenoid Preparation"
    ANCHOR_PLACEMENT = "Anchor Placement"
    SUTURE_PASSAGE = "Suture Passage"
    SUTURE_TENSIONING = "Suture Tensioning"
    FINAL_INSPECTION = "Final Inspection"

class InstrumentType(Enum):
    """Enumeration of surgical instruments."""
    ARTHROSCOPIC_CAMERA = "Arthroscopic Camera"
    TROCAR = "Trocar"
    CANNULA = "Cannula"
    SHAVER = "Shaver"
    ELECTROCAUTERY_PROBE = "Electrocautery Probe"
    PROBE = "Probe"
    GRASPER = "Grasper"
    BURR = "Burr"
    RASP = "Rasp"
    DRILL_GUIDE = "Drill Guide"
    SUTURE_ANCHOR = "Suture Anchor"
    SUTURE_PASSER = "Suture Passer"
    KNOT_PUSHER = "Knot Pusher"
    SUTURE_CUTTER = "Suture Cutter"

class BleedingSeverity(Enum):
    """Bleeding severity levels."""
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"

class SutureOutcome(Enum):
    """Suture attempt outcomes."""
    SUCCESS = "Success"
    FAIL = "Fail"

class AnchorMaterial(Enum):
    """Anchor materials."""
    PEEK = "PEEK"
    BIOCOMPOSITE = "Biocomposite"
    METAL = "Metal"
    ALL_SUTURE = "All-Suture"
    OTHER = "Other"

# ==================== CORE DATA STRUCTURES ====================

@dataclass
class FrameAnnotation:
    """Individual frame annotation with metadata."""
    frame_number: int
    timestamp: float
    confidence: float
    bbox: Optional[List[float]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SurgicalPhase:
    """Complete surgical phase with detailed metrics."""
    phase_type: SurgicalPhaseType
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    confidence: float
    idle_time: float = 0.0
    active_time: float = 0.0
    anchor_number: Optional[int] = None
    phase_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.active_time == 0.0:
            self.active_time = max(0, self.duration - self.idle_time)

@dataclass
class InstrumentEvent:
    """Instrument entry/exit tracking with detailed metadata."""
    instrument_type: InstrumentType
    event_type: str  # 'entry' or 'exit'
    frame: int
    timestamp: float
    confidence: float
    bbox: Optional[List[float]] = None
    usage_duration: Optional[float] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SutureAttempt:
    """Detailed suture attempt with outcome tracking."""
    anchor_number: int
    attempt_number: int
    outcome: SutureOutcome
    frame: int
    timestamp: float
    duration: Optional[float] = None
    reason_for_failure: Optional[str] = None
    confidence: float = 0.0
    cartilage_damage: bool = False
    pass_time_over_5min: bool = False
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class BleedingEvent:
    """Bleeding event with medical severity assessment."""
    severity: BleedingSeverity
    controlled: bool
    start_frame: int
    end_frame: Optional[int] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: float = 0.0
    intervention_required: bool = False
    bleeding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time

@dataclass
class AnchorTracking:
    """Complete anchor lifecycle management."""
    anchor_number: int
    placement_frame: int
    placement_time: float
    material: AnchorMaterial
    manufacturer: str
    repositions: List[Dict[str, Any]] = field(default_factory=list)
    pullout_frame: Optional[int] = None
    pullout_time: Optional[float] = None
    final_position_acceptable: bool = True
    anchor_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    def reposition_count(self) -> int:
        return len(self.repositions)
    
    @property
    def was_pulled_out(self) -> bool:
        return self.pullout_frame is not None

@dataclass
class AnatomicalStructure:
    """Anatomical structure identification and assessment."""
    structure_name: str
    frame: int
    timestamp: float
    abnormality: Optional[str] = None
    confidence: float = 0.0
    bbox: Optional[List[float]] = None
    structure_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class CustomEvent:
    """Flexible custom event tracking."""
    category: str
    description: str
    frame: int
    timestamp: float
    severity: str = "Medium"
    confidence: float = 0.0
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcedureMetrics:
    """Comprehensive procedure-level metrics."""
    total_duration: float
    total_idle_time: float
    total_active_time: float
    efficiency_score: float
    phase_transition_count: int
    bleeding_event_count: int
    suture_success_rate: float
    anchor_success_rate: float
    instrument_changes: int
    electrocautery_usage_percent: float
    time_to_first_suture: float
    
    @property
    def idle_time_percentage(self) -> float:
        return (self.total_idle_time / self.total_duration) * 100 if self.total_duration > 0 else 0

@dataclass
class SurgicalCase:
    """Complete surgical case with all analysis data."""
    # Case Identification
    case_id: str
    surgeon_id: str
    procedure_date: str
    procedure_type: str = "Labral Repair"
    
    # Video Metadata
    video_path: str = ""
    video_duration: float = 0.0
    video_fps: float = 30.0
    total_frames: int = 0
    
    # Analysis Results
    phases: List[SurgicalPhase] = field(default_factory=list)
    instruments: List[InstrumentEvent] = field(default_factory=list)
    suture_attempts: List[SutureAttempt] = field(default_factory=list)
    bleeding_events: List[BleedingEvent] = field(default_factory=list)
    anchor_tracking: List[AnchorTracking] = field(default_factory=list)
    anatomical_structures: List[AnatomicalStructure] = field(default_factory=list)
    custom_events: List[CustomEvent] = field(default_factory=list)
    
    # Phase Timing Metrics (required by inference engine)
    diagnostic_arthroscopy_time: float = 0.0
    glenoid_preparation_time: float = 0.0
    labral_mobilization_time: float = 0.0
    anchor_placement_time: float = 0.0
    suture_passage_time: float = 0.0
    suture_tensioning_time: float = 0.0
    final_inspection_time: float = 0.0
    total_idle_time: float = 0.0
    total_duration: float = 0.0
    number_of_disposables: int = 0
    number_of_implants: int = 0
    time_to_first_suture: float = 0.0
    bleeding_events_count: int = 0
    suture_failure_rate: float = 0.0
    
    # Comprehensive Metrics
    metrics: Optional[ProcedureMetrics] = None
    
    # Analysis Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_version: str = "1.0.0"
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> ProcedureMetrics:
        """Calculate comprehensive procedure metrics."""
        total_duration = sum(phase.duration for phase in self.phases)
        total_idle = sum(phase.idle_time for phase in self.phases)
        
        # Suture success rate
        successful_sutures = sum(1 for attempt in self.suture_attempts 
                               if attempt.outcome == SutureOutcome.SUCCESS)
        suture_success_rate = (successful_sutures / len(self.suture_attempts) 
                             if self.suture_attempts else 1.0)
        
        # Anchor success rate
        successful_anchors = sum(1 for anchor in self.anchor_tracking 
                               if not anchor.was_pulled_out and anchor.final_position_acceptable)
        anchor_success_rate = (successful_anchors / len(self.anchor_tracking) 
                             if self.anchor_tracking else 1.0)
        
        # Efficiency calculation
        efficiency_base = 100.0
        idle_penalty = (total_idle / total_duration) * 30 if total_duration > 0 else 0
        failure_penalty = (1 - suture_success_rate) * 20
        bleeding_penalty = len(self.bleeding_events) * 5
        efficiency_score = max(0, efficiency_base - idle_penalty - failure_penalty - bleeding_penalty) / 100
        
        # Time to first suture
        successful_suture_times = [attempt.timestamp for attempt in self.suture_attempts 
                                 if attempt.outcome == SutureOutcome.SUCCESS]
        time_to_first_suture = min(successful_suture_times) if successful_suture_times else 0.0
        
        return ProcedureMetrics(
            total_duration=total_duration,
            total_idle_time=total_idle,
            total_active_time=total_duration - total_idle,
            efficiency_score=efficiency_score,
            phase_transition_count=len(self.phases) - 1,
            bleeding_event_count=len(self.bleeding_events),
            suture_success_rate=suture_success_rate,
            anchor_success_rate=anchor_success_rate,
            instrument_changes=len(set(inst.instrument_type for inst in self.instruments)),
            electrocautery_usage_percent=self._calculate_electrocautery_usage(),
            time_to_first_suture=time_to_first_suture / 60.0  # Convert to minutes
        )
    
    def _calculate_electrocautery_usage(self) -> float:
        """Calculate percentage of procedure time using electrocautery."""
        electrocautery_events = [inst for inst in self.instruments 
                               if inst.instrument_type == InstrumentType.ELECTROCAUTERY_PROBE]
        total_electrocautery_time = sum(inst.usage_duration for inst in electrocautery_events 
                                      if inst.usage_duration)
        return (total_electrocautery_time / self.video_duration) * 100 if self.video_duration > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        def enum_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            return obj
        
        return {
            'case_id': self.case_id,
            'surgeon_id': self.surgeon_id,
            'procedure_date': self.procedure_date,
            'procedure_type': self.procedure_type,
            'video_metadata': {
                'video_path': self.video_path,
                'duration': self.video_duration,
                'fps': self.video_fps,
                'total_frames': self.total_frames
            },
            'phases': [
                {
                    'phase_type': phase.phase_type.value,
                    'start_time': phase.start_time,
                    'end_time': phase.end_time,
                    'duration': phase.duration,
                    'idle_time': phase.idle_time,
                    'confidence': phase.confidence
                } for phase in self.phases
            ],
            'instruments': [
                {
                    'instrument_type': inst.instrument_type.value,
                    'event_type': inst.event_type,
                    'timestamp': inst.timestamp,
                    'confidence': inst.confidence
                } for inst in self.instruments
            ],
            'suture_attempts': [
                {
                    'anchor_number': attempt.anchor_number,
                    'attempt_number': attempt.attempt_number,
                    'outcome': attempt.outcome.value,
                    'timestamp': attempt.timestamp,
                    'duration': attempt.duration
                } for attempt in self.suture_attempts
            ],
            'bleeding_events': [
                {
                    'severity': event.severity.value,
                    'controlled': event.controlled,
                    'start_time': event.start_time,
                    'duration': event.duration,
                    'confidence': event.confidence
                } for event in self.bleeding_events
            ],
            'metrics': {
                'total_procedure_time_minutes': self.metrics.total_duration / 60,
                'total_idle_time_minutes': self.metrics.total_idle_time / 60,
                'efficiency_score': self.metrics.efficiency_score,
                'bleeding_events': self.metrics.bleeding_event_count,
                'suture_success_rate': self.metrics.suture_success_rate,
                'time_to_first_suture_minutes': self.metrics.time_to_first_suture
            },
            'analysis_metadata': {
                'timestamp': self.analysis_timestamp,
                'version': self.analysis_version,
                'model_versions': self.model_versions
            }
        }
    
    def to_csv_rows(self) -> List[Dict[str, Any]]:
        """Convert to CSV-compatible rows matching client schema."""
        rows = []
        
        for phase in self.phases:
            row = {
                'case_id': self.case_id,
                'surgeon_id': self.surgeon_id,
                'procedure_date': self.procedure_date,
                'procedure_type': self.procedure_type,
                'step_name': phase.phase_type.value,
                'step_start_time': f"{int(phase.start_time//60):02d}:{int(phase.start_time%60):02d}",
                'step_end_time': f"{int(phase.end_time//60):02d}:{int(phase.end_time%60):02d}",
                'step_duration_seconds': phase.duration,
                'idle_time_seconds': phase.idle_time,
                'bleeding_detected': any(
                    phase.start_time <= event.start_time <= phase.end_time 
                    for event in self.bleeding_events
                ),
                'anchor_count': len(self.anchor_tracking),
                'suture_pass_attempts': len(self.suture_attempts),
                'efficiency_score': self.metrics.efficiency_score,
                'total_procedure_time_minutes': self.metrics.total_duration / 60,
                'bleeding_events': self.metrics.bleeding_event_count,
                'time_to_first_suture_minutes': self.metrics.time_to_first_suture
            }
            rows.append(row)
        
        return rows

# ==================== UTILITY FUNCTIONS ====================

def create_sample_surgical_case() -> SurgicalCase:
    """Create a sample surgical case for testing."""
    case = SurgicalCase(
        case_id="CASE_001",
        surgeon_id="DR_SMITH_001",
        procedure_date="2024-09-05",
        video_path="videos/sample_surgery.mp4",
        video_duration=2400.0,  # 40 minutes
        video_fps=30.0,
        total_frames=72000
    )
    
    # Add sample phases
    case.phases = [
        SurgicalPhase(
            phase_type=SurgicalPhaseType.PORTAL_PLACEMENT,
            start_frame=0, end_frame=900,
            start_time=0.0, end_time=30.0,
            duration=30.0, confidence=0.95
        ),
        SurgicalPhase(
            phase_type=SurgicalPhaseType.ANCHOR_PLACEMENT,
            start_frame=18000, end_frame=27000,
            start_time=600.0, end_time=900.0,
            duration=300.0, confidence=0.92,
            anchor_number=1
        )
    ]
    
    return case

if __name__ == "__main__":
    # Example usage
    case = create_sample_surgical_case()
    print("Sample Surgical Case Created:")
    print(f"Case ID: {case.case_id}")
    print(f"Efficiency Score: {case.metrics.efficiency_score:.2f}")
    print(f"Total Duration: {case.metrics.total_duration/60:.1f} minutes")