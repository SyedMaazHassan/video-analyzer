import torch
import cv2
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import logging
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import all specialized models
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.phase_detection.phase_detector import AdvancedPhaseDetector
from models.instrument_detection.instrument_tracker import AdvancedInstrumentDetector  
from models.event_detection.event_detector import MultiBranchEventDetector
from models.motion_analysis.motion_analyzer import SurgicalMotionAnalyzer
from core.data_structures.surgical_entities import (
    SurgicalCase, SurgicalPhase, InstrumentEvent, BleedingEvent, 
    SutureAttempt, CustomEvent, SurgicalPhaseType, InstrumentType,
    BleedingSeverity, SutureOutcome, AnchorTracking, AnchorMaterial
)

class MasterInferenceEngine:
    """
    Enterprise-grade master inference engine that orchestrates all specialized AI models
    to produce comprehensive surgical video analysis matching exact client requirements.
    """
    
    def __init__(self, config_path: Optional[str] = None, models_dir: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / "trained_models"
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Initialize all specialized models
        self.phase_detector = None
        self.instrument_tracker = None
        self.event_detector = None
        self.motion_analyzer = None
        
        self._load_all_models()
        
        # Analysis state
        self.current_case: Optional[SurgicalCase] = None
        self.video_info: Dict = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        default_config = {
            "models": {
                "phase_detector": "phase_detector.pth",
                "instrument_tracker": "instrument_tracker.pth", 
                "event_detector": "event_detector.pth",
                "motion_analyzer": "motion_analyzer.pth"
            },
            "processing": {
                "batch_size": 32,
                "frame_skip": 1,
                "parallel_processing": True,
                "max_workers": 4
            },
            "output": {
                "save_json": True,
                "save_csv": True,
                "save_excel": True,
                "detailed_timeline": True,
                "confidence_threshold": 0.5
            },
            "video": {
                "target_fps": 30.0,
                "resize_height": 480,
                "resize_width": 640
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MasterInferenceEngine")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_all_models(self) -> None:
        """Load all specialized AI models"""
        try:
            # Phase Detection Model
            self.phase_detector = AdvancedPhaseDetector()
            phase_model_path = self.models_dir / self.config["models"]["phase_detector"]
            if phase_model_path.exists():
                self.phase_detector.load_state_dict(torch.load(phase_model_path, map_location='cpu'))
            self.phase_detector.eval()
            self.logger.info("Phase detector loaded successfully")
            
            # Instrument Tracking Model  
            self.instrument_tracker = AdvancedInstrumentDetector()
            instrument_model_path = self.models_dir / self.config["models"]["instrument_tracker"]
            if instrument_model_path.exists():
                self.instrument_tracker.load_state_dict(torch.load(instrument_model_path, map_location='cpu'))
            self.instrument_tracker.eval()
            self.logger.info("Instrument tracker loaded successfully")
            
            # Event Detection Model
            self.event_detector = MultiBranchEventDetector()
            event_model_path = self.models_dir / self.config["models"]["event_detector"] 
            if event_model_path.exists():
                self.event_detector.load_state_dict(torch.load(event_model_path, map_location='cpu'))
            self.event_detector.eval()
            self.logger.info("Event detector loaded successfully")
            
            # Motion Analysis Model
            self.motion_analyzer = SurgicalMotionAnalyzer()
            motion_model_path = self.models_dir / self.config["models"]["motion_analyzer"]
            if motion_model_path.exists():
                self.motion_analyzer.load_state_dict(torch.load(motion_model_path, map_location='cpu'))
            self.motion_analyzer.eval()
            self.logger.info("Motion analyzer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def analyze_video(self, video_path: str, case_metadata: Dict) -> SurgicalCase:
        """
        Master analysis method that coordinates all models to analyze surgical video
        and produce comprehensive results matching client schema requirements.
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.logger.info(f"Starting comprehensive analysis of: {video_path.name}")
        
        # Extract video information
        self._extract_video_info(str(video_path))
        
        # Initialize surgical case
        self.current_case = SurgicalCase(
            case_id=case_metadata.get('case_id', f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            surgeon_id=case_metadata.get('surgeon_id', 'UNKNOWN'),
            video_path=str(video_path),
            total_duration=self.video_info['duration'],
            fps=self.video_info['fps']
        )
        
        # Run all analysis models in parallel for efficiency
        if self.config["processing"]["parallel_processing"]:
            analysis_results = self._run_parallel_analysis(str(video_path))
        else:
            analysis_results = self._run_sequential_analysis(str(video_path))
        
        # Integrate all analysis results
        self._integrate_analysis_results(analysis_results)
        
        # Calculate comprehensive metrics
        self._calculate_comprehensive_metrics()
        
        analysis_time = time.time() - start_time
        self.logger.info(f"Analysis complete in {analysis_time:.2f}s")
        
        return self.current_case
    
    def _extract_video_info(self, video_path: str) -> None:
        """Extract basic video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
        
        cap.release()
        self.logger.info(f"Video info: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")
    
    def _run_parallel_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run all models in parallel for maximum efficiency"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config["processing"]["max_workers"]) as executor:
            # Submit all analysis tasks
            futures = {
                executor.submit(self.phase_detector.analyze_video_phases, video_path, self.video_info['fps']): 'phases',
                executor.submit(self.instrument_tracker.track_video_instruments, video_path, self.video_info['fps']): 'instruments', 
                executor.submit(self.event_detector.detect_video_events, video_path, self.video_info['fps']): 'events',
                executor.submit(self.motion_analyzer.analyze_video_motion, video_path, self.video_info['fps']): 'motion'
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                analysis_type = futures[future]
                try:
                    result = future.result()
                    results[analysis_type] = result
                    self.logger.info(f"Completed {analysis_type} analysis")
                except Exception as e:
                    self.logger.error(f"Error in {analysis_type} analysis: {e}")
                    results[analysis_type] = []
        
        return results
    
    def _run_sequential_analysis(self, video_path: str) -> Dict[str, Any]:
        """Run all models sequentially"""
        results = {}
        
        # Phase detection
        self.logger.info("Running phase detection...")
        results['phases'] = self.phase_detector.analyze_video_phases(video_path, self.video_info['fps'])
        
        # Instrument tracking
        self.logger.info("Running instrument tracking...")
        results['instruments'] = self.instrument_tracker.track_video_instruments(video_path, self.video_info['fps'])
        
        # Event detection
        self.logger.info("Running event detection...")
        results['events'] = self.event_detector.detect_video_events(video_path, self.video_info['fps'])
        
        # Motion analysis
        self.logger.info("Running motion analysis...")
        results['motion'] = self.motion_analyzer.analyze_video_motion(video_path, self.video_info['fps'])
        
        return results
    
    def _integrate_analysis_results(self, results: Dict[str, Any]) -> None:
        """Integrate results from all models into unified surgical case structure"""
        
        # Process phases
        if 'phases' in results:
            for phase_result in results['phases']:
                phase = SurgicalPhase(
                    phase_type=PhaseType(phase_result.phase_type.value),
                    start_time=phase_result.start_time,
                    end_time=phase_result.end_time,
                    confidence=phase_result.confidence
                )
                self.current_case.phases.append(phase)
        
        # Process instrument events
        if 'instruments' in results:
            for instrument_result in results['instruments']:
                event = InstrumentEvent(
                    instrument_type=InstrumentType(instrument_result.instrument_type.value),
                    timestamp=instrument_result.timestamp,
                    duration=instrument_result.duration,
                    confidence=instrument_result.confidence
                )
                self.current_case.instrument_events.append(event)
        
        # Process bleeding events
        if 'events' in results:
            for event_result in results['events']:
                if hasattr(event_result, 'severity'):  # Bleeding event
                    bleeding_event = BleedingEvent(
                        timestamp=event_result.timestamp,
                        duration=event_result.duration,
                        severity=BleedingSeverity(event_result.severity.value),
                        confidence=event_result.confidence
                    )
                    self.current_case.bleeding_events.append(bleeding_event)
                
                elif hasattr(event_result, 'outcome'):  # Suture attempt
                    suture_event = SutureAttempt(
                        timestamp=event_result.timestamp,
                        anchor_number=getattr(event_result, 'anchor_number', 1),
                        attempt_number=getattr(event_result, 'attempt_number', 1),
                        outcome=SutureOutcome(event_result.outcome.value),
                        confidence=event_result.confidence
                    )
                    self.current_case.suture_attempts.append(suture_event)
        
        # Process motion events to calculate idle time
        if 'motion' in results:
            idle_stats = self.motion_analyzer.get_idle_time_statistics()
            self.current_case.total_idle_time = idle_stats['total_idle_time']
    
    def _calculate_comprehensive_metrics(self) -> None:
        """Calculate all metrics required by client specifications"""
        
        # Phase durations
        for phase in self.current_case.phases:
            phase_duration = phase.end_time - phase.start_time
            
            # Add to phase-specific counters
            if phase.phase_type == PhaseType.DIAGNOSTIC_ARTHROSCOPY:
                self.current_case.diagnostic_arthroscopy_time += phase_duration
            elif phase.phase_type == PhaseType.GLENOID_PREPARATION:
                self.current_case.glenoid_preparation_time += phase_duration
            elif phase.phase_type == PhaseType.LABRAL_MOBILIZATION:
                self.current_case.labral_mobilization_time += phase_duration
            elif phase.phase_type == PhaseType.ANCHOR_PLACEMENT:
                self.current_case.anchor_placement_time += phase_duration
            elif phase.phase_type == PhaseType.SUTURE_PASSAGE:
                self.current_case.suture_passage_time += phase_duration
            elif phase.phase_type == PhaseType.SUTURE_TENSIONING:
                self.current_case.suture_tensioning_time += phase_duration
            elif phase.phase_type == PhaseType.FINAL_INSPECTION:
                self.current_case.final_inspection_time += phase_duration
        
        # Instrument utilization
        unique_instruments = set(event.instrument_type for event in self.current_case.instrument_events)
        self.current_case.number_of_disposables = len(unique_instruments)
        
        # Anchor and suture metrics
        self.current_case.number_of_implants = len(set(
            attempt.anchor_number for attempt in self.current_case.suture_attempts
        ))
        
        # Calculate time to first suture
        suture_passage_phases = [p for p in self.current_case.phases 
                               if p.phase_type == PhaseType.SUTURE_PASSAGE]
        if suture_passage_phases:
            first_suture_phase = min(suture_passage_phases, key=lambda p: p.start_time)
            successful_attempts = [a for a in self.current_case.suture_attempts 
                                 if a.outcome == SutureOutcome.SUCCESS and 
                                 a.timestamp >= first_suture_phase.start_time]
            if successful_attempts:
                first_success = min(successful_attempts, key=lambda a: a.timestamp)
                self.current_case.time_to_first_suture = first_success.timestamp - first_suture_phase.start_time
        
        # Bleeding statistics
        self.current_case.bleeding_events_count = len(self.current_case.bleeding_events)
        
        # Suture failure rate
        failed_attempts = len([a for a in self.current_case.suture_attempts 
                             if a.outcome == SutureOutcome.FAIL])
        total_attempts = len(self.current_case.suture_attempts)
        self.current_case.suture_failure_rate = failed_attempts / total_attempts if total_attempts > 0 else 0
    
    def save_comprehensive_results(self, output_dir: str) -> Dict[str, str]:
        """Save results in all required formats (JSON, CSV, Excel)"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        case_name = self.current_case.case_id
        saved_files = {}
        
        # Save JSON (detailed format)
        if self.config["output"]["save_json"]:
            json_path = output_path / f"{case_name}_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(asdict(self.current_case), f, indent=2, default=str)
            saved_files['json'] = str(json_path)
            self.logger.info(f"Saved JSON results: {json_path}")
        
        # Save CSV (metrics format)
        if self.config["output"]["save_csv"]:
            csv_path = output_path / f"{case_name}_metrics.csv"
            metrics_df = self._create_metrics_dataframe()
            metrics_df.to_csv(csv_path, index=False)
            saved_files['csv'] = str(csv_path)
            self.logger.info(f"Saved CSV metrics: {csv_path}")
        
        # Save Excel (comprehensive format with multiple sheets)
        if self.config["output"]["save_excel"]:
            excel_path = output_path / f"{case_name}_comprehensive.xlsx"
            self._save_excel_report(excel_path)
            saved_files['excel'] = str(excel_path)
            self.logger.info(f"Saved Excel report: {excel_path}")
        
        return saved_files
    
    def _create_metrics_dataframe(self) -> pd.DataFrame:
        """Create CSV-compatible metrics dataframe matching client schema"""
        metrics = {
            'case_id': [self.current_case.case_id],
            'surgeon_id': [self.current_case.surgeon_id],
            'total_procedure_time': [self.current_case.total_duration],
            'total_idle_time': [self.current_case.total_idle_time],
            'diagnostic_arthroscopy_time': [self.current_case.diagnostic_arthroscopy_time],
            'glenoid_preparation_time': [self.current_case.glenoid_preparation_time],
            'labral_mobilization_time': [self.current_case.labral_mobilization_time],
            'anchor_placement_time': [self.current_case.anchor_placement_time],
            'suture_passage_time': [self.current_case.suture_passage_time],
            'suture_tensioning_time': [self.current_case.suture_tensioning_time],
            'final_inspection_time': [self.current_case.final_inspection_time],
            'number_of_implants': [self.current_case.number_of_implants],
            'number_of_disposables': [self.current_case.number_of_disposables],
            'time_to_first_suture': [self.current_case.time_to_first_suture],
            'bleeding_events_count': [self.current_case.bleeding_events_count],
            'suture_failure_rate': [self.current_case.suture_failure_rate]
        }
        return pd.DataFrame(metrics)
    
    def _save_excel_report(self, excel_path: Path) -> None:
        """Save comprehensive Excel report with multiple sheets"""
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main metrics sheet
            metrics_df = self._create_metrics_dataframe()
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Phases timeline sheet
            if self.current_case.phases:
                phases_data = []
                for i, phase in enumerate(self.current_case.phases):
                    phases_data.append({
                        'phase_number': i + 1,
                        'phase_type': phase.phase_type.value,
                        'start_time': phase.start_time,
                        'end_time': phase.end_time,
                        'duration': phase.end_time - phase.start_time,
                        'confidence': phase.confidence
                    })
                phases_df = pd.DataFrame(phases_data)
                phases_df.to_excel(writer, sheet_name='Phases', index=False)
            
            # Events timeline sheet
            if self.current_case.bleeding_events or self.current_case.suture_attempts:
                events_data = []
                
                for event in self.current_case.bleeding_events:
                    events_data.append({
                        'timestamp': event.timestamp,
                        'event_type': 'bleeding',
                        'severity': event.severity.value,
                        'duration': event.duration,
                        'confidence': event.confidence
                    })
                
                for attempt in self.current_case.suture_attempts:
                    events_data.append({
                        'timestamp': attempt.timestamp,
                        'event_type': 'suture_attempt',
                        'anchor_number': attempt.anchor_number,
                        'attempt_number': attempt.attempt_number,
                        'outcome': attempt.outcome.value,
                        'confidence': attempt.confidence
                    })
                
                if events_data:
                    events_df = pd.DataFrame(events_data).sort_values('timestamp')
                    events_df.to_excel(writer, sheet_name='Events', index=False)

def create_master_engine(config_path: Optional[str] = None, 
                        models_dir: Optional[str] = None) -> MasterInferenceEngine:
    """Factory function to create master inference engine"""
    return MasterInferenceEngine(config_path, models_dir)

if __name__ == "__main__":
    # Demo usage
    engine = create_master_engine()
    
    # Example analysis
    case_metadata = {
        'case_id': 'DEMO_001',
        'surgeon_id': 'SURGEON_A'
    }
    
    video_path = "sample_surgical_video.mp4"
    if Path(video_path).exists():
        try:
            case = engine.analyze_video(video_path, case_metadata)
            saved_files = engine.save_comprehensive_results("results")
            print(f"Analysis complete! Saved files: {saved_files}")
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        print("Sample video not found - create engine successfully initialized")