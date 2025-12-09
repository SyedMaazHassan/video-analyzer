"""
Mock AI inference engine for demo/development
Simulates video processing and generates realistic data
"""
import random
import time
from datetime import datetime
from app.models.database import Case, Phase, Event, Resource


class MockInferenceEngine:
    """Simulates AI inference for demo purposes"""
    
    PHASE_TEMPLATES = {
        'Portal Placement': (30, 180),  # (min_duration_sec, max_duration_sec)
        'Diagnostic Arthroscopy': (120, 960),
        'Labral Mobilization': (300, 720),
        'Glenoid Preparation': (120, 600),
        'Anchor Placement': (120, 360),
        'Suture Passage': (180, 480),
        'Suture Tensioning': (120, 300),
        'Final Inspection': (120, 600),
    }
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def process_video(self, case_id, video_path, progress_callback=None):
        """
        Mock video processing
        
        Args:
            case_id: ID of the case being processed
            video_path: Path to video file
            progress_callback: Function to call with progress updates (percent, message)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if progress_callback:
                progress_callback(5, "Extracting video metadata...")
            time.sleep(1)
            
            # Get video info (mock - would use OpenCV in real implementation)
            video_info = self._extract_video_info(video_path)
            
            if progress_callback:
                progress_callback(15, "Detecting phases...")
            time.sleep(2)
            
            # Generate mock phases
            phases = self._generate_mock_phases(video_info)
            
            if progress_callback:
                progress_callback(50, "Detecting events...")
            time.sleep(2)
            
            # Generate mock events
            events = self._generate_mock_events(video_info, phases)
            
            if progress_callback:
                progress_callback(75, "Analyzing motion...")
            time.sleep(1)
            
            # Generate resources
            resources = self._generate_mock_resources()
            
            if progress_callback:
                progress_callback(90, "Saving results...")
            time.sleep(1)
            
            # Save to database
            self._save_results(case_id, video_info, phases, events, resources)
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            print(f"✅ Mock processing complete for case {case_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error in mock processing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_video_info(self, video_path):
        """Mock video info extraction"""
        # In real implementation, would use OpenCV
        # cv2.VideoCapture(video_path) to get real values
        
        duration_min = random.gauss(52, 8)  # Mean 52 min, std 8 min
        duration_min = max(35, min(75, duration_min))
        
        return {
            'duration_sec': duration_min * 60,
            'fps': 30,
            'total_frames': int(duration_min * 60 * 30),
            'width': 1280,
            'height': 720
        }
    
    def _generate_mock_phases(self, video_info):
        """Generate realistic phase data"""
        phases = []
        current_frame = 0
        fps = video_info['fps']
        
        # Determine number of anchors (2-5)
        num_anchors = random.randint(2, 5)
        
        # Portal placement (brief)
        duration_sec = random.uniform(30, 180)
        phases.append({
            'phase_name': 'Portal Placement',
            'start_frame': current_frame,
            'end_frame': current_frame + int(duration_sec * fps),
            'duration_sec': duration_sec,
            'anchor_number': None,
            'confidence_score': random.uniform(0.85, 0.98)
        })
        current_frame = phases[-1]['end_frame']
        
        # Diagnostic Arthroscopy
        duration_sec = random.uniform(120, 960)  # 2-16 min
        phases.append({
            'phase_name': 'Diagnostic Arthroscopy',
            'start_frame': current_frame,
            'end_frame': current_frame + int(duration_sec * fps),
            'duration_sec': duration_sec,
            'anchor_number': None,
            'confidence_score': random.uniform(0.85, 0.98)
        })
        current_frame = phases[-1]['end_frame']
        
        # Labral Mobilization
        duration_sec = random.uniform(300, 720)  # 5-12 min
        phases.append({
            'phase_name': 'Labral Mobilization',
            'start_frame': current_frame,
            'end_frame': current_frame + int(duration_sec * fps),
            'duration_sec': duration_sec,
            'anchor_number': None,
            'confidence_score': random.uniform(0.85, 0.98)
        })
        current_frame = phases[-1]['end_frame']
        
        # Glenoid Preparation
        duration_sec = random.uniform(120, 600)  # 2-10 min
        phases.append({
            'phase_name': 'Glenoid Preparation',
            'start_frame': current_frame,
            'end_frame': current_frame + int(duration_sec * fps),
            'duration_sec': duration_sec,
            'anchor_number': None,
            'confidence_score': random.uniform(0.85, 0.98)
        })
        current_frame = phases[-1]['end_frame']
        
        # Anchor-specific phases
        for anchor_num in range(1, num_anchors + 1):
            # Anchor Placement
            duration_sec = random.uniform(120, 360)  # 2-6 min
            phases.append({
                'phase_name': 'Anchor Placement',
                'start_frame': current_frame,
                'end_frame': current_frame + int(duration_sec * fps),
                'duration_sec': duration_sec,
                'anchor_number': anchor_num,
                'confidence_score': random.uniform(0.85, 0.98)
            })
            current_frame = phases[-1]['end_frame']
            
            # Suture Passage
            duration_sec = random.uniform(180, 480)  # 3-8 min
            phases.append({
                'phase_name': 'Suture Passage',
                'start_frame': current_frame,
                'end_frame': current_frame + int(duration_sec * fps),
                'duration_sec': duration_sec,
                'anchor_number': anchor_num,
                'confidence_score': random.uniform(0.85, 0.98)
            })
            current_frame = phases[-1]['end_frame']
            
            # Suture Tensioning
            duration_sec = random.uniform(120, 300)  # 2-5 min
            phases.append({
                'phase_name': 'Suture Tensioning',
                'start_frame': current_frame,
                'end_frame': current_frame + int(duration_sec * fps),
                'duration_sec': duration_sec,
                'anchor_number': anchor_num,
                'confidence_score': random.uniform(0.85, 0.98)
            })
            current_frame = phases[-1]['end_frame']
        
        # Final Inspection
        duration_sec = random.uniform(120, 600)  # 2-10 min
        phases.append({
            'phase_name': 'Final Inspection',
            'start_frame': current_frame,
            'end_frame': current_frame + int(duration_sec * fps),
            'duration_sec': duration_sec,
            'anchor_number': None,
            'confidence_score': random.uniform(0.85, 0.98)
        })
        
        return phases
    
    def _generate_mock_events(self, video_info, phases):
        """Generate realistic event data"""
        events = []
        total_frames = phases[-1]['end_frame']
        
        # Bleeding events (0-5 per case)
        num_bleeding = random.randint(0, 5)
        for _ in range(num_bleeding):
            event_frame = random.randint(100, total_frames - 100)
            severity = random.choice(['Mild', 'Moderate', 'Severe'])
            
            events.append({
                'event_type': 'Bleeding',
                'event_frame': event_frame,
                'event_time_sec': event_frame / 30.0,
                'severity': severity,
                'anchor_number': None,
                'attempt_number': None,
                'outcome': None,
                'confidence_score': random.uniform(0.75, 0.95)
            })
        
        # Suture attempts (based on anchor phases)
        anchor_phases = [p for p in phases if p['phase_name'] == 'Suture Passage']
        for phase in anchor_phases:
            # 1-3 attempts per anchor
            num_attempts = random.randint(1, 3)
            for attempt_num in range(1, num_attempts + 1):
                # Within the suture passage phase
                event_frame = random.randint(phase['start_frame'], phase['end_frame'])
                # First attempts more likely to fail
                outcome = 'Success' if (attempt_num > 1 or random.random() > 0.2) else 'Fail'
                
                events.append({
                    'event_type': 'Suture Attempt',
                    'event_frame': event_frame,
                    'event_time_sec': event_frame / 30.0,
                    'anchor_number': phase['anchor_number'],
                    'attempt_number': attempt_num,
                    'outcome': outcome,
                    'severity': None,
                    'confidence_score': random.uniform(0.8, 0.95)
                })
        
        return events
    
    def _generate_mock_resources(self):
        """Generate resource usage data"""
        return {
            'implants_count': random.randint(2, 5),
            'disposables_count': random.randint(3, 8),
            'electrocautery_usage_percent': random.uniform(5, 30),
            'anchor_repositions': random.randint(0, 2)
        }
    
    def _save_results(self, case_id, video_info, phases, events, resources):
        """Save analysis results to database"""
        session = self.db.get_session()
        
        try:
            # Update case with video info
            case = session.query(Case).filter_by(case_id=case_id).first()
            if case:
                case.video_duration_sec = video_info['duration_sec']
                case.video_fps = video_info['fps']
                case.total_frames = video_info['total_frames']
                case.actual_duration_min = video_info['duration_sec'] / 60.0
                case.processing_status = 'completed'
            
            # Add phases
            for phase_data in phases:
                phase = Phase(
                    case_id=case_id,
                    **phase_data
                )
                session.add(phase)
            
            # Add events
            for event_data in events:
                event = Event(
                    case_id=case_id,
                    **event_data
                )
                session.add(event)
            
            # Add resources
            resource = Resource(
                case_id=case_id,
                **resources
            )
            session.add(resource)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()



