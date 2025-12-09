"""
Load existing CSV annotations into database
"""
import pandas as pd
from datetime import datetime, timedelta
import os
import random
from app.models.database import Case, Phase, Event, Resource


class AnnotationLoader:
    """Load existing CSV annotations into database"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def load_from_csv(self, phases_csv, events_csv, surgeon_id, case_id_prefix='ANN'):
        """
        Load phases and events from CSV files
        
        Args:
            phases_csv: Path to phases CSV file
            events_csv: Path to events CSV file  
            surgeon_id: ID of surgeon to assign cases to
            case_id_prefix: Prefix for generated case IDs
            
        Returns:
            List of created case IDs
        """
        session = self.db.get_session()
        case_ids = []
        
        try:
            # Load CSVs
            if not os.path.exists(phases_csv):
                print(f"❌ Phases CSV not found: {phases_csv}")
                return case_ids
            
            phases_df = pd.read_csv(phases_csv)
            events_df = pd.DataFrame()
            
            if os.path.exists(events_csv):
                events_df = pd.read_csv(events_csv)
            else:
                print(f"⚠️ Events CSV not found: {events_csv}, continuing without events")
            
            print(f"📊 Loaded {len(phases_df)} phase records")
            if not events_df.empty:
                print(f"📊 Loaded {len(events_df)} event records")
            
            # Group by video (if multiple videos in CSV)
            if 'video' in phases_df.columns:
                videos = phases_df['video'].unique()
            else:
                videos = ['annotations']
            
            print(f"📹 Found {len(videos)} video(s) in annotation data")
            
            for idx, video_name in enumerate(videos):
                case_id = f"{case_id_prefix}{idx+1:03d}"
                
                # Filter data for this video
                if 'video' in phases_df.columns:
                    video_phases = phases_df[phases_df['video'] == video_name].copy()
                    video_events = events_df[events_df['video'] == video_name].copy() if not events_df.empty else pd.DataFrame()
                else:
                    video_phases = phases_df.copy()
                    video_events = events_df.copy()
                
                # Check if case already exists
                existing_case = session.query(Case).filter_by(case_id=case_id).first()
                if existing_case:
                    print(f"⚠️ Case {case_id} already exists, skipping...")
                    continue
                
                # Calculate total duration
                if len(video_phases) > 0:
                    total_frames = int(video_phases['end_frame'].max())
                    total_duration_sec = float(video_phases['duration_sec'].sum())
                else:
                    total_frames = 0
                    total_duration_sec = 0
                
                # Create case
                case = Case(
                    case_id=case_id,
                    surgeon_id=surgeon_id,
                    procedure_type='Labral Repair',
                    procedure_date=datetime.now() - timedelta(days=random.randint(1, 90)),
                    video_path=f'videos/{video_name}.mp4',
                    video_duration_sec=total_duration_sec,
                    video_fps=30,
                    total_frames=total_frames,
                    estimated_duration_min=52.0,
                    actual_duration_min=total_duration_sec / 60.0,
                    processing_status='completed',
                    notes='Loaded from annotation file'
                )
                session.add(case)
                
                # Add phases
                phases_added = 0
                for _, row in video_phases.iterrows():
                    phase = Phase(
                        case_id=case_id,
                        phase_name=str(row['label']),
                        start_frame=int(row['start_frame']),
                        end_frame=int(row['end_frame']),
                        duration_sec=float(row['duration_sec']),
                        anchor_number=int(row['Anchor Number']) if pd.notna(row.get('Anchor Number')) else None,
                        confidence_score=0.95
                    )
                    session.add(phase)
                    phases_added += 1
                
                # Add events
                events_added = 0
                if not video_events.empty:
                    for _, row in video_events.iterrows():
                        event = Event(
                            case_id=case_id,
                            event_type=str(row['label']),
                            event_frame=int(row['start_frame']),
                            event_time_sec=float(row['duration_sec']),
                            anchor_number=int(row['Anchor Number']) if pd.notna(row.get('Anchor Number')) else None,
                            attempt_number=int(row['Attempt Number']) if pd.notna(row.get('Attempt Number')) else None,
                            outcome=str(row['Outcome']) if pd.notna(row.get('Outcome')) else None,
                            severity=str(row['Severity']) if pd.notna(row.get('Severity')) else None,
                            confidence_score=0.92
                        )
                        session.add(event)
                        events_added += 1
                
                # Add mock resources
                resource = Resource(
                    case_id=case_id,
                    implants_count=random.randint(2, 5),
                    disposables_count=random.randint(3, 8),
                    electrocautery_usage_percent=random.uniform(5, 30),
                    anchor_repositions=random.randint(0, 2)
                )
                session.add(resource)
                
                case_ids.append(case_id)
                print(f"✅ Created case {case_id}: {phases_added} phases, {events_added} events")
            
            session.commit()
            print(f"✅ Successfully loaded {len(case_ids)} cases from annotation files")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error loading annotations: {e}")
            import traceback
            traceback.print_exc()
        finally:
            session.close()
        
        return case_ids
    
    def load_all_annotation_files(self, surgeon_id='S001'):
        """
        Load all annotation files from analysis_results directory
        
        Args:
            surgeon_id: ID of surgeon to assign cases to
            
        Returns:
            List of created case IDs
        """
        base_dir = 'analysis_results'
        
        # Try to load aggregate analysis data
        aggregate_phases = os.path.join(base_dir, 'aggregate_analysis', 'all_phases_data.csv')
        aggregate_events = os.path.join(base_dir, 'aggregate_analysis', 'all_events_data.csv')
        
        case_ids = []
        
        if os.path.exists(aggregate_phases):
            print(f"📂 Loading aggregate annotation data...")
            ids = self.load_from_csv(aggregate_phases, aggregate_events, surgeon_id, case_id_prefix='AGG')
            case_ids.extend(ids)
        
        # Try to load per-surgery analysis data
        per_surgery_phases = os.path.join(base_dir, 'per_surgery_analysis', 'annotations', 'phases.csv')
        per_surgery_events = os.path.join(base_dir, 'per_surgery_analysis', 'annotations', 'events.csv')
        
        if os.path.exists(per_surgery_phases):
            print(f"📂 Loading per-surgery annotation data...")
            ids = self.load_from_csv(per_surgery_phases, per_surgery_events, surgeon_id, case_id_prefix='PER')
            case_ids.extend(ids)
        
        if not case_ids:
            print(f"⚠️ No annotation files found in {base_dir}")
        
        return case_ids



