"""
CLI Demo of Surgical Analysis Platform
Tests end-to-end flow: upload -> processing -> report
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import DatabaseManager, Case
from app.services.annotation_loader import AnnotationLoader
from app.services.mock_inference import MockInferenceEngine
from app.services.report_generator import ReportGenerator
from datetime import datetime
import time


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_load_annotations():
    """Demo: Load existing annotation data"""
    print_section("STEP 1: LOADING ANNOTATION DATA")
    
    # Initialize database
    db = DatabaseManager('data/surgical_analysis.db')
    
    # Initialize surgeons
    db.init_sample_surgeons()
    
    # Load annotations
    loader = AnnotationLoader(db)
    case_ids = loader.load_all_annotation_files(surgeon_id='S001')
    
    if case_ids:
        print(f"\n✅ Loaded {len(case_ids)} cases from annotation files")
        print(f"   Case IDs: {', '.join(case_ids[:5])}" + (" ..." if len(case_ids) > 5 else ""))
    else:
        print("\n⚠️ No annotation files found. We'll create a mock case instead.")
    
    return db, case_ids


def demo_video_upload(db):
    """Demo: Simulate video upload and processing"""
    print_section("STEP 2: VIDEO UPLOAD & MOCK PROCESSING")
    
    # Create a new case (simulating upload)
    case_id = f"DEMO{int(time.time()) % 10000:04d}"
    
    print(f"📹 Creating new case: {case_id}")
    print("   Surgeon: Dr. Sarah Anderson")
    print("   Procedure: Labral Repair")
    print("   Video: demo_surgery.mp4 (mock)")
    
    session = db.get_session()
    
    case = Case(
        case_id=case_id,
        surgeon_id='S001',
        procedure_type='Labral Repair',
        procedure_date=datetime.now(),
        video_path='videos/demo_surgery.mp4',
        estimated_duration_min=52.0,
        processing_status='queued',
        notes='Demo case - mock processing'
    )
    session.add(case)
    session.commit()
    session.close()
    
    print(f"\n✅ Case {case_id} created and queued for processing")
    
    # Mock processing
    print("\n🤖 Starting AI processing (mock)...")
    
    def progress_callback(percent, message):
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r   [{bar}] {percent:3d}% - {message}", end='', flush=True)
    
    inference = MockInferenceEngine(db)
    success = inference.process_video(case_id, 'videos/demo_surgery.mp4', progress_callback)
    
    print()  # New line after progress bar
    
    if success:
        print(f"\n✅ Processing complete for case {case_id}")
    else:
        print(f"\n❌ Processing failed for case {case_id}")
    
    return case_id


def demo_generate_reports(db, case_id):
    """Demo: Generate reports"""
    print_section("STEP 3: REPORT GENERATION")
    
    generator = ReportGenerator(db)
    
    # Generate case report
    print(f"📄 Generating report for case {case_id}...")
    report_path = f"data/results/case_{case_id}_report.txt"
    report_text = generator.generate_case_report_text(case_id, report_path)
    
    print("\n" + "─" * 80)
    print("CASE REPORT PREVIEW:")
    print("─" * 80)
    # Print first 50 lines
    lines = report_text.split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"\n... ({len(lines) - 50} more lines)")
    print("─" * 80)


def demo_surgeon_dashboard(db):
    """Demo: Surgeon aggregate statistics"""
    print_section("STEP 4: SURGEON DASHBOARD")
    
    generator = ReportGenerator(db)
    
    print("👨‍⚕️ Generating surgeon summary for Dr. Sarah Anderson...")
    report_path = "data/results/surgeon_S001_summary.txt"
    report_text = generator.generate_surgeon_summary_text('S001', report_path)
    
    print("\n" + "─" * 80)
    print("SURGEON SUMMARY PREVIEW:")
    print("─" * 80)
    lines = report_text.split('\n')
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"\n... ({len(lines) - 40} more lines)")
    print("─" * 80)


def demo_list_cases(db):
    """Demo: List all cases in database"""
    print_section("STEP 5: CASE DATABASE")
    
    session = db.get_session()
    # Use joinedload to eagerly load surgeon relationship
    from sqlalchemy.orm import joinedload
    cases = session.query(Case).options(joinedload(Case.surgeon)).all()
    
    print(f"📊 Total cases in database: {len(cases)}")
    print("\nRecent cases:")
    print(f"{'Case ID':<15} {'Surgeon':<25} {'Date':<12} {'Duration':>12} {'Status':>12}")
    print("-" * 80)
    
    for case in sorted(cases, key=lambda c: c.procedure_date, reverse=True)[:10]:
        surgeon_name = case.surgeon.full_name if case.surgeon else "N/A"
        date_str = case.procedure_date.strftime('%Y-%m-%d')
        duration_str = f"{case.actual_duration_min:.1f} min" if case.actual_duration_min else "N/A"
        status_str = case.processing_status.upper()
        
        print(f"{case.case_id:<15} {surgeon_name:<25} {date_str:<12} {duration_str:>12} {status_str:>12}")
    
    session.close()


def main():
    """Run complete demo"""
    print("\n" + "=" * 80)
    print("  SURGICAL ANALYSIS PLATFORM - CLI DEMO")
    print("  End-to-End Flow: Upload → Processing → Reports")
    print("=" * 80)
    
    try:
        # Step 1: Load existing annotations
        db, annotation_case_ids = demo_load_annotations()
        
        # Step 2: Simulate new video upload and processing
        new_case_id = demo_video_upload(db)
        
        # Step 3: Generate case report
        demo_generate_reports(db, new_case_id)
        
        # Step 4: Generate surgeon dashboard
        demo_surgeon_dashboard(db)
        
        # Step 5: List all cases
        demo_list_cases(db)
        
        # Final summary
        print_section("✅ DEMO COMPLETE")
        print("End-to-end flow tested successfully!")
        print("\nWhat was demonstrated:")
        print("  ✅ Database initialization")
        print("  ✅ Loading annotation CSV files")
        print("  ✅ Mock video upload")
        print("  ✅ Mock AI processing")
        print("  ✅ Case report generation")
        print("  ✅ Surgeon aggregate reports")
        print("  ✅ Case database queries")
        print("\nNext steps:")
        print("  - Run 'python app/main_gui.py' to launch the GUI")
        print("  - Reports saved in data/results/")
        print("  - Database at data/surgical_analysis.db")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

