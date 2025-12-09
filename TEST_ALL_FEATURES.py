"""
Comprehensive test of all features
Run this to validate everything works
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.database import DatabaseManager, Case, Surgeon
from app.services.mock_inference import MockInferenceEngine
from app.services.pdf_report_generator import PDFReportGenerator
from app.services.timeline_visualizer import TimelineVisualizer
from app.services.annotation_loader import AnnotationLoader
from datetime import datetime

def test_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_database():
    """Test database operations"""
    test_section("TEST 1: DATABASE")
    
    db = DatabaseManager('data/surgical_analysis.db')
    session = db.get_session()
    
    surgeons = session.query(Surgeon).all()
    cases = session.query(Case).all()
    
    print(f"✅ Surgeons in database: {len(surgeons)}")
    for s in surgeons:
        print(f"   - {s.full_name} ({s.surgeon_id})")
    
    print(f"✅ Cases in database: {len(cases)}")
    for c in cases[:5]:
        print(f"   - {c.case_id}: {c.surgeon.full_name if c.surgeon else 'N/A'}")
    
    session.close()
    return len(cases) > 0

def test_mock_ai():
    """Test mock AI processing"""
    test_section("TEST 2: MOCK AI PROCESSING")
    
    db = DatabaseManager('data/surgical_analysis.db')
    
    # Create test case
    case_id = f"TEST{int(datetime.now().timestamp()) % 10000:04d}"
    session = db.get_session()
    
    case = Case(
        case_id=case_id,
        surgeon_id='S001',
        procedure_type='Labral Repair',
        procedure_date=datetime.now(),
        video_path='videos/test.mp4',
        estimated_duration_min=52.0,
        processing_status='queued'
    )
    session.add(case)
    session.commit()
    session.close()
    
    print(f"✅ Created test case: {case_id}")
    
    # Process with mock AI
    inference = MockInferenceEngine(db)
    
    def progress(p, m):
        if p % 25 == 0:
            print(f"   {p}% - {m}")
    
    success = inference.process_video(case_id, 'videos/test.mp4', progress)
    
    if success:
        print(f"✅ Mock AI processing successful")
        
        # Check results
        session = db.get_session()
        case = session.query(Case).filter_by(case_id=case_id).first()
        print(f"   Phases generated: {len(case.phases)}")
        print(f"   Events generated: {len(case.events)}")
        session.close()
        return True
    else:
        print("❌ Mock AI processing failed")
        return False

def test_timeline_viz():
    """Test timeline visualization"""
    test_section("TEST 3: TIMELINE VISUALIZATION")
    
    db = DatabaseManager('data/surgical_analysis.db')
    session = db.get_session()
    
    # Get a case
    case = session.query(Case).filter_by(processing_status='completed').first()
    session.close()
    
    if not case:
        print("❌ No completed cases to visualize")
        return False
    
    print(f"✅ Generating timeline for {case.case_id}")
    
    viz = TimelineVisualizer(db)
    output_path = f'data/results/test_timeline_{case.case_id}.png'
    result = viz.create_case_timeline(case.case_id, output_path=output_path)
    
    if result and os.path.exists(result):
        print(f"✅ Timeline saved: {result}")
        
        # Test base64 version
        base64_img = viz.create_case_timeline(case.case_id, return_base64=True)
        if base64_img and base64_img.startswith('data:image/png;base64,'):
            print(f"✅ Base64 timeline generated (for web)")
        
        return True
    else:
        print("❌ Timeline generation failed")
        return False

def test_pdf_reports():
    """Test PDF report generation"""
    test_section("TEST 4: PDF REPORT GENERATION")
    
    db = DatabaseManager('data/surgical_analysis.db')
    session = db.get_session()
    
    # Get a case
    case = session.query(Case).filter_by(processing_status='completed').first()
    session.close()
    
    if not case:
        print("❌ No completed cases for PDF")
        return False
    
    print(f"✅ Generating PDF for {case.case_id}")
    
    pdf_gen = PDFReportGenerator(db)
    pdf_path = pdf_gen.generate_case_report_pdf(case.case_id)
    
    if pdf_path and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"✅ PDF generated: {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print("❌ PDF generation failed")
        return False

def test_surgeon_report():
    """Test surgeon report generation"""
    test_section("TEST 5: SURGEON REPORT")
    
    db = DatabaseManager('data/surgical_analysis.db')
    
    pdf_gen = PDFReportGenerator(db)
    pdf_path = pdf_gen.generate_surgeon_report_pdf('S001')
    
    if pdf_path and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"✅ Surgeon PDF generated: {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print("❌ Surgeon PDF generation failed")
        return False

def main():
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE FEATURE TEST")
    print("  Testing all components of Surgical Analysis Platform")
    print("=" * 80)
    
    results = []
    
    results.append(("Database", test_database()))
    results.append(("Mock AI Processing", test_mock_ai()))
    results.append(("Timeline Visualization", test_timeline_viz()))
    results.append(("PDF Reports", test_pdf_reports()))
    results.append(("Surgeon Reports", test_surgeon_report()))
    
    # Summary
    test_section("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}\n")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "=" * 80)
    
    if passed == total:
        print("  🎉 ALL TESTS PASSED! System is fully functional!")
        print("=" * 80)
        print("\n  You can now:")
        print("    - Run web app: python app/web_app.py")
        print("    - Run desktop app: ./run_desktop_app.sh")
        print("    - Check data/results/ for generated PDFs and images")
        print("\n" + "=" * 80 + "\n")
        return 0
    else:
        print("  ⚠️ Some tests failed. Check errors above.")
        print("=" * 80 + "\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())



