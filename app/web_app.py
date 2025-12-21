"""
Simple web interface for Surgical Analysis Platform
Quick visualization of data and reports
"""
from flask import Flask, render_template, jsonify, request, send_file
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import DatabaseManager, Case, Surgeon, Phase, Event
from app.services.report_generator import ReportGenerator
from app.services.real_inference import RealInferenceEngine
from app.services.mock_inference import MockInferenceEngine
from app.services.timeline_visualizer import TimelineVisualizer
from app.services.pdf_report_generator import PDFReportGenerator
from sqlalchemy.orm import joinedload
from sqlalchemy import func, desc
from datetime import datetime
import json
import threading

app = Flask(__name__, static_folder='static', static_url_path='/static')
db = DatabaseManager('data/surgical_analysis.db')

# Initialize sample surgeons if database is empty (for new installations)
db.init_sample_surgeons()

report_gen = ReportGenerator(db)
pdf_gen = PDFReportGenerator(db)

# Try to use real inference engine, fall back to mock if model not available
try:
    inference = RealInferenceEngine(db)
    print("Using Real AI Inference Engine (BiLSTM-CRF)")
except Exception as e:
    print(f"Real inference engine not available ({e}), using mock inference")
    inference = MockInferenceEngine(db)

timeline_viz = TimelineVisualizer(db)


@app.route('/')
def index():
    """Dashboard page"""
    return render_template('index.html')


@app.route('/api/dashboard')
def api_dashboard():
    """Get dashboard statistics"""
    session = db.get_session()
    
    try:
        # Get statistics
        total_cases = session.query(Case).filter_by(processing_status='completed').count()
        
        avg_duration = session.query(func.avg(Case.actual_duration_min)).filter_by(
            processing_status='completed'
        ).scalar() or 0
        
        avg_estimated = session.query(func.avg(Case.estimated_duration_min)).filter_by(
            processing_status='completed'
        ).scalar() or 0
        
        total_bleeding = session.query(Event).join(Case).filter(
            Event.event_type == 'Bleeding',
            Case.processing_status == 'completed'
        ).count()
        
        total_sutures = session.query(Event).join(Case).filter(
            Event.event_type == 'Suture Attempt',
            Case.processing_status == 'completed'
        ).count()
        
        successful_sutures = session.query(Event).join(Case).filter(
            Event.event_type == 'Suture Attempt',
            Event.outcome == 'Success',
            Case.processing_status == 'completed'
        ).count()
        
        data = {
            'total_cases': total_cases,
            'avg_duration': round(avg_duration, 1),
            'avg_estimated': round(avg_estimated, 1),
            'diff_percent': round(((avg_duration - avg_estimated) / avg_estimated * 100), 1) if avg_estimated > 0 else 0,
            'total_bleeding': total_bleeding,
            'avg_bleeding': round(total_bleeding / total_cases, 1) if total_cases > 0 else 0,
            'suture_success_rate': round((successful_sutures / total_sutures * 100), 1) if total_sutures > 0 else 0,
            'total_sutures': total_sutures,
            'successful_sutures': successful_sutures
        }
        
        return jsonify(data)
        
    finally:
        session.close()


@app.route('/api/cases')
def api_cases():
    """Get list of all cases"""
    session = db.get_session()
    
    try:
        cases = session.query(Case).options(joinedload(Case.surgeon)).order_by(desc(Case.procedure_date)).all()
        
        cases_data = []
        for case in cases:
            cases_data.append({
                'case_id': case.case_id,
                'surgeon_name': case.surgeon.full_name if case.surgeon else 'N/A',
                'procedure_type': case.procedure_type,
                'procedure_date': case.procedure_date.strftime('%Y-%m-%d %H:%M'),
                'actual_duration_min': round(case.actual_duration_min, 1) if case.actual_duration_min else None,
                'estimated_duration_min': round(case.estimated_duration_min, 1) if case.estimated_duration_min else None,
                'processing_status': case.processing_status
            })
        
        return jsonify(cases_data)
        
    finally:
        session.close()


@app.route('/api/case/<case_id>')
def api_case_detail(case_id):
    """Get detailed case information"""
    session = db.get_session()
    
    try:
        case = session.query(Case).options(
            joinedload(Case.surgeon),
            joinedload(Case.phases),
            joinedload(Case.events),
            joinedload(Case.resources)
        ).filter_by(case_id=case_id).first()
        
        if not case:
            return jsonify({'error': 'Case not found'}), 404
        
        # Build case data
        case_data = {
            'case_id': case.case_id,
            'surgeon_name': case.surgeon.full_name if case.surgeon else 'N/A',
            'procedure_type': case.procedure_type,
            'procedure_date': case.procedure_date.strftime('%Y-%m-%d %H:%M'),
            'actual_duration_min': round(case.actual_duration_min, 1) if case.actual_duration_min else None,
            'estimated_duration_min': round(case.estimated_duration_min, 1) if case.estimated_duration_min else None,
            'processing_status': case.processing_status,
            'notes': case.notes,
            'phases': [],
            'events': [],
            'resources': None
        }
        
        # Add phases
        for phase in sorted(case.phases, key=lambda p: p.start_frame):
            phase_data = {
                'phase_name': phase.phase_name,
                'start_frame': phase.start_frame,
                'end_frame': phase.end_frame,
                'duration_sec': round(phase.duration_sec, 1),
                'duration_min': round(phase.duration_sec / 60, 1),
                'anchor_number': phase.anchor_number,
                'confidence_score': round(phase.confidence_score * 100, 1) if phase.confidence_score else None
            }
            case_data['phases'].append(phase_data)
        
        # Add events
        for event in sorted(case.events, key=lambda e: e.event_frame):
            event_data = {
                'event_type': event.event_type,
                'event_frame': event.event_frame,
                'event_time_sec': round(event.event_time_sec, 1),
                'event_time_min': round(event.event_time_sec / 60, 1),
                'anchor_number': event.anchor_number,
                'attempt_number': event.attempt_number,
                'outcome': event.outcome,
                'severity': event.severity,
                'confidence_score': round(event.confidence_score * 100, 1) if event.confidence_score else None
            }
            case_data['events'].append(event_data)
        
        # Add resources
        if case.resources:
            case_data['resources'] = {
                'implants_count': case.resources.implants_count,
                'disposables_count': case.resources.disposables_count,
                'electrocautery_usage_percent': round(case.resources.electrocautery_usage_percent, 1),
                'anchor_repositions': case.resources.anchor_repositions
            }
        
        return jsonify(case_data)
        
    finally:
        session.close()


@app.route('/api/surgeons')
def api_surgeons():
    """Get list of all surgeons"""
    session = db.get_session()
    
    try:
        surgeons = session.query(Surgeon).all()
        
        surgeons_data = []
        for surgeon in surgeons:
            # Get case statistics
            cases = session.query(Case).filter_by(
                surgeon_id=surgeon.surgeon_id,
                processing_status='completed'
            ).all()
            
            surgeon_data = {
                'surgeon_id': surgeon.surgeon_id,
                'full_name': surgeon.full_name,
                'department': surgeon.department,
                'specialty': surgeon.specialty,
                'total_cases': len(cases),
                'avg_duration': round(sum(c.actual_duration_min for c in cases) / len(cases), 1) if cases else 0
            }
            surgeons_data.append(surgeon_data)
        
        return jsonify(surgeons_data)
        
    finally:
        session.close()


@app.route('/api/report/<case_id>')
def api_report(case_id):
    """Generate and return case report"""
    try:
        report_text = report_gen.generate_case_report_text(case_id)
        return jsonify({'report': report_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/surgeon-report/<surgeon_id>')
def api_surgeon_report(surgeon_id):
    """Generate and return surgeon summary report"""
    try:
        report_text = report_gen.generate_surgeon_summary_text(surgeon_id)
        return jsonify({'report': report_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/phase-durations')
def api_phase_durations():
    """Get phase duration data for charts"""
    session = db.get_session()
    
    try:
        phases = session.query(Phase).join(Case).filter(
            Case.processing_status == 'completed'
        ).all()
        
        # Group by phase name
        phase_data = {}
        for phase in phases:
            name = phase.phase_name
            if name not in phase_data:
                phase_data[name] = []
            phase_data[name].append(phase.duration_sec / 60)  # Convert to minutes
        
        # Calculate statistics
        chart_data = {
            'labels': [],
            'avg': [],
            'min': [],
            'max': []
        }
        
        for name, durations in sorted(phase_data.items()):
            chart_data['labels'].append(name)
            chart_data['avg'].append(round(sum(durations) / len(durations), 1))
            chart_data['min'].append(round(min(durations), 1))
            chart_data['max'].append(round(max(durations), 1))
        
        return jsonify(chart_data)
        
    finally:
        session.close()


@app.route('/api/charts/event-distribution')
def api_event_distribution():
    """Get event distribution data for charts"""
    session = db.get_session()
    
    try:
        events = session.query(Event).join(Case).filter(
            Case.processing_status == 'completed'
        ).all()
        
        # Count by type
        event_counts = {}
        for event in events:
            event_type = event.event_type
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
        
        chart_data = {
            'labels': list(event_counts.keys()),
            'values': list(event_counts.values())
        }
        
        return jsonify(chart_data)
        
    finally:
        session.close()


@app.route('/upload')
def upload_page():
    """Upload new case page"""
    return render_template('upload.html')


@app.route('/api/upload-case', methods=['POST'])
def api_upload_case():
    """Handle new case upload with async processing"""
    try:
        # Get form data
        case_id = request.form.get('case_id')
        surgeon_id = request.form.get('surgeon_id')
        procedure_type = request.form.get('procedure_type', 'Labral Repair')
        estimated_duration = float(request.form.get('estimated_duration', 52.0))
        notes = request.form.get('notes', '')
        
        # Handle video file
        video_file = request.files.get('video_file')
        video_path = 'videos/placeholder.mp4'
        
        if video_file:
            # Save video file
            import os
            os.makedirs('data/videos', exist_ok=True)
            video_filename = f"{case_id}_{video_file.filename}"
            video_path = os.path.join('data/videos', video_filename)
            video_file.save(video_path)
        
        # Check if case already exists
        session = db.get_session()
        existing_case = session.query(Case).filter(Case.case_id == case_id).first()
        
        if existing_case:
            session.close()
            return jsonify({
                'success': False,
                'error': f'Case ID "{case_id}" already exists in the database. Please use a different Case ID.'
            }), 400
        
        # Create case with "processing" status
        case = Case(
            case_id=case_id,
            surgeon_id=surgeon_id,
            procedure_type=procedure_type,
            procedure_date=datetime.now(),
            video_path=video_path,
            estimated_duration_min=estimated_duration,
            processing_status='processing',  # Changed from 'queued'
            notes=notes
        )
        session.add(case)
        session.commit()
        case_id_created = case.case_id
        session.close()
        
        # Start processing in background thread
        def process_in_background():
            try:
                def progress(p, m):
                    # Update processing status in database
                    session = db.get_session()
                    try:
                        case = session.query(Case).filter(Case.case_id == case_id_created).first()
                        if case:
                            case.processing_status = 'processing' if p < 100 else 'completed'
                            session.commit()
                    finally:
                        session.close()
                
                inference.process_video(case_id_created, video_path, progress)
                
                # Final status update
                session = db.get_session()
                try:
                    case = session.query(Case).filter(Case.case_id == case_id_created).first()
                    if case:
                        case.processing_status = 'completed'
                        session.commit()
                finally:
                    session.close()
                    
            except Exception as e:
                print(f"Error in background processing: {e}")
                # Mark as failed
                session = db.get_session()
                try:
                    case = session.query(Case).filter(Case.case_id == case_id_created).first()
                    if case:
                        case.processing_status = 'failed'
                        session.commit()
                finally:
                    session.close()
        
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'case_id': case_id_created,
            'message': 'Case uploaded successfully! Processing in background (~10 seconds).'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/add-surgeon', methods=['POST'])
def api_add_surgeon():
    """Add a new surgeon"""
    try:
        data = request.get_json()

        session = db.get_session()

        try:
            # Check if surgeon already exists
            existing = session.query(Surgeon).filter_by(surgeon_id=data['surgeon_id']).first()
            if existing:
                return jsonify({
                    'success': False,
                    'error': f"Surgeon with ID {data['surgeon_id']} already exists"
                }), 400

            # Create surgeon
            surgeon = Surgeon(
                surgeon_id=data['surgeon_id'],
                first_name=data['first_name'],
                last_name=data['last_name'],
                department=data.get('department', 'Orthopedics'),
                specialty=data.get('specialty', 'Sports Medicine'),
                email=data.get('email', '')
            )
            session.add(surgeon)
            session.commit()

            # Capture name before closing session
            full_name = f"{data['first_name']} {data['last_name']}"

            return jsonify({
                'success': True,
                'message': f"Surgeon {full_name} added successfully!"
            })
        finally:
            session.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/timeline/<case_id>')
def api_timeline(case_id):
    """Get timeline visualization as base64 image"""
    try:
        img_base64 = timeline_viz.create_case_timeline(case_id, return_base64=True)
        if img_base64:
            return jsonify({'success': True, 'image': img_base64})
        else:
            return jsonify({'success': False, 'error': 'Failed to generate timeline'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pdf-report/<case_id>')
def api_pdf_report(case_id):
    """Generate and download PDF report"""
    try:
        pdf_path = pdf_gen.generate_case_report_pdf(case_id)
        if pdf_path and os.path.exists(pdf_path):
            # Use absolute path to avoid path issues
            abs_path = os.path.abspath(pdf_path)
            return send_file(abs_path, as_attachment=True, 
                           download_name=f'case_{case_id}_report.pdf',
                           mimetype='application/pdf')
        else:
            return jsonify({'error': 'Failed to generate PDF'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdf-surgeon-report/<surgeon_id>')
def api_pdf_surgeon_report(surgeon_id):
    """Generate and download surgeon PDF report"""
    try:
        pdf_path = pdf_gen.generate_surgeon_report_pdf(surgeon_id)
        if pdf_path and os.path.exists(pdf_path):
            # Use absolute path to avoid path issues
            abs_path = os.path.abspath(pdf_path)
            return send_file(abs_path, as_attachment=True,
                           download_name=f'surgeon_{surgeon_id}_report.pdf',
                           mimetype='application/pdf')
        else:
            return jsonify({'error': 'Failed to generate PDF'}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("  🏥 SURGICAL ANALYSIS PLATFORM - WEB INTERFACE")
    print("=" * 80)
    print(f"\n  ✅ Database: data/surgical_analysis.db")
    print(f"  ✅ Server starting at: http://127.0.0.1:5005")
    print(f"\n  📊 Dashboard:     http://127.0.0.1:5005")
    print(f"  📋 Cases List:    http://127.0.0.1:5005/api/cases")
    print(f"  👨‍⚕️ Surgeons:       http://127.0.0.1:5005/api/surgeons")
    print("\n" + "=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5005, threaded=True)

