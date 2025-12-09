"""
Generate PDF and Excel reports from case data
"""
from datetime import datetime
import os


class ReportGenerator:
    """Generate professional reports from surgical analysis data"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def generate_case_report_text(self, case_id, output_path=None):
        """
        Generate a text-based case report (simplified for demo)
        In production, this would generate a formatted PDF with charts
        
        Args:
            case_id: ID of case to generate report for
            output_path: Where to save the report (optional)
            
        Returns:
            Report text content
        """
        session = self.db.get_session()
        
        try:
            # Get case data
            from app.models.database import Case
            case = session.query(Case).filter_by(case_id=case_id).first()
            
            if not case:
                return f"❌ Case {case_id} not found"
            
            # Build report
            report = []
            report.append("=" * 80)
            report.append("SURGICAL ANALYSIS REPORT")
            report.append("=" * 80)
            report.append("")
            report.append(f"Case ID:        {case.case_id}")
            report.append(f"Surgeon:        {case.surgeon.full_name}")
            report.append(f"Date:           {case.procedure_date.strftime('%Y-%m-%d %H:%M')}")
            report.append(f"Procedure:      {case.procedure_type}")
            report.append(f"Status:         {case.processing_status.upper()}")
            report.append("")
            report.append("=" * 80)
            report.append("EXECUTIVE SUMMARY")
            report.append("=" * 80)
            report.append("")
            report.append(f"Total Duration:     {case.actual_duration_min:.1f} minutes")
            report.append(f"Estimated Duration: {case.estimated_duration_min:.1f} minutes")
            
            diff_pct = ((case.actual_duration_min - case.estimated_duration_min) / case.estimated_duration_min) * 100
            status_icon = "✅" if abs(diff_pct) < 10 else "⚠️"
            report.append(f"Difference:         {diff_pct:+.1f}% {status_icon}")
            report.append("")
            
            # Phases
            report.append("=" * 80)
            report.append("PHASE BREAKDOWN")
            report.append("=" * 80)
            report.append("")
            report.append(f"{'Phase Name':<30} {'Duration':>12} {'Frames':>10}")
            report.append("-" * 80)
            
            total_phase_time = 0
            for phase in case.phases:
                anchor_str = f" (Anchor #{phase.anchor_number})" if phase.anchor_number else ""
                phase_name = f"{phase.phase_name}{anchor_str}"
                duration_str = f"{phase.duration_sec/60:.1f} min"
                frames_str = f"{phase.end_frame - phase.start_frame}"
                report.append(f"{phase_name:<30} {duration_str:>12} {frames_str:>10}")
                total_phase_time += phase.duration_sec
            
            report.append("-" * 80)
            report.append(f"{'TOTAL':<30} {total_phase_time/60:.1f} min")
            report.append("")
            
            # Events
            report.append("=" * 80)
            report.append("EVENT LOG")
            report.append("=" * 80)
            report.append("")
            
            # Group events by type
            event_types = {}
            for event in case.events:
                if event.event_type not in event_types:
                    event_types[event.event_type] = []
                event_types[event.event_type].append(event)
            
            for event_type, events in event_types.items():
                report.append(f"\n{event_type} ({len(events)} total):")
                report.append("-" * 40)
                
                for event in sorted(events, key=lambda e: e.event_time_sec):
                    time_str = f"{int(event.event_time_sec//60):02d}:{int(event.event_time_sec%60):02d}"
                    details = []
                    
                    if event.severity:
                        details.append(f"Severity: {event.severity}")
                    if event.anchor_number:
                        details.append(f"Anchor #{event.anchor_number}")
                    if event.attempt_number:
                        details.append(f"Attempt #{event.attempt_number}")
                    if event.outcome:
                        details.append(f"Outcome: {event.outcome}")
                    
                    details_str = ", ".join(details) if details else "N/A"
                    report.append(f"  {time_str} - {details_str}")
            
            report.append("")
            
            # Statistics
            report.append("=" * 80)
            report.append("PERFORMANCE METRICS")
            report.append("=" * 80)
            report.append("")
            
            # Suture success rate
            suture_events = [e for e in case.events if e.event_type == 'Suture Attempt']
            if suture_events:
                success = len([e for e in suture_events if e.outcome == 'Success'])
                total = len(suture_events)
                success_rate = (success / total) * 100
                report.append(f"Suture Attempts:    {total} total, {success} successful ({success_rate:.1f}%)")
            
            # Bleeding events
            bleeding_events = [e for e in case.events if e.event_type == 'Bleeding']
            if bleeding_events:
                severe = len([e for e in bleeding_events if e.severity == 'Severe'])
                moderate = len([e for e in bleeding_events if e.severity == 'Moderate'])
                mild = len([e for e in bleeding_events if e.severity == 'Mild'])
                report.append(f"Bleeding Events:    {len(bleeding_events)} total")
                report.append(f"  - Severe: {severe}, Moderate: {moderate}, Mild: {mild}")
            
            # Resources
            if case.resources:
                report.append("")
                report.append(f"Implants Used:      {case.resources.implants_count}")
                report.append(f"Disposables Used:   {case.resources.disposables_count}")
                report.append(f"Electrocautery:     {case.resources.electrocautery_usage_percent:.1f}%")
                report.append(f"Anchor Repositions: {case.resources.anchor_repositions}")
            
            report.append("")
            report.append("=" * 80)
            report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 80)
            
            report_text = "\n".join(report)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report_text)
                print(f"✅ Report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating report: {e}"
        finally:
            session.close()
    
    def generate_surgeon_summary_text(self, surgeon_id, output_path=None):
        """
        Generate aggregate surgeon summary report
        
        Args:
            surgeon_id: ID of surgeon
            output_path: Where to save the report (optional)
            
        Returns:
            Report text content
        """
        session = self.db.get_session()
        
        try:
            from app.models.database import Surgeon, Case, Event
            from sqlalchemy import func
            
            surgeon = session.query(Surgeon).filter_by(surgeon_id=surgeon_id).first()
            if not surgeon:
                return f"❌ Surgeon {surgeon_id} not found"
            
            # Get all cases for surgeon
            cases = session.query(Case).filter_by(surgeon_id=surgeon_id, processing_status='completed').all()
            
            if not cases:
                return f"❌ No completed cases found for {surgeon.full_name}"
            
            # Calculate statistics
            total_cases = len(cases)
            avg_duration = sum(c.actual_duration_min for c in cases) / total_cases
            avg_estimated = sum(c.estimated_duration_min for c in cases) / total_cases
            
            # Build report
            report = []
            report.append("=" * 80)
            report.append("SURGEON PERFORMANCE REPORT")
            report.append("=" * 80)
            report.append("")
            report.append(f"Surgeon:        {surgeon.full_name}")
            report.append(f"Department:     {surgeon.department}")
            report.append(f"Specialty:      {surgeon.specialty}")
            report.append(f"Total Cases:    {total_cases}")
            report.append("")
            report.append("=" * 80)
            report.append("AGGREGATE METRICS")
            report.append("=" * 80)
            report.append("")
            report.append(f"Average Procedure Time:     {avg_duration:.1f} minutes")
            report.append(f"Average Estimated Time:     {avg_estimated:.1f} minutes")
            
            diff_pct = ((avg_duration - avg_estimated) / avg_estimated) * 100
            status = "✅ On target" if abs(diff_pct) < 10 else "⚠️ Review needed"
            report.append(f"Difference:                 {diff_pct:+.1f}% {status}")
            report.append("")
            
            # Event statistics
            total_bleeding = 0
            total_sutures = 0
            successful_sutures = 0
            
            for case in cases:
                for event in case.events:
                    if event.event_type == 'Bleeding':
                        total_bleeding += 1
                    elif event.event_type == 'Suture Attempt':
                        total_sutures += 1
                        if event.outcome == 'Success':
                            successful_sutures += 1
            
            avg_bleeding = total_bleeding / total_cases
            report.append(f"Average Bleeding Events:    {avg_bleeding:.1f} per case ({total_bleeding} total)")
            
            if total_sutures > 0:
                success_rate = (successful_sutures / total_sutures) * 100
                report.append(f"Suture Success Rate:        {success_rate:.1f}% ({successful_sutures}/{total_sutures})")
            
            report.append("")
            report.append("=" * 80)
            report.append("CASE LIST")
            report.append("=" * 80)
            report.append("")
            report.append(f"{'Case ID':<15} {'Date':<12} {'Duration':>12} {'vs Est':>10}")
            report.append("-" * 80)
            
            for case in sorted(cases, key=lambda c: c.procedure_date, reverse=True):
                date_str = case.procedure_date.strftime('%Y-%m-%d')
                duration_str = f"{case.actual_duration_min:.1f} min"
                diff = case.actual_duration_min - case.estimated_duration_min
                diff_str = f"{diff:+.1f} min"
                report.append(f"{case.case_id:<15} {date_str:<12} {duration_str:>12} {diff_str:>10}")
            
            report.append("")
            report.append("=" * 80)
            report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 80)
            
            report_text = "\n".join(report)
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(report_text)
                print(f"✅ Report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            print(f"❌ Error generating surgeon report: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating surgeon report: {e}"
        finally:
            session.close()



