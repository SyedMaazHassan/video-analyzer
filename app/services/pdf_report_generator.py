"""
Professional PDF report generation with formatting and charts
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO


class PDFReportGenerator:
    """Generate beautiful PDF reports"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate_case_report_pdf(self, case_id, output_path=None):
        """Generate PDF report for a case"""
        from app.models.database import Case
        
        session = self.db.get_session()
        
        try:
            case = session.query(Case).filter_by(case_id=case_id).first()
            if not case:
                return None
            
            # Create output path
            if not output_path:
                os.makedirs('data/results', exist_ok=True)
                output_path = f'data/results/case_{case_id}_report.pdf'
            
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Title
            story.append(Paragraph("SURGICAL ANALYSIS REPORT", self.styles['CustomTitle']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Case information
            case_info = [
                ['Case ID:', case.case_id],
                ['Surgeon:', case.surgeon.full_name if case.surgeon else 'N/A'],
                ['Date:', case.procedure_date.strftime('%Y-%m-%d %H:%M')],
                ['Procedure:', case.procedure_type],
                ['Duration:', f"{case.actual_duration_min:.1f} minutes"],
                ['Estimated:', f"{case.estimated_duration_min:.1f} minutes"],
                ['Difference:', f"{((case.actual_duration_min - case.estimated_duration_min) / case.estimated_duration_min) * 100:+.1f}%"],
                ['Status:', case.processing_status.upper()]
            ]
            
            t = Table(case_info, colWidths=[2*inch, 4*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * inch))
            
            # Phase breakdown
            story.append(Paragraph("PHASE BREAKDOWN", self.styles['SectionHeader']))
            
            phase_data = [['Phase Name', 'Duration (min)', 'Frames', 'Anchor #']]
            for phase in sorted(case.phases, key=lambda p: p.start_frame):
                phase_data.append([
                    phase.phase_name,
                    f"{phase.duration_sec / 60:.1f}",
                    f"{phase.start_frame} - {phase.end_frame}",
                    str(phase.anchor_number) if phase.anchor_number else '-'
                ])
            
            t = Table(phase_data, colWidths=[2.5*inch, 1.2*inch, 1.5*inch, 1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * inch))
            
            # Timeline visualization
            timeline_img = self.create_timeline_image(case)
            if timeline_img:
                story.append(Paragraph("TIMELINE VISUALIZATION", self.styles['SectionHeader']))
                story.append(Image(timeline_img, width=6.5*inch, height=2*inch))
                story.append(Spacer(1, 0.3 * inch))
            
            # Events
            story.append(Paragraph("EVENT LOG", self.styles['SectionHeader']))
            
            event_data = [['Time', 'Event Type', 'Details']]
            for event in sorted(case.events, key=lambda e: e.event_frame):
                time_str = f"{int(event.event_time_sec//60):02d}:{int(event.event_time_sec%60):02d}"
                details = []
                if event.severity:
                    details.append(f"Severity: {event.severity}")
                if event.anchor_number:
                    details.append(f"Anchor #{event.anchor_number}")
                if event.outcome:
                    details.append(f"Outcome: {event.outcome}")
                
                event_data.append([
                    time_str,
                    event.event_type,
                    ", ".join(details) if details else '-'
                ])
            
            t = Table(event_data, colWidths=[1*inch, 2*inch, 3.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * inch))
            
            # Performance metrics
            story.append(Paragraph("PERFORMANCE METRICS", self.styles['SectionHeader']))
            
            # Calculate metrics
            suture_events = [e for e in case.events if e.event_type == 'Suture Attempt']
            bleeding_events = [e for e in case.events if e.event_type == 'Bleeding']
            
            metrics_data = []
            if suture_events:
                success = len([e for e in suture_events if e.outcome == 'Success'])
                total = len(suture_events)
                metrics_data.append(['Suture Success Rate:', f"{success}/{total} ({success/total*100:.1f}%)"])
            
            if bleeding_events:
                severe = len([e for e in bleeding_events if e.severity == 'Severe'])
                moderate = len([e for e in bleeding_events if e.severity == 'Moderate'])
                mild = len([e for e in bleeding_events if e.severity == 'Mild'])
                metrics_data.append(['Bleeding Events:', f"{len(bleeding_events)} (Severe: {severe}, Moderate: {moderate}, Mild: {mild})"])
            
            if case.resources:
                metrics_data.append(['Implants Used:', str(case.resources.implants_count)])
                metrics_data.append(['Disposables Used:', str(case.resources.disposables_count)])
                metrics_data.append(['Electrocautery Usage:', f"{case.resources.electrocautery_usage_percent:.1f}%"])
                metrics_data.append(['Anchor Repositions:', str(case.resources.anchor_repositions)])
            
            t = Table(metrics_data, colWidths=[2.5*inch, 4*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)
            
            # Footer
            story.append(Spacer(1, 0.5 * inch))
            footer_text = f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Surgical Analysis Platform v1.0"
            story.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            print(f"✅ PDF report saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            session.close()
    
    def create_timeline_image(self, case):
        """Create timeline visualization as image"""
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            
            # Colors for different phases
            phase_colors = {
                'Portal Placement': '#9C27B0',
                'Diagnostic Arthroscopy': '#3F51B5',
                'Labral Mobilization': '#2196F3',
                'Glenoid Preparation': '#00BCD4',
                'Anchor Placement': '#4CAF50',
                'Suture Passage': '#FFC107',
                'Suture Tensioning': '#FF9800',
                'Final Inspection': '#795548'
            }
            
            # Draw phases as horizontal bars
            total_frames = case.total_frames if case.total_frames else 1
            
            for phase in case.phases:
                start_pct = phase.start_frame / total_frames
                width_pct = (phase.end_frame - phase.start_frame) / total_frames
                
                color = phase_colors.get(phase.phase_name, '#999999')
                ax.barh(0, width_pct, left=start_pct, height=0.6, 
                       color=color, edgecolor='white', linewidth=0.5)
            
            # Add event markers
            event_markers = {
                'Bleeding': ('v', '#F44336', 10),
                'Suture Attempt': ('^', '#4CAF50', 10)
            }
            
            for event in case.events:
                pos_pct = event.event_frame / total_frames
                marker, color, size = event_markers.get(event.event_type, ('o', '#999999', 8))
                ax.scatter(pos_pct, -0.3, marker=marker, color=color, s=size**2, 
                          edgecolors='white', linewidths=0.5, zorder=10)
            
            # Formatting
            ax.set_ylim(-0.8, 0.8)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Surgery Timeline', fontsize=10)
            ax.set_title('Phase and Event Timeline', fontsize=12, fontweight='bold')
            
            # X-axis in minutes
            duration_min = case.actual_duration_min if case.actual_duration_min else 60
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels([
                '0:00',
                f'{int(duration_min*0.25)}:00',
                f'{int(duration_min*0.5)}:00',
                f'{int(duration_min*0.75)}:00',
                f'{int(duration_min)}:00'
            ])
            
            # Legend for phases
            legend_elements = [mpatches.Patch(color=color, label=name) 
                              for name, color in phase_colors.items() 
                              if any(p.phase_name == name for p in case.phases)]
            
            # Add event markers to legend
            for event_type in ['Bleeding', 'Suture Attempt']:
                if any(e.event_type == event_type for e in case.events):
                    marker, color, size = event_markers[event_type]
                    legend_elements.append(mpatches.Patch(color=color, label=event_type))
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0), 
                     fontsize=8, frameon=False)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return img_buffer
            
        except Exception as e:
            print(f"Error creating timeline image: {e}")
            return None
    
    def generate_surgeon_report_pdf(self, surgeon_id, output_path=None):
        """Generate PDF report for a surgeon"""
        from app.models.database import Surgeon, Case
        from sqlalchemy import func
        
        session = self.db.get_session()
        
        try:
            surgeon = session.query(Surgeon).filter_by(surgeon_id=surgeon_id).first()
            if not surgeon:
                return None
            
            cases = session.query(Case).filter_by(
                surgeon_id=surgeon_id,
                processing_status='completed'
            ).all()
            
            if not cases:
                return None
            
            # Create output path
            if not output_path:
                os.makedirs('data/results', exist_ok=True)
                output_path = f'data/results/surgeon_{surgeon_id}_report.pdf'
            
            # Create PDF
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Title
            story.append(Paragraph("SURGEON PERFORMANCE REPORT", self.styles['CustomTitle']))
            story.append(Spacer(1, 0.2 * inch))
            
            # Surgeon info
            surgeon_info = [
                ['Surgeon:', surgeon.full_name],
                ['Department:', surgeon.department],
                ['Specialty:', surgeon.specialty],
                ['Total Cases:', str(len(cases))]
            ]
            
            t = Table(surgeon_info, colWidths=[2*inch, 4*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * inch))
            
            # Aggregate metrics
            story.append(Paragraph("AGGREGATE METRICS", self.styles['SectionHeader']))
            
            avg_duration = sum(c.actual_duration_min for c in cases) / len(cases)
            avg_estimated = sum(c.estimated_duration_min for c in cases) / len(cases)
            
            metrics = [
                ['Average Procedure Time:', f"{avg_duration:.1f} minutes"],
                ['Average Estimated Time:', f"{avg_estimated:.1f} minutes"],
                ['Difference:', f"{((avg_duration - avg_estimated) / avg_estimated) * 100:+.1f}%"]
            ]
            
            t = Table(metrics, colWidths=[2.5*inch, 3.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * inch))
            
            # Case list
            story.append(Paragraph("CASE LIST", self.styles['SectionHeader']))
            
            case_data = [['Case ID', 'Date', 'Duration', 'vs Est']]
            for case in sorted(cases, key=lambda c: c.procedure_date, reverse=True):
                diff = case.actual_duration_min - case.estimated_duration_min
                case_data.append([
                    case.case_id,
                    case.procedure_date.strftime('%Y-%m-%d'),
                    f"{case.actual_duration_min:.1f} min",
                    f"{diff:+.1f} min"
                ])
            
            t = Table(case_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
            ]))
            story.append(t)
            
            # Footer
            story.append(Spacer(1, 0.5 * inch))
            footer_text = f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Surgical Analysis Platform v1.0"
            story.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            print(f"✅ PDF report saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating surgeon PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            session.close()



