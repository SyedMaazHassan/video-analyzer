"""
Enhanced Desktop PyQt6 Application for Surgical Analysis Platform
With PDF reports, charts, timeline visualization, and async processing
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                             QTableWidgetItem, QTabWidget, QTextEdit, QLineEdit,
                             QComboBox, QFormLayout, QFileDialog, QMessageBox,
                             QProgressBar, QDialog, QScrollArea, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap

from app.models.database import DatabaseManager, Case, Surgeon
from app.services.mock_inference import MockInferenceEngine
from app.services.pdf_report_generator import PDFReportGenerator
from app.services.timeline_visualizer import TimelineVisualizer
from sqlalchemy.orm import joinedload
from sqlalchemy import func, desc
from datetime import datetime
import numpy as np
import subprocess
import platform


class ProcessingThread(QThread):
    """Thread for processing video - non-blocking"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, str)  # success, message, case_id
    
    def __init__(self, db, case_id, video_path):
        super().__init__()
        self.db = db
        self.case_id = case_id
        self.video_path = video_path
    
    def run(self):
        try:
            inference = MockInferenceEngine(self.db)
            
            def progress_callback(percent, message):
                self.progress.emit(percent, message)
            
            success = inference.process_video(self.case_id, self.video_path, progress_callback)
            
            if success:
                self.finished.emit(True, f"Case {self.case_id} processed successfully!", self.case_id)
            else:
                self.finished.emit(False, "Processing failed", self.case_id)
                
        except Exception as e:
            self.finished.emit(False, str(e), self.case_id)


class CaseDetailDialog(QDialog):
    """Large, resizable dialog for case details with timeline and charts"""
    
    def __init__(self, parent, db, case_id):
        super().__init__(parent)
        self.db = db
        self.case_id = case_id
        
        self.setWindowTitle(f"Case Details - {case_id}")
        self.resize(1000, 700)  # Large default size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set dialog background and colors
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QTableWidget {
                background-color: white;
                color: #333;
            }
            QFrame {
                background-color: #f8f9fa;
            }
        """)
        
        self.setup_ui()
        self.load_case_data()
    
    def setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Buttons at bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        pdf_btn = QPushButton("📄 Generate PDF Report")
        pdf_btn.clicked.connect(self.generate_pdf_report)
        pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
        """)
        button_layout.addWidget(pdf_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)  # Proper close using accept()
        close_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                border-radius: 5px;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_case_data(self):
        """Load and display case data"""
        session = self.db.get_session()
        
        try:
            case = session.query(Case).options(
                joinedload(Case.surgeon),
                joinedload(Case.phases),
                joinedload(Case.events),
                joinedload(Case.resources)
            ).filter_by(case_id=self.case_id).first()
            
            if not case:
                self.content_layout.addWidget(QLabel("Case not found"))
                return
            
            # Title
            title = QLabel(f"📋 Case: {case.case_id}")
            title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
            title.setStyleSheet("color: #667eea; padding: 10px;")
            self.content_layout.addWidget(title)
            
            # Case info card
            info_frame = QFrame()
            info_frame.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                }
                QLabel {
                    color: #333;
                }
            """)
            info_layout = QVBoxLayout(info_frame)
            
            info_layout.addWidget(QLabel(f"<b>Surgeon:</b> {case.surgeon.full_name if case.surgeon else 'N/A'}"))
            info_layout.addWidget(QLabel(f"<b>Date:</b> {case.procedure_date.strftime('%Y-%m-%d %H:%M')}"))
            info_layout.addWidget(QLabel(f"<b>Procedure:</b> {case.procedure_type}"))
            
            duration_text = f"<b>Duration:</b> {case.actual_duration_min:.1f} min"
            if case.estimated_duration_min:
                diff = ((case.actual_duration_min - case.estimated_duration_min) / case.estimated_duration_min) * 100
                duration_text += f" (Est: {case.estimated_duration_min:.1f} min, <span style='color: {'green' if abs(diff) < 10 else 'orange'};'>{diff:+.1f}%</span>)"
            info_layout.addWidget(QLabel(duration_text))
            
            self.content_layout.addWidget(info_frame)
            self.content_layout.addSpacing(15)
            
            # Timeline visualization
            timeline_label = QLabel("⏱️ Timeline Visualization")
            timeline_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            self.content_layout.addWidget(timeline_label)
            
            timeline_viz = TimelineVisualizer(self.db)
            timeline_img_path = f'data/results/timeline_{self.case_id}.png'
            timeline_viz.create_case_timeline(self.case_id, output_path=timeline_img_path)
            
            if os.path.exists(timeline_img_path):
                timeline_label_widget = QLabel()
                pixmap = QPixmap(timeline_img_path)
                scaled_pixmap = pixmap.scaledToWidth(950, Qt.TransformationMode.SmoothTransformation)
                timeline_label_widget.setPixmap(scaled_pixmap)
                self.content_layout.addWidget(timeline_label_widget)
            
            # Detailed Analytics Section
            analytics_label = QLabel("📊 Detailed Analytics")
            analytics_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            analytics_label.setStyleSheet("color: #000000;")
            self.content_layout.addWidget(analytics_label)
            
            # Create mini charts for this case
            try:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                
                # Phase Duration Breakdown
                phase_frame = QFrame()
                phase_frame.setStyleSheet("""
                    QFrame {
                        background-color: white;
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
                phase_layout = QVBoxLayout(phase_frame)
                
                phase_chart_title = QLabel("Phase Duration Breakdown")
                phase_chart_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                phase_chart_title.setStyleSheet("color: #000000;")
                phase_layout.addWidget(phase_chart_title)
                
                phase_fig = Figure(figsize=(10, 4))
                phase_canvas = FigureCanvas(phase_fig)
                ax_phase = phase_fig.add_subplot(111)
                
                # Plot phase durations
                if case.phases:
                    phase_names = [p.phase_name.replace(" (Anchor #", "\n#").replace(")", "") for p in case.phases]
                    durations = [p.duration_min for p in case.phases]
                    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
                    
                    bars = ax_phase.bar(range(len(phase_names)), durations, color=colors[:len(phase_names)])
                    ax_phase.set_xlabel('Phases')
                    ax_phase.set_ylabel('Duration (min)')
                    ax_phase.set_title(f'Phase Breakdown - {case_id}')
                    ax_phase.set_xticks(range(len(phase_names)))
                    ax_phase.set_xticklabels(phase_names, rotation=45, ha='right')
                    ax_phase.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, duration in zip(bars, durations):
                        height = bar.get_height()
                        ax_phase.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                    f'{duration:.1f}', ha='center', va='bottom', fontsize=8)
                else:
                    ax_phase.text(0.5, 0.5, 'No phase data available', ha='center', va='center', 
                                transform=ax_phase.transAxes, fontsize=12)
                
                phase_fig.tight_layout()
                phase_layout.addWidget(phase_canvas)
                self.content_layout.addWidget(phase_frame)
            except Exception as e:
                print(f"Error creating phase chart: {e}")
                error_label = QLabel(f"Error creating phase chart: {str(e)}")
                error_label.setStyleSheet("color: #ff0000;")
                self.content_layout.addWidget(error_label)
            
            # Event Timeline
            try:
                event_frame = QFrame()
                event_frame.setStyleSheet("""
                    QFrame {
                        background-color: white;
                        border: 2px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
                event_layout = QVBoxLayout(event_frame)
                
                event_chart_title = QLabel("Event Timeline")
                event_chart_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                event_chart_title.setStyleSheet("color: #000000;")
                event_layout.addWidget(event_chart_title)
                
                event_fig = Figure(figsize=(10, 4))
                event_canvas = FigureCanvas(event_fig)
                ax_event = event_fig.add_subplot(111)
                
                # Plot events over time
                if case.events:
                    event_times = [(e.frame_number / case.total_frames * case.actual_duration_min) if case.total_frames else 0 for e in case.events]
                    event_types = [e.event_type for e in case.events]
                    event_colors = {'Bleeding': '#ff6b6b', 'Suture Attempt': '#4ecdc4', 'Portal': '#45b7d1'}
                    
                    y_positions = [0.5 if et == 'Bleeding' else 1.5 if et == 'Suture Attempt' else 2.5 for et in event_types]
                    
                    for i, (time, y_pos, event_type) in enumerate(zip(event_times, y_positions, event_types)):
                        color = event_colors.get(event_type, '#999999')
                        ax_event.scatter(time, y_pos, c=color, s=100, alpha=0.7, edgecolors='black')
                        ax_event.annotate(event_type, (time, y_pos), xytext=(5, 5), 
                                        textcoords='offset points', fontsize=8)
                    
                    ax_event.set_xlabel('Time (minutes)')
                    ax_event.set_ylabel('Event Type')
                    ax_event.set_title(f'Event Distribution - {case_id}')
                    ax_event.set_yticks([0.5, 1.5, 2.5])
                    ax_event.set_yticklabels(['Bleeding', 'Suture Attempt', 'Portal'])
                    ax_event.grid(True, alpha=0.3)
                    ax_event.set_ylim(0, 3)
                else:
                    ax_event.text(0.5, 0.5, 'No event data available', ha='center', va='center', 
                                transform=ax_event.transAxes, fontsize=12)
                
                event_fig.tight_layout()
                event_layout.addWidget(event_canvas)
                self.content_layout.addWidget(event_frame)
            except Exception as e:
                print(f"Error creating event chart: {e}")
                error_label = QLabel(f"Error creating event chart: {str(e)}")
                error_label.setStyleSheet("color: #ff0000;")
                self.content_layout.addWidget(error_label)
            
            self.content_layout.addSpacing(15)
            
            # Phases
            phases_label = QLabel(f"🔷 Phases ({len(case.phases)})")
            phases_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            self.content_layout.addWidget(phases_label)
            
            phases_table = QTableWidget()
            phases_table.setColumnCount(4)
            phases_table.setHorizontalHeaderLabels(['Phase Name', 'Duration', 'Frames', 'Anchor #'])
            phases_table.setRowCount(len(case.phases))
            phases_table.horizontalHeader().setStretchLastSection(True)
            
            for row, phase in enumerate(sorted(case.phases, key=lambda p: p.start_frame)):
                phases_table.setItem(row, 0, QTableWidgetItem(phase.phase_name))
                phases_table.setItem(row, 1, QTableWidgetItem(f"{phase.duration_sec/60:.1f} min"))
                phases_table.setItem(row, 2, QTableWidgetItem(f"{phase.start_frame} - {phase.end_frame}"))
                phases_table.setItem(row, 3, QTableWidgetItem(str(phase.anchor_number) if phase.anchor_number else '-'))
            
            phases_table.setMaximumHeight(300)
            self.content_layout.addWidget(phases_table)
            self.content_layout.addSpacing(15)
            
            # Events
            events_label = QLabel(f"⚡ Events ({len(case.events)})")
            events_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            self.content_layout.addWidget(events_label)
            
            events_table = QTableWidget()
            events_table.setColumnCount(3)
            events_table.setHorizontalHeaderLabels(['Time', 'Event Type', 'Details'])
            events_table.setRowCount(len(case.events))
            events_table.horizontalHeader().setStretchLastSection(True)
            
            for row, event in enumerate(sorted(case.events, key=lambda e: e.event_frame)):
                time_str = f"{int(event.event_time_sec//60):02d}:{int(event.event_time_sec%60):02d}"
                events_table.setItem(row, 0, QTableWidgetItem(time_str))
                events_table.setItem(row, 1, QTableWidgetItem(event.event_type))
                
                details = []
                if event.severity:
                    details.append(f"Severity: {event.severity}")
                if event.anchor_number:
                    details.append(f"Anchor #{event.anchor_number}")
                if event.outcome:
                    details.append(f"Outcome: {event.outcome}")
                
                events_table.setItem(row, 2, QTableWidgetItem(', '.join(details) if details else '-'))
            
            events_table.setMaximumHeight(250)
            self.content_layout.addWidget(events_table)
            self.content_layout.addSpacing(15)
            
            # Resources
            if case.resources:
                resources_label = QLabel("💰 Resources")
                resources_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
                resources_label.setStyleSheet("color: #333;")
                self.content_layout.addWidget(resources_label)
                
                resources_frame = QFrame()
                resources_frame.setStyleSheet("""
                    QFrame {
                        background-color: #f8f9fa;
                        border-radius: 8px;
                        padding: 15px;
                    }
                    QLabel {
                        color: #333;
                    }
                """)
                resources_layout = QVBoxLayout(resources_frame)
                
                resources_layout.addWidget(QLabel(f"<b>Implants:</b> {case.resources.implants_count}"))
                resources_layout.addWidget(QLabel(f"<b>Disposables:</b> {case.resources.disposables_count}"))
                resources_layout.addWidget(QLabel(f"<b>Electrocautery:</b> {case.resources.electrocautery_usage_percent:.1f}%"))
                resources_layout.addWidget(QLabel(f"<b>Anchor Repositions:</b> {case.resources.anchor_repositions}"))
                
                self.content_layout.addWidget(resources_frame)
            
            self.content_layout.addStretch()
            
        except Exception as e:
            self.content_layout.addWidget(QLabel(f"Error loading case: {str(e)}"))
            import traceback
            traceback.print_exc()
        finally:
            session.close()
    
    def generate_pdf_report(self):
        """Generate and open PDF report"""
        try:
            pdf_gen = PDFReportGenerator(self.db)
            pdf_path = pdf_gen.generate_case_report_pdf(self.case_id)
            
            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(self, "Success", f"PDF report generated:\n{pdf_path}")
                
                # Try to open PDF
                self.open_file(pdf_path)
            else:
                QMessageBox.warning(self, "Error", "Failed to generate PDF report")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def open_file(self, filepath):
        """Open file with default application"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':
                os.startfile(filepath)
            else:  # Linux
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            print(f"Could not open file: {e}")


class ProcessingStatusWidget(QWidget):
    """Widget showing processing queue status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.processing_cases = {}  # case_id -> progress_bar
        
        layout = QVBoxLayout(self)
        
        self.title = QLabel("🔄 Processing Queue")
        self.title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(self.title)
        
        self.queue_layout = QVBoxLayout()
        layout.addLayout(self.queue_layout)
        
        self.no_processing_label = QLabel("No cases currently processing")
        self.no_processing_label.setStyleSheet("color: #999; padding: 20px;")
        layout.addWidget(self.no_processing_label)
        
        layout.addStretch()
    
    def add_processing_case(self, case_id):
        """Add a case to processing queue"""
        self.no_processing_label.hide()
        
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f0f4ff;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        frame_layout = QVBoxLayout(frame)
        
        label = QLabel(f"Processing: {case_id}")
        label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        frame_layout.addWidget(label)
        
        progress = QProgressBar()
        progress.setTextVisible(True)
        progress.setFormat("%p% - Initializing...")
        frame_layout.addWidget(progress)
        
        self.queue_layout.addWidget(frame)
        self.processing_cases[case_id] = {'frame': frame, 'progress': progress, 'label': label}
    
    def update_progress(self, case_id, percent, message):
        """Update progress for a case"""
        if case_id in self.processing_cases:
            self.processing_cases[case_id]['progress'].setValue(percent)
            self.processing_cases[case_id]['progress'].setFormat(f"%p% - {message}")
    
    def remove_case(self, case_id):
        """Remove case from queue"""
        if case_id in self.processing_cases:
            frame = self.processing_cases[case_id]['frame']
            self.queue_layout.removeWidget(frame)
            frame.deleteLater()
            del self.processing_cases[case_id]
            
            if not self.processing_cases:
                self.no_processing_label.show()


class SurgicalAnalysisApp(QMainWindow):
    """Enhanced Main Desktop Application"""
    
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager('data/surgical_analysis.db')
        self.processing_threads = []
        
        self.setWindowTitle("🏥 Surgical Analysis Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set window properties
        self.setMinimumSize(1000, 600)
        
        # Set global stylesheet with better contrast
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QWidget {
                background-color: transparent;
                color: #000000;
            }
            QLabel {
                color: #000000;
            }
            QTableWidget {
                background-color: white;
                color: #000000;
                gridline-color: #e0e0e0;
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #000000;
                font-weight: bold;
                padding: 8px;
                border: 1px solid #ddd;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: white;
                color: #000000;
                border: 2px solid #ddd;
                padding: 6px;
                border-radius: 4px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #667eea;
            }
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6fd8;
            }
            QPushButton:pressed {
                background-color: #4e63d2;
            }
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        
        # Create UI
        self.setup_ui()
        
        # Load data
        self.refresh_cases()
        self.refresh_surgeons()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QWidget()
        header.setStyleSheet("background-color: #667eea; padding: 20px;")
        header_layout = QVBoxLayout(header)
        
        title = QLabel("🏥 Surgical Analysis Platform")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("AI-Powered Surgical Video Analysis & Reporting")
        subtitle.setStyleSheet("color: white; font-size: 14px;")
        header_layout.addWidget(subtitle)
        
        main_layout.addWidget(header)
        
        # Content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Main tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #333;
                background-color: #f5f5f5;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 3px solid #667eea;
                color: #667eea;
            }
            QTabBar::tab:!selected {
                background-color: #e0e0e0;
            }
            QTableWidget {
                background-color: white;
                color: #333;
            }
            QLabel {
                color: #333;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: white;
                color: #333;
            }
        """)
        content_layout.addWidget(self.tabs, stretch=7)
        
        # Processing queue sidebar
        self.processing_widget = ProcessingStatusWidget()
        self.processing_widget.setMaximumWidth(350)
        self.processing_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-left: 1px solid #ddd;
                padding: 10px;
            }
            QLabel {
                color: #333;
            }
        """)
        content_layout.addWidget(self.processing_widget, stretch=3)
        
        main_layout.addWidget(content_widget)
        
        # Create tabs
        self.create_cases_tab()
        self.create_analytics_tab()  # Add analytics tab with graphs
        self.create_surgeons_tab()
        self.create_upload_tab()
        self.create_add_surgeon_tab()
    
    def create_cases_tab(self):
        """Cases list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with refresh button
        header_layout = QHBoxLayout()
        header = QLabel("📋 All Cases")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(header)
        header_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_cases)
        refresh_btn.setStyleSheet("padding: 8px 16px; border-radius: 4px;")
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)
        
        # Table
        self.cases_table = QTableWidget()
        self.cases_table.setColumnCount(6)
        self.cases_table.setHorizontalHeaderLabels([
            "Case ID", "Surgeon", "Date", "Duration (min)", "Status", "Actions"
        ])
        self.cases_table.horizontalHeader().setStretchLastSection(True)
        self.cases_table.setAlternatingRowColors(True)
        self.cases_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 8px;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 8px;
            }
        """)
        layout.addWidget(self.cases_table)
        
        self.tabs.addTab(tab, "📋 Cases")
    
    def create_surgeons_tab(self):
        """Surgeons list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("👨‍⚕️ Surgeons")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Table
        self.surgeons_table = QTableWidget()
        self.surgeons_table.setColumnCount(6)
        self.surgeons_table.setHorizontalHeaderLabels([
            "ID", "Name", "Department", "Specialty", "Cases", "Actions"
        ])
        self.surgeons_table.horizontalHeader().setStretchLastSection(True)
        self.surgeons_table.setAlternatingRowColors(True)
        self.surgeons_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 8px;
                gridline-color: #e0e0e0;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.surgeons_table)
        
        self.tabs.addTab(tab, "👨‍⚕️ Surgeons")
    
    def create_analytics_tab(self):
        """Analytics tab with graphs"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("📊 Analytics & Insights")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #667eea; padding: 10px;")
        layout.addWidget(header)
        
        # Create scroll area for graphs
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Phase Duration Chart
        phase_frame = QFrame()
        phase_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        phase_layout = QVBoxLayout(phase_frame)
        
        phase_title = QLabel("📈 Average Phase Durations by Surgeon")
        phase_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        phase_title.setStyleSheet("color: #000000; margin-bottom: 10px;")
        phase_layout.addWidget(phase_title)
        
        # Create matplotlib widget for phase chart
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        
        self.phase_fig = Figure(figsize=(12, 6))
        self.phase_canvas = FigureCanvas(self.phase_fig)
        phase_layout.addWidget(self.phase_canvas)
        
        scroll_layout.addWidget(phase_frame)
        
        # Event Distribution Chart
        event_frame = QFrame()
        event_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        event_layout = QVBoxLayout(event_frame)
        
        event_title = QLabel("🥧 Event Distribution")
        event_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        event_title.setStyleSheet("color: #000000; margin-bottom: 10px;")
        event_layout.addWidget(event_title)
        
        # Create matplotlib widget for event chart
        self.event_fig = Figure(figsize=(12, 6))
        self.event_canvas = FigureCanvas(self.event_fig)
        event_layout.addWidget(self.event_canvas)
        
        scroll_layout.addWidget(event_frame)
        
        # Performance Metrics
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        metrics_layout = QVBoxLayout(metrics_frame)
        
        metrics_title = QLabel("📊 Performance Metrics")
        metrics_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        metrics_title.setStyleSheet("color: #000000; margin-bottom: 10px;")
        metrics_layout.addWidget(metrics_title)
        
        # Create metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Surgeon", "Avg Duration", "Total Cases", "Success Rate"])
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                color: #000000;
                gridline-color: #e0e0e0;
                border: 1px solid #ddd;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #000000;
                font-weight: bold;
                padding: 8px;
            }
        """)
        metrics_layout.addWidget(self.metrics_table)
        
        scroll_layout.addWidget(metrics_frame)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Analytics")
        refresh_btn.clicked.connect(self.refresh_analytics)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5a6fd8;
            }
        """)
        layout.addWidget(refresh_btn)
        
        self.tabs.addTab(tab, "📊 Analytics")
        
        # Load initial data after a short delay to ensure UI is ready
        QTimer.singleShot(500, self.refresh_analytics)
    
    def create_upload_tab(self):
        """Upload new case tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        header = QLabel("➕ Upload New Case")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Form container
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 12px;
                padding: 30px;
            }
            QLabel {
                color: #333;
            }
        """)
        form_frame.setMaximumWidth(600)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Add label style
        label_style = "color: #333; font-weight: bold;"
        
        case_id_label = QLabel("Case ID:")
        case_id_label.setStyleSheet(label_style)
        self.case_id_input = QLineEdit()
        self.case_id_input.setPlaceholderText("e.g., CASE12345")
        self.case_id_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #333;")
        form_layout.addRow(case_id_label, self.case_id_input)
        
        surgeon_label = QLabel("Surgeon:")
        surgeon_label.setStyleSheet(label_style)
        self.surgeon_combo = QComboBox()
        self.surgeon_combo.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #333;")
        form_layout.addRow(surgeon_label, self.surgeon_combo)
        
        procedure_label = QLabel("Procedure Type:")
        procedure_label.setStyleSheet(label_style)
        self.procedure_input = QLineEdit()
        self.procedure_input.setText("Labral Repair")
        self.procedure_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #333;")
        form_layout.addRow(procedure_label, self.procedure_input)
        
        video_label = QLabel("Video File:")
        video_label.setStyleSheet(label_style)
        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_path_input = QLineEdit()
        self.video_path_input.setReadOnly(True)
        self.video_path_input.setPlaceholderText("Click Browse to select video file...")
        self.video_path_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #333;")
        video_layout.addWidget(self.video_path_input)
        browse_btn = QPushButton("📁 Browse Video...")
        browse_btn.clicked.connect(self.browse_video)
        browse_btn.setStyleSheet("padding: 8px 16px; border-radius: 4px; background-color: #667eea; color: white;")
        video_layout.addWidget(browse_btn)
        form_layout.addRow(video_label, video_widget)
        
        notes_label = QLabel("Notes:")
        notes_label.setStyleSheet(label_style)
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)
        self.notes_input.setPlaceholderText("Additional notes...")
        self.notes_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #333;")
        form_layout.addRow(notes_label, self.notes_input)
        
        layout.addWidget(form_frame)
        
        # Upload button
        upload_btn = QPushButton("🚀 Upload and Process Case (Async)")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
        """)
        upload_btn.setMaximumWidth(600)
        upload_btn.clicked.connect(self.upload_case_async)
        layout.addWidget(upload_btn)
        
        layout.addStretch()
        
        self.tabs.addTab(tab, "➕ Upload Case")
    
    def create_add_surgeon_tab(self):
        """Add surgeon tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        header = QLabel("➕ Add New Surgeon")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Form container
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 12px;
                padding: 30px;
            }
            QLabel {
                color: #000000;
                font-weight: bold;
            }
        """)
        form_frame.setMaximumWidth(600)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        label_style = "color: #000000; font-weight: bold;"
        
        surgeon_id_label = QLabel("Surgeon ID:")
        surgeon_id_label.setStyleSheet(label_style)
        self.surgeon_id_input = QLineEdit()
        self.surgeon_id_input.setPlaceholderText("e.g., S004")
        self.surgeon_id_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(surgeon_id_label, self.surgeon_id_input)
        
        first_name_label = QLabel("First Name:")
        first_name_label.setStyleSheet(label_style)
        self.first_name_input = QLineEdit()
        self.first_name_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(first_name_label, self.first_name_input)
        
        last_name_label = QLabel("Last Name:")
        last_name_label.setStyleSheet(label_style)
        self.last_name_input = QLineEdit()
        self.last_name_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(last_name_label, self.last_name_input)
        
        department_label = QLabel("Department:")
        department_label.setStyleSheet(label_style)
        self.department_input = QLineEdit()
        self.department_input.setText("Orthopedics")
        self.department_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(department_label, self.department_input)
        
        specialty_label = QLabel("Specialty:")
        specialty_label.setStyleSheet(label_style)
        self.specialty_input = QLineEdit()
        self.specialty_input.setText("Sports Medicine")
        self.specialty_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(specialty_label, self.specialty_input)
        
        email_label = QLabel("Email:")
        email_label.setStyleSheet(label_style)
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("surgeon@hospital.com")
        self.email_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: white; color: #000000;")
        form_layout.addRow(email_label, self.email_input)
        
        layout.addWidget(form_frame)
        
        # Add button
        add_btn = QPushButton("➕ Add Surgeon")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
        """)
        add_btn.setMaximumWidth(600)
        add_btn.clicked.connect(self.add_surgeon)
        layout.addWidget(add_btn)
        
        layout.addStretch()
        
        self.tabs.addTab(tab, "➕ Add Surgeon")
    
    def refresh_cases(self):
        """Refresh cases table"""
        session = self.db.get_session()
        try:
            cases = session.query(Case).options(joinedload(Case.surgeon)).order_by(
                Case.procedure_date.desc()
            ).all()
            
            self.cases_table.setRowCount(len(cases))
            
            for row, case in enumerate(cases):
                self.cases_table.setItem(row, 0, QTableWidgetItem(case.case_id))
                self.cases_table.setItem(row, 1, QTableWidgetItem(
                    case.surgeon.full_name if case.surgeon else "N/A"
                ))
                self.cases_table.setItem(row, 2, QTableWidgetItem(
                    case.procedure_date.strftime('%Y-%m-%d %H:%M')
                ))
                self.cases_table.setItem(row, 3, QTableWidgetItem(
                    f"{case.actual_duration_min:.1f}" if case.actual_duration_min else "Processing..."
                ))
                
                status_item = QTableWidgetItem(case.processing_status.upper())
                if case.processing_status == 'completed':
                    status_item.setForeground(Qt.GlobalColor.darkGreen)
                elif case.processing_status == 'processing':
                    status_item.setForeground(Qt.GlobalColor.blue)
                self.cases_table.setItem(row, 4, status_item)
                
                # Action buttons
                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.setContentsMargins(5, 2, 5, 2)
                
                if case.processing_status == 'completed':
                    # PDF Report button only
                    pdf_btn = QPushButton("📄 Generate PDF Report")
                    pdf_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #28a745;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 6px;
                            font-weight: bold;
                            font-size: 12px;
                        }
                        QPushButton:hover {
                            background-color: #218838;
                        }
                        QPushButton:pressed {
                            background-color: #1e7e34;
                        }
                    """)
                    pdf_btn.clicked.connect(lambda checked, cid=case.case_id: self.generate_pdf_report(cid))
                    btn_layout.addWidget(pdf_btn)
                else:
                    pending_label = QLabel("⏳ Processing...")
                    pending_label.setStyleSheet("color: #999; font-weight: bold;")
                    btn_layout.addWidget(pending_label)
                
                self.cases_table.setCellWidget(row, 5, btn_widget)
            
            self.cases_table.resizeColumnsToContents()
            
        finally:
            session.close()
    
    def refresh_analytics(self):
        """Refresh analytics charts and data"""
        try:
            session = self.db.get_session()
            
            # Phase Duration Chart
            try:
                self.phase_fig.clear()
                ax1 = self.phase_fig.add_subplot(111)
                
                # Get phase data by surgeon
                surgeons = session.query(Surgeon).all()
                phase_names = ["Portal Placement", "Diagnostic Arthroscopy", "Glenoid Preparation", 
                              "Anchor Placement", "Suture Passage", "Suture Management", "Final Inspection"]
                
                if surgeons:
                    x = np.arange(len(phase_names))
                    width = 0.2
                    
                    for i, surgeon in enumerate(surgeons[:4]):  # Limit to 4 surgeons
                        durations = []
                        for phase_name in phase_names:
                            avg_duration = session.query(func.avg(Phase.duration_min)).join(Case).filter(
                                Case.surgeon_id == surgeon.surgeon_id,
                                Phase.phase_name.like(f"%{phase_name}%"),
                                Case.processing_status == 'completed'
                            ).scalar() or 0
                            durations.append(avg_duration)
                        
                        ax1.bar(x + i * width, durations, width, label=surgeon.full_name[:15])
                    
                    ax1.set_xlabel('Phases')
                    ax1.set_ylabel('Duration (minutes)')
                    ax1.set_title('Average Phase Durations by Surgeon')
                    ax1.set_xticks(x + width * 1.5)
                    ax1.set_xticklabels([name.replace(' ', '\\n') for name in phase_names], rotation=45, ha='right')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No surgeon data available', ha='center', va='center', 
                           transform=ax1.transAxes, fontsize=12)
                
                self.phase_fig.tight_layout()
                self.phase_canvas.draw()
            except Exception as e:
                print(f"Error creating phase chart: {e}")
            
            # Event Distribution Chart
            try:
                self.event_fig.clear()
                ax2 = self.event_fig.add_subplot(111)
                
                # Get event counts
                events = session.query(Event.event_type, func.count(Event.id)).group_by(Event.event_type).all()
                if events:
                    event_types, counts = zip(*events)
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
                    ax2.pie(counts, labels=event_types, autopct='%1.1f%%', colors=colors[:len(event_types)])
                    ax2.set_title('Event Distribution')
                else:
                    ax2.text(0.5, 0.5, 'No event data available', ha='center', va='center', 
                           transform=ax2.transAxes, fontsize=12)
                
                self.event_fig.tight_layout()
                self.event_canvas.draw()
            except Exception as e:
                print(f"Error creating event chart: {e}")
            
            # Performance Metrics Table
            try:
                self.metrics_table.setRowCount(len(surgeons))
                for row, surgeon in enumerate(surgeons):
                    # Get metrics
                    avg_duration = session.query(func.avg(Case.actual_duration_min)).filter(
                        Case.surgeon_id == surgeon.surgeon_id,
                        Case.processing_status == 'completed'
                    ).scalar() or 0
                    
                    total_cases = session.query(Case).filter(
                        Case.surgeon_id == surgeon.surgeon_id,
                        Case.processing_status == 'completed'
                    ).count()
                    
                    success_rate = 95.0 + (row * 2)  # Mock success rate
                    
                    self.metrics_table.setItem(row, 0, QTableWidgetItem(surgeon.full_name))
                    self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{avg_duration:.1f}"))
                    self.metrics_table.setItem(row, 2, QTableWidgetItem(str(total_cases)))
                    self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{success_rate:.1f}%"))
                
                self.metrics_table.resizeColumnsToContents()
            except Exception as e:
                print(f"Error updating metrics table: {e}")
            
        except Exception as e:
            print(f"Error refreshing analytics: {e}")
        finally:
            session.close()
    
    def generate_pdf_report(self, case_id):
        """Generate and open PDF report for a case"""
        try:
            pdf_gen = PDFReportGenerator(self.db)
            pdf_path = pdf_gen.generate_case_report_pdf(case_id)
            
            if pdf_path and os.path.exists(pdf_path):
                # Open PDF with default application
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', pdf_path])
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(pdf_path)
                else:  # Linux
                    subprocess.run(['xdg-open', pdf_path])
                
                QMessageBox.information(self, "PDF Generated", 
                    f"PDF report for case {case_id} has been generated and opened!")
            else:
                QMessageBox.warning(self, "Error", "Failed to generate PDF report.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating PDF: {str(e)}")
    
    def refresh_surgeons(self):
        """Refresh surgeons table and combo box"""
        session = self.db.get_session()
        try:
            surgeons = session.query(Surgeon).all()
            
            # Update table
            self.surgeons_table.setRowCount(len(surgeons))
            
            # Update combo box
            self.surgeon_combo.clear()
            
            for row, surgeon in enumerate(surgeons):
                self.surgeons_table.setItem(row, 0, QTableWidgetItem(surgeon.surgeon_id))
                self.surgeons_table.setItem(row, 1, QTableWidgetItem(surgeon.full_name))
                self.surgeons_table.setItem(row, 2, QTableWidgetItem(surgeon.department))
                self.surgeons_table.setItem(row, 3, QTableWidgetItem(surgeon.specialty))
                
                # Get case count
                case_count = session.query(Case).filter_by(
                    surgeon_id=surgeon.surgeon_id,
                    processing_status='completed'
                ).count()
                self.surgeons_table.setItem(row, 4, QTableWidgetItem(str(case_count)))
                
                # Action button
                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.setContentsMargins(5, 2, 5, 2)
                
                report_btn = QPushButton("View Report")
                report_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #667eea;
                        color: white;
                        padding: 6px 12px;
                        border-radius: 4px;
                    }
                    QPushButton:hover {
                        background-color: #5568d3;
                    }
                """)
                report_btn.clicked.connect(lambda checked, sid=surgeon.surgeon_id: self.view_surgeon_report(sid))
                btn_layout.addWidget(report_btn)
                
                self.surgeons_table.setCellWidget(row, 5, btn_widget)
                
                # Add to combo box
                self.surgeon_combo.addItem(surgeon.full_name, surgeon.surgeon_id)
            
            self.surgeons_table.resizeColumnsToContents()
            
        finally:
            session.close()
    
    def browse_video(self):
        """Browse for video file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_name:
            self.video_path_input.setText(file_name)
    
    def upload_case_async(self):
        """Upload and process case asynchronously (non-blocking)"""
        case_id = self.case_id_input.text().strip()
        surgeon_id = self.surgeon_combo.currentData()
        procedure_type = self.procedure_input.text().strip()
        video_path = self.video_path_input.text().strip() or 'videos/placeholder.mp4'
        notes = self.notes_input.toPlainText().strip()
        
        if not case_id:
            QMessageBox.warning(self, "Error", "Please enter a Case ID")
            return
        
        if not surgeon_id:
            QMessageBox.warning(self, "Error", "Please select a surgeon")
            return
        
        # Check if case already exists
        session = self.db.get_session()
        try:
            existing_case = session.query(Case).filter(Case.case_id == case_id).first()
            
            if existing_case:
                QMessageBox.warning(self, "Case Already Exists", 
                                  f'Case ID "{case_id}" already exists in the database.\n\nPlease use a different Case ID.')
                return
            
            # Create case
            case = Case(
                case_id=case_id,
                surgeon_id=surgeon_id,
                procedure_type=procedure_type,
                procedure_date=datetime.now(),
                video_path=video_path,
                estimated_duration_min=52.0,
                processing_status='queued',
                notes=notes
            )
            session.add(case)
            session.commit()
            
            QMessageBox.information(self, "Queued", 
                f"Case {case_id} has been queued for processing.\n\n"
                "You can continue using the application.\n"
                "You'll be notified when processing is complete.")
            
            # Clear form
            self.case_id_input.clear()
            self.video_path_input.clear()
            self.notes_input.clear()
            
        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "Error", f"Failed to create case: {str(e)}")
            return
        finally:
            session.close()
        
        # Add to processing queue widget
        self.processing_widget.add_processing_case(case_id)
        
        # Start processing in background thread
        thread = ProcessingThread(self.db, case_id, video_path)
        
        def update_progress(percent, message):
            self.processing_widget.update_progress(case_id, percent, message)
        
        def processing_done(success, message, cid):
            self.processing_widget.remove_case(cid)
            if success:
                QMessageBox.information(self, "Complete", 
                    f"✅ {message}\n\nClick 'Refresh' in Cases tab to see the new case.")
                self.refresh_cases()
            else:
                QMessageBox.warning(self, "Failed", f"❌ Processing failed: {message}")
        
        thread.progress.connect(update_progress)
        thread.finished.connect(processing_done)
        thread.start()
        
        # Keep reference to avoid garbage collection
        self.processing_threads.append(thread)
    
    def add_surgeon(self):
        """Add new surgeon"""
        surgeon_id = self.surgeon_id_input.text().strip()
        first_name = self.first_name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        department = self.department_input.text().strip()
        specialty = self.specialty_input.text().strip()
        email = self.email_input.text().strip()
        
        if not all([surgeon_id, first_name, last_name]):
            QMessageBox.warning(self, "Error", "Please fill in Surgeon ID, First Name, and Last Name")
            return
        
        session = self.db.get_session()
        try:
            # Check if exists
            existing = session.query(Surgeon).filter_by(surgeon_id=surgeon_id).first()
            if existing:
                QMessageBox.warning(self, "Error", f"Surgeon with ID {surgeon_id} already exists")
                return
            
            surgeon = Surgeon(
                surgeon_id=surgeon_id,
                first_name=first_name,
                last_name=last_name,
                department=department,
                specialty=specialty,
                email=email
            )
            session.add(surgeon)
            session.commit()
            
            QMessageBox.information(self, "Success", f"✅ Surgeon {surgeon.full_name} added successfully!")
            
            # Clear form
            self.surgeon_id_input.clear()
            self.first_name_input.clear()
            self.last_name_input.clear()
            self.email_input.clear()
            
            # Refresh
            self.refresh_surgeons()
            
        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "Error", f"Failed to add surgeon: {str(e)}")
        finally:
            session.close()
    
    def view_case_details(self, case_id):
        """View case details in large dialog"""
        dialog = CaseDetailDialog(self, self.db, case_id)
        dialog.exec()
    
    def view_surgeon_report(self, surgeon_id):
        """View surgeon report and generate PDF"""
        try:
            pdf_gen = PDFReportGenerator(self.db)
            pdf_path = pdf_gen.generate_surgeon_report_pdf(surgeon_id)
            
            if pdf_path and os.path.exists(pdf_path):
                reply = QMessageBox.question(self, "PDF Generated", 
                    f"PDF report generated:\n{pdf_path}\n\nOpen the file?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.open_file(pdf_path)
            else:
                QMessageBox.warning(self, "Error", "Failed to generate PDF report")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def open_file(self, filepath):
        """Open file with default application"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':
                os.startfile(filepath)
            else:  # Linux
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            QMessageBox.information(self, "File Location", f"File saved at:\n{filepath}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set application-wide style
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QMainWindow {
            background-color: #f5f7fa;
        }
    """)
    
    window = SurgicalAnalysisApp()
    window.show()
    
    print("\n" + "=" * 80)
    print("  🏥 SURGICAL ANALYSIS PLATFORM - DESKTOP APPLICATION")
    print("=" * 80)
    print("\n  ✅ Database: data/surgical_analysis.db")
    print("  ✅ Application started successfully")
    print("\n  Features:")
    print("    - Async video processing (non-blocking)")
    print("    - PDF reports with formatting")
    print("    - Timeline visualizations")
    print("    - Large, resizable case detail dialogs")
    print("\n" + "=" * 80 + "\n")
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

