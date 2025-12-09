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
from datetime import datetime
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
                self.content_layout.addWidget(resources_label)
                
                resources_frame = QFrame()
                resources_frame.setStyleSheet("""
                    QFrame {
                        background-color: #f8f9fa;
                        border-radius: 8px;
                        padding: 15px;
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
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2); padding: 20px;")
        header_layout = QVBoxLayout(header)
        
        title = QLabel("🏥 Surgical Analysis Platform")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("AI-Powered Surgical Video Analysis & Reporting")
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 14px;")
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
                background: white;
            }
            QTabBar::tab {
                padding: 12px 24px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 3px solid #667eea;
            }
            QTabBar::tab:!selected {
                background: #f5f5f5;
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
        """)
        content_layout.addWidget(self.processing_widget, stretch=3)
        
        main_layout.addWidget(content_widget)
        
        # Create tabs
        self.create_cases_tab()
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
        """)
        form_frame.setMaximumWidth(600)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(15)
        
        self.case_id_input = QLineEdit()
        self.case_id_input.setPlaceholderText("e.g., CASE12345")
        self.case_id_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Case ID:", self.case_id_input)
        
        self.surgeon_combo = QComboBox()
        self.surgeon_combo.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Surgeon:", self.surgeon_combo)
        
        self.procedure_input = QLineEdit()
        self.procedure_input.setText("Labral Repair")
        self.procedure_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Procedure Type:", self.procedure_input)
        
        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_path_input = QLineEdit()
        self.video_path_input.setReadOnly(True)
        self.video_path_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        video_layout.addWidget(self.video_path_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        browse_btn.setStyleSheet("padding: 8px 16px; border-radius: 4px;")
        video_layout.addWidget(browse_btn)
        form_layout.addRow("Video File (Optional):", video_widget)
        
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)
        self.notes_input.setPlaceholderText("Additional notes...")
        self.notes_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Notes:", self.notes_input)
        
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
        """)
        form_frame.setMaximumWidth(600)
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(15)
        
        self.surgeon_id_input = QLineEdit()
        self.surgeon_id_input.setPlaceholderText("e.g., S004")
        self.surgeon_id_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Surgeon ID:", self.surgeon_id_input)
        
        self.first_name_input = QLineEdit()
        self.first_name_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("First Name:", self.first_name_input)
        
        self.last_name_input = QLineEdit()
        self.last_name_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Last Name:", self.last_name_input)
        
        self.department_input = QLineEdit()
        self.department_input.setText("Orthopedics")
        self.department_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Department:", self.department_input)
        
        self.specialty_input = QLineEdit()
        self.specialty_input.setText("Sports Medicine")
        self.specialty_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Specialty:", self.specialty_input)
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("surgeon@hospital.com")
        self.email_input.setStyleSheet("padding: 8px; border: 1px solid #ddd; border-radius: 4px;")
        form_layout.addRow("Email:", self.email_input)
        
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
                    view_btn = QPushButton("View Details")
                    view_btn.setStyleSheet("""
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
                    view_btn.clicked.connect(lambda checked, cid=case.case_id: self.view_case_details(cid))
                    btn_layout.addWidget(view_btn)
                else:
                    pending_label = QLabel("Processing...")
                    pending_label.setStyleSheet("color: #999;")
                    btn_layout.addWidget(pending_label)
                
                self.cases_table.setCellWidget(row, 5, btn_widget)
            
            self.cases_table.resizeColumnsToContents()
            
        finally:
            session.close()
    
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
        
        # Create case
        session = self.db.get_session()
        try:
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



