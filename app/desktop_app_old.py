"""
Desktop PyQt6 Application for Surgical Analysis Platform
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                             QTableWidgetItem, QTabWidget, QTextEdit, QLineEdit,
                             QComboBox, QFormLayout, QFileDialog, QMessageBox,
                             QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from app.models.database import DatabaseManager, Case, Surgeon
from app.services.mock_inference import MockInferenceEngine
from app.services.report_generator import ReportGenerator
from sqlalchemy.orm import joinedload
from datetime import datetime


class ProcessingThread(QThread):
    """Thread for processing video"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    
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
                self.finished.emit(True, f"Case {self.case_id} processed successfully!")
            else:
                self.finished.emit(False, "Processing failed")
                
        except Exception as e:
            self.finished.emit(False, str(e))


class SurgicalAnalysisApp(QMainWindow):
    """Main Desktop Application"""
    
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager('data/surgical_analysis.db')
        self.report_gen = ReportGenerator(self.db)
        self.processing_thread = None
        
        self.setWindowTitle("Surgical Analysis Platform")
        self.setGeometry(100, 100, 1200, 800)
        
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
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("🏥 Surgical Analysis Platform")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setStyleSheet("color: #667eea; padding: 15px;")
        layout.addWidget(header)
        
        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_cases_tab()
        self.create_surgeons_tab()
        self.create_upload_tab()
        self.create_add_surgeon_tab()
    
    def create_cases_tab(self):
        """Cases list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header with refresh button
        header_layout = QHBoxLayout()
        header = QLabel("📋 Cases")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(header)
        header_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_cases)
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)
        
        # Table
        self.cases_table = QTableWidget()
        self.cases_table.setColumnCount(6)
        self.cases_table.setHorizontalHeaderLabels([
            "Case ID", "Surgeon", "Date", "Duration (min)", "Status", "Actions"
        ])
        self.cases_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.cases_table)
        
        self.tabs.addTab(tab, "📋 Cases")
    
    def create_surgeons_tab(self):
        """Surgeons list tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
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
        layout.addWidget(self.surgeons_table)
        
        self.tabs.addTab(tab, "👨‍⚕️ Surgeons")
    
    def create_upload_tab(self):
        """Upload new case tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("➕ Upload New Case")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Form
        form = QFormLayout()
        
        self.case_id_input = QLineEdit()
        self.case_id_input.setPlaceholderText("e.g., CASE12345")
        form.addRow("Case ID:", self.case_id_input)
        
        self.surgeon_combo = QComboBox()
        form.addRow("Surgeon:", self.surgeon_combo)
        
        self.procedure_input = QLineEdit()
        self.procedure_input.setText("Labral Repair")
        form.addRow("Procedure Type:", self.procedure_input)
        
        self.video_path_input = QLineEdit()
        self.video_path_input.setReadOnly(True)
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_path_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(browse_btn)
        form.addRow("Video File (Optional):", video_layout)
        
        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(100)
        self.notes_input.setPlaceholderText("Additional notes...")
        form.addRow("Notes:", self.notes_input)
        
        layout.addLayout(form)
        
        # Upload button
        upload_btn = QPushButton("🚀 Upload and Process Case")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
        """)
        upload_btn.clicked.connect(self.upload_case)
        layout.addWidget(upload_btn)
        
        layout.addStretch()
        
        self.tabs.addTab(tab, "➕ Upload Case")
    
    def create_add_surgeon_tab(self):
        """Add surgeon tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("➕ Add New Surgeon")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Form
        form = QFormLayout()
        
        self.surgeon_id_input = QLineEdit()
        self.surgeon_id_input.setPlaceholderText("e.g., S004")
        form.addRow("Surgeon ID:", self.surgeon_id_input)
        
        self.first_name_input = QLineEdit()
        form.addRow("First Name:", self.first_name_input)
        
        self.last_name_input = QLineEdit()
        form.addRow("Last Name:", self.last_name_input)
        
        self.department_input = QLineEdit()
        self.department_input.setText("Orthopedics")
        form.addRow("Department:", self.department_input)
        
        self.specialty_input = QLineEdit()
        self.specialty_input.setText("Sports Medicine")
        form.addRow("Specialty:", self.specialty_input)
        
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("surgeon@hospital.com")
        form.addRow("Email:", self.email_input)
        
        layout.addLayout(form)
        
        # Add button
        add_btn = QPushButton("➕ Add Surgeon")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #667eea;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
        """)
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
                    f"{case.actual_duration_min:.1f}" if case.actual_duration_min else "N/A"
                ))
                self.cases_table.setItem(row, 4, QTableWidgetItem(case.processing_status.upper()))
                
                # Action buttons
                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.setContentsMargins(5, 0, 5, 0)
                
                view_btn = QPushButton("View Report")
                view_btn.clicked.connect(lambda checked, cid=case.case_id: self.view_case_report(cid))
                btn_layout.addWidget(view_btn)
                
                self.cases_table.setCellWidget(row, 5, btn_widget)
            
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
                btn_layout.setContentsMargins(5, 0, 5, 0)
                
                report_btn = QPushButton("View Report")
                report_btn.clicked.connect(lambda checked, sid=surgeon.surgeon_id: self.view_surgeon_report(sid))
                btn_layout.addWidget(report_btn)
                
                self.surgeons_table.setCellWidget(row, 5, btn_widget)
                
                # Add to combo box
                self.surgeon_combo.addItem(surgeon.full_name, surgeon.surgeon_id)
            
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
    
    def upload_case(self):
        """Upload and process new case"""
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
        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "Error", f"Failed to create case: {str(e)}")
            return
        finally:
            session.close()
        
        # Start processing in thread
        self.processing_thread = ProcessingThread(self.db, case_id, video_path)
        
        # Progress dialog
        progress = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        def update_progress(percent, message):
            progress.setValue(percent)
            progress.setLabelText(f"{message}\n{percent}%")
        
        def processing_done(success, message):
            progress.close()
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_cases()
                # Clear form
                self.case_id_input.clear()
                self.video_path_input.clear()
                self.notes_input.clear()
            else:
                QMessageBox.critical(self, "Error", f"Processing failed: {message}")
        
        self.processing_thread.progress.connect(update_progress)
        self.processing_thread.finished.connect(processing_done)
        self.processing_thread.start()
    
    def add_surgeon(self):
        """Add new surgeon"""
        surgeon_id = self.surgeon_id_input.text().strip()
        first_name = self.first_name_input.text().strip()
        last_name = self.last_name_input.text().strip()
        department = self.department_input.text().strip()
        specialty = self.specialty_input.text().strip()
        email = self.email_input.text().strip()
        
        if not all([surgeon_id, first_name, last_name]):
            QMessageBox.warning(self, "Error", "Please fill in all required fields")
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
            
            QMessageBox.information(self, "Success", f"Surgeon {surgeon.full_name} added successfully!")
            
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
    
    def view_case_report(self, case_id):
        """View case report"""
        report_text = self.report_gen.generate_case_report_text(case_id)
        
        # Show in dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"Report - {case_id}")
        dialog.setText(f"Case Report for {case_id}")
        dialog.setDetailedText(report_text)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Add save button
        save_btn = dialog.addButton("Save to File", QMessageBox.ButtonRole.ActionRole)
        save_btn.clicked.connect(lambda: self.save_report(case_id, report_text, 'case'))
        
        dialog.exec()
    
    def view_surgeon_report(self, surgeon_id):
        """View surgeon report"""
        report_text = self.report_gen.generate_surgeon_summary_text(surgeon_id)
        
        # Show in dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"Surgeon Report - {surgeon_id}")
        dialog.setText(f"Performance Report for {surgeon_id}")
        dialog.setDetailedText(report_text)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Add save button
        save_btn = dialog.addButton("Save to File", QMessageBox.ButtonRole.ActionRole)
        save_btn.clicked.connect(lambda: self.save_report(surgeon_id, report_text, 'surgeon'))
        
        dialog.exec()
    
    def save_report(self, id_str, report_text, report_type):
        """Save report to file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            f"{report_type}_{id_str}_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    f.write(report_text)
                QMessageBox.information(self, "Success", f"Report saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = SurgicalAnalysisApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

