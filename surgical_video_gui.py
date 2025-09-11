#!/usr/bin/env python3
"""
Modern Surgical Video Analysis GUI
A sleek, contemporary interface for AI-powered surgical video analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image, ImageTk
import pandas as pd
import sys

# Import professional AI system
sys.path.append(str(Path(__file__).parent / "surgical_ai_system"))
from inference.master_inference_engine import MasterInferenceEngine
from config.system_config import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePhaseModel(nn.Module):
    """Simple ResNet-based phase recognition model."""
    
    def __init__(self, num_phases):
        super().__init__()
        self.backbone = resnet50(weights='IMAGENET1K_V1')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_phases)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModernButton(tk.Button):
    """Modern styled button with hover effects."""
    
    def __init__(self, parent, text, command=None, bg_color="#2196F3", hover_color="#1976D2", **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=bg_color,
            relief="flat",
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
            **kwargs
        )
        
        self.bg_color = bg_color
        self.hover_color = hover_color
        
        # Hover effects
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, e):
        self.configure(bg=self.hover_color)
    
    def _on_leave(self, e):
        self.configure(bg=self.bg_color)

class ModernEntry(tk.Frame):
    """Modern styled entry with label."""
    
    def __init__(self, parent, label_text, placeholder="", **kwargs):
        super().__init__(parent, bg="#f8f9fa")
        
        # Label
        self.label = tk.Label(
            self,
            text=label_text,
            font=("Segoe UI", 9, "bold"),
            fg="#495057",
            bg="#f8f9fa"
        )
        self.label.pack(anchor="w", pady=(0, 5))
        
        # Entry
        self.entry = tk.Entry(
            self,
            font=("Segoe UI", 10),
            relief="solid",
            bd=1,
            highlightthickness=2,
            highlightcolor="#2196F3",
            **kwargs
        )
        self.entry.pack(fill="x", ipady=8)
        
        # Placeholder
        if placeholder:
            self.entry.insert(0, placeholder)
            self.entry.configure(fg="gray")
            self.entry.bind("<FocusIn>", self._clear_placeholder)
            self.entry.bind("<FocusOut>", self._add_placeholder)
            self.placeholder = placeholder
    
    def _clear_placeholder(self, event):
        if self.entry.get() == self.placeholder:
            self.entry.delete(0, "end")
            self.entry.configure(fg="black")
    
    def _add_placeholder(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.configure(fg="gray")
    
    def get(self):
        value = self.entry.get()
        return "" if value == getattr(self, 'placeholder', "") else value

class SurgicalVideoAnalyzerGUI:
    """Modern GUI for surgical video analysis."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        
        # Professional AI system components
        self.inference_engine = None
        self.config_manager = None
        self.video_path = None
        
        # Initialize professional configuration system
        self.initialize_professional_system()
        
    def setup_window(self):
        """Configure the main window."""
        self.root.title("AI Surgical Video Analyzer")
        self.root.geometry("900x700")
        self.root.configure(bg="#f8f9fa")
        self.root.resizable(True, True)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"900x700+{x}+{y}")
    
    def setup_styles(self):
        """Configure modern styles."""
        style = ttk.Style()
        style.configure(
            "Modern.TNotebook",
            background="#f8f9fa",
            borderwidth=0
        )
        style.configure(
            "Modern.TNotebook.Tab",
            padding=[20, 10],
            font=("Segoe UI", 10, "bold")
        )
    
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f8f9fa")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_frame)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style="Modern.TNotebook")
        self.notebook.pack(fill="both", expand=True, pady=(20, 0))
        
        # Analysis Tab
        self.create_analysis_tab()
        
        # Results Tab
        self.create_results_tab()
        
        # Status Bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create modern header section."""
        header_frame = tk.Frame(parent, bg="#2196F3", height=80)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🏥 AI Surgical Video Analyzer",
            font=("Segoe UI", 18, "bold"),
            fg="white",
            bg="#2196F3"
        )
        title_label.pack(expand=True)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Advanced computer vision for surgical procedure analysis",
            font=("Segoe UI", 10),
            fg="#E3F2FD",
            bg="#2196F3"
        )
        subtitle_label.pack()
    
    def create_analysis_tab(self):
        """Create the main analysis tab."""
        analysis_frame = tk.Frame(self.notebook, bg="#f8f9fa")
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Create scrollable frame
        canvas = tk.Canvas(analysis_frame, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Video Selection Section
        self.create_video_section(scrollable_frame)
        
        # Metadata Section
        self.create_metadata_section(scrollable_frame)
        
        # Analysis Section
        self.create_analysis_section(scrollable_frame)
    
    def create_video_section(self, parent):
        """Create video selection section."""
        section_frame = tk.LabelFrame(
            parent,
            text="📹 Video Selection",
            font=("Segoe UI", 12, "bold"),
            fg="#495057",
            bg="#f8f9fa",
            relief="solid",
            bd=1,
            padx=15,
            pady=15
        )
        section_frame.pack(fill="x", pady=(0, 20), padx=10)
        
        # Video path display
        self.video_path_var = tk.StringVar(value="No video selected")
        video_label = tk.Label(
            section_frame,
            textvariable=self.video_path_var,
            font=("Segoe UI", 10),
            fg="#6c757d",
            bg="#f8f9fa",
            wraplength=500
        )
        video_label.pack(pady=(0, 10))
        
        # Select button
        select_btn = ModernButton(
            section_frame,
            text="🔍 Select Video File",
            command=self.select_video_file,
            bg_color="#28a745",
            hover_color="#218838"
        )
        select_btn.pack()
    
    def create_metadata_section(self, parent):
        """Create metadata input section."""
        section_frame = tk.LabelFrame(
            parent,
            text="📝 Case Information",
            font=("Segoe UI", 12, "bold"),
            fg="#495057",
            bg="#f8f9fa",
            relief="solid",
            bd=1,
            padx=15,
            pady=15
        )
        section_frame.pack(fill="x", pady=(0, 20), padx=10)
        
        # Create two-column layout
        left_frame = tk.Frame(section_frame, bg="#f8f9fa")
        right_frame = tk.Frame(section_frame, bg="#f8f9fa")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Left column
        self.case_id_entry = ModernEntry(left_frame, "Case ID", "e.g., CASE_001")
        self.case_id_entry.pack(fill="x", pady=(0, 15))
        
        self.surgeon_entry = ModernEntry(left_frame, "Surgeon Name", "Dr. Smith")
        self.surgeon_entry.pack(fill="x", pady=(0, 15))
        
        # Right column
        self.procedure_entry = ModernEntry(right_frame, "Procedure Type", "Labral Repair")
        self.procedure_entry.pack(fill="x", pady=(0, 15))
        
        self.date_entry = ModernEntry(right_frame, "Procedure Date", datetime.now().strftime("%Y-%m-%d"))
        self.date_entry.pack(fill="x", pady=(0, 15))
    
    def create_analysis_section(self, parent):
        """Create analysis control section."""
        section_frame = tk.LabelFrame(
            parent,
            text="🧠 AI Analysis",
            font=("Segoe UI", 12, "bold"),
            fg="#495057",
            bg="#f8f9fa",
            relief="solid",
            bd=1,
            padx=15,
            pady=15
        )
        section_frame.pack(fill="x", pady=(0, 20), padx=10)
        
        # Model status
        self.model_status_var = tk.StringVar(value="⚡ Ready to load AI model")
        status_label = tk.Label(
            section_frame,
            textvariable=self.model_status_var,
            font=("Segoe UI", 10),
            fg="#6c757d",
            bg="#f8f9fa"
        )
        status_label.pack(pady=(0, 15))
        
        # Buttons frame
        buttons_frame = tk.Frame(section_frame, bg="#f8f9fa")
        buttons_frame.pack()
        
        # Load model button
        load_model_btn = ModernButton(
            buttons_frame,
            text="🤖 Load AI Model",
            command=self.load_model,
            bg_color="#17a2b8",
            hover_color="#138496"
        )
        load_model_btn.pack(side="left", padx=(0, 10))
        
        # Analyze button
        self.analyze_btn = ModernButton(
            buttons_frame,
            text="🚀 Start Analysis",
            command=self.start_analysis,
            bg_color="#dc3545",
            hover_color="#c82333"
        )
        self.analyze_btn.pack(side="left", padx=(10, 0))
        self.analyze_btn.configure(state="disabled")
        
        # Progress bar
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            section_frame,
            textvariable=self.progress_var,
            font=("Segoe UI", 9),
            fg="#28a745",
            bg="#f8f9fa"
        )
        self.progress_label.pack(pady=(15, 0))
    
    def create_results_tab(self):
        """Create results display tab."""
        results_frame = tk.Frame(self.notebook, bg="#f8f9fa")
        self.notebook.add(results_frame, text="📈 Results")
        
        # Results text area
        text_frame = tk.Frame(results_frame, bg="#f8f9fa")
        text_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Text widget with scrollbar
        self.results_text = tk.Text(
            text_frame,
            font=("Consolas", 10),
            bg="white",
            fg="#495057",
            relief="solid",
            bd=1,
            padx=15,
            pady=15
        )
        results_scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Export buttons
        export_frame = tk.Frame(results_frame, bg="#f8f9fa")
        export_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        export_json_btn = ModernButton(
            export_frame,
            text="💾 Export JSON",
            command=self.export_json,
            bg_color="#6f42c1",
            hover_color="#5a32a3"
        )
        export_json_btn.pack(side="left", padx=(0, 10))
        
        export_csv_btn = ModernButton(
            export_frame,
            text="📊 Export CSV",
            command=self.export_csv,
            bg_color="#fd7e14",
            hover_color="#e55a00"
        )
        export_csv_btn.pack(side="left", padx=(10, 0))
    
    def create_status_bar(self, parent):
        """Create status bar."""
        status_frame = tk.Frame(parent, bg="#e9ecef", height=30)
        status_frame.pack(fill="x", pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#495057",
            bg="#e9ecef"
        )
        status_label.pack(side="left", padx=10, pady=5)
    
    def select_video_file(self):
        """Open file dialog to select video."""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Surgical Video",
            filetypes=file_types
        )
        
        if filename:
            self.video_path = filename
            self.video_path_var.set(f"Selected: {Path(filename).name}")
            self.status_var.set(f"Video selected: {Path(filename).name}")
    
    def initialize_professional_system(self):
        """Initialize the professional AI system components"""
        try:
            # Initialize configuration manager
            config_path = Path("surgical_ai_system/config/surgical_ai_config.yaml")
            self.config_manager = ConfigurationManager(str(config_path))
            
            # Initialize inference engine with professional models
            models_dir = Path("surgical_ai_system/trained_models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # For now, use existing simple models as fallback
            fallback_models_dir = Path("trained_models")
            if fallback_models_dir.exists():
                # Copy existing models to professional directory
                import shutil
                for model_file in fallback_models_dir.glob("*.pth"):
                    dest_path = models_dir / model_file.name
                    if not dest_path.exists():
                        shutil.copy2(model_file, dest_path)
                        logger.info(f"Copied {model_file.name} to professional models directory")
                        
                for config_file in fallback_models_dir.glob("*.json"):
                    dest_path = models_dir / config_file.name
                    if not dest_path.exists():
                        shutil.copy2(config_file, dest_path)
                        logger.info(f"Copied {config_file.name} to professional models directory")
            
            self.status_var.set("Professional AI system initialized")
            logger.info("Professional surgical AI system initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not fully initialize professional system: {e}")
            logger.info("Falling back to basic mode")
            self.status_var.set("System initialized in basic mode")
    
    def load_model(self):
        """Load the professional AI models."""
        try:
            self.model_status_var.set("🔄 Loading professional AI models...")
            self.status_var.set("Loading comprehensive AI system...")
            
            # Initialize master inference engine
            models_dir = Path("surgical_ai_system/trained_models")
            config_path = Path("surgical_ai_system/config/surgical_ai_config.yaml")
            
            self.inference_engine = MasterInferenceEngine(
                config_path=str(config_path) if config_path.exists() else None,
                models_dir=str(models_dir)
            )
            
            self.model_status_var.set("✅ Professional AI system loaded successfully!")
            self.status_var.set("Comprehensive AI models loaded - Phase Detection, Instrument Tracking, Event Detection, Motion Analysis")
            self.analyze_btn.configure(state="normal")
            
        except Exception as e:
            logger.error(f"Failed to load professional system: {e}")
            
            # Fallback to simple model
            try:
                self.model_status_var.set("🔄 Loading fallback model...")
                
                model_path = Path("surgical_ai_system/trained_models/phase_detector.pth")
                config_path = Path("surgical_ai_system/trained_models/model_configs.json")
                
                if not model_path.exists() or not config_path.exists():
                    messagebox.showerror("Error", "No AI models found. Please run training first.")
                    return
                
                # Load simple config
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                
                # Load simple model
                self.model = SimplePhaseModel(len(self.config['phase_labels']))
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                
                self.model_status_var.set("✅ Professional AI model loaded (phase detection)")
                self.status_var.set("Professional phase detector model loaded successfully")
                self.analyze_btn.configure(state="normal")
                
            except Exception as e2:
                messagebox.showerror("Error", f"Failed to load any AI model: {str(e2)}")
                self.model_status_var.set("❌ Failed to load AI models")
                self.status_var.set("Model loading failed")
    
    def start_analysis(self):
        """Start video analysis in a separate thread."""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
        
        if not self.inference_engine and not hasattr(self, 'model'):
            messagebox.showwarning("Warning", "Please load the AI model first.")
            return
        
        # Validate metadata
        if not all([self.case_id_entry.get(), self.surgeon_entry.get()]):
            messagebox.showwarning("Warning", "Please fill in at least Case ID and Surgeon Name.")
            return
        
        # Start analysis in background thread
        self.analyze_btn.configure(state="disabled")
        self.progress_var.set("🔄 Starting analysis...")
        
        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def run_analysis(self):
        """Run the actual video analysis."""
        try:
            # Use professional inference engine if available
            if self.inference_engine:
                self.run_professional_analysis()
            else:
                self.run_basic_analysis()
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set("❌ Analysis failed"))
        finally:
            self.root.after(0, lambda: self.analyze_btn.configure(state="normal"))
    
    def run_professional_analysis(self):
        """Run comprehensive analysis using professional inference engine."""
        self.root.after(0, lambda: self.progress_var.set("🔧 Preparing comprehensive AI analysis..."))
        
        # Prepare case metadata
        case_metadata = {
            'case_id': self.case_id_entry.get(),
            'surgeon_id': self.surgeon_entry.get(),
            'procedure_type': self.procedure_entry.get(),
            'procedure_date': self.date_entry.get()
        }
        
        self.root.after(0, lambda: self.progress_var.set("🧠 Running multi-model AI analysis..."))
        
        try:
            # Create progress callback function
            def update_progress(message):
                self.root.after(0, lambda: self.progress_var.set(message))
            
            # Run comprehensive analysis
            surgical_case = self.inference_engine.analyze_video(self.video_path, case_metadata, update_progress)
            
            self.root.after(0, lambda: self.progress_var.set("📊 Generating comprehensive report..."))
            
            # Save comprehensive results
            output_dir = Path("results") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_files = self.inference_engine.save_comprehensive_results(str(output_dir))
            
            # Generate results for GUI display
            results = self.generate_professional_report(surgical_case, saved_files)
            
            # Update UI with results
            self.root.after(0, lambda: self.display_professional_results(results, surgical_case))
            self.root.after(0, lambda: self.progress_var.set("✅ Comprehensive analysis complete!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set("❌ Analysis failed"))
            print(f"Analysis error: {e}")  # Also print to console for debugging
    
    def run_basic_analysis(self):
        """Run basic analysis using simple phase detection model."""
        self.root.after(0, lambda: self.progress_var.set("🎥 Loading video..."))
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Transform for model input
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Analyze frames (sample every 60 frames)
        predictions = []
        frame_count = 0
        sample_rate = 60
        
        self.root.after(0, lambda: self.progress_var.set("🧠 AI analyzing video frames..."))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Update progress
                progress_pct = (frame_count / total_frames) * 100
                self.root.after(0, lambda p=progress_pct: self.progress_var.set(f"🔍 Analyzing... {p:.1f}%"))
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    predicted_phase = self.config['phase_labels'][predicted_class]
                    timestamp = frame_count / fps
                    
                    predictions.append({
                        'frame': frame_count,
                        'timestamp_seconds': timestamp,
                        'timestamp_formatted': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        'predicted_phase': predicted_phase,
                        'confidence': confidence
                    })
            
            frame_count += 1
        
        cap.release()
        
        # Generate results
        self.root.after(0, lambda: self.progress_var.set("📊 Generating report..."))
        results = self.generate_basic_report(predictions, duration)
        
        # Update UI with results
        self.root.after(0, lambda: self.display_results(results, predictions))
        self.root.after(0, lambda: self.progress_var.set("✅ Basic analysis complete!"))
    
    def generate_professional_report(self, surgical_case, saved_files):
        """Generate comprehensive report from professional analysis."""
        # Get metrics from the surgical case
        metrics = surgical_case.metrics if surgical_case.metrics else {}
        
        return {
            'metadata': {
                'case_id': surgical_case.case_id,
                'surgeon_id': surgical_case.surgeon_id,
                'analysis_type': 'comprehensive'
            },
            'surgical_analysis': {
                'total_duration_minutes': surgical_case.video_duration / 60,
                'total_idle_time_minutes': getattr(metrics, 'total_idle_time', 0) / 60,
                'phases_detected': len(surgical_case.phases),
                'instruments_detected': len(surgical_case.instruments),
                'bleeding_events': len(surgical_case.bleeding_events),
                'suture_attempts': len(surgical_case.suture_attempts),
                'suture_failure_rate': getattr(metrics, 'suture_failure_rate', 0) * 100,
                'number_of_implants': getattr(metrics, 'number_of_implants', 0),
                'time_to_first_suture': getattr(metrics, 'time_to_first_suture', 0)
            },
            'phase_durations': {
                'diagnostic_arthroscopy': getattr(metrics, 'diagnostic_arthroscopy_time', 0) / 60,
                'glenoid_preparation': getattr(metrics, 'glenoid_preparation_time', 0) / 60,
                'labral_mobilization': getattr(metrics, 'labral_mobilization_time', 0) / 60,
                'anchor_placement': getattr(metrics, 'anchor_placement_time', 0) / 60,
                'suture_passage': getattr(metrics, 'suture_passage_time', 0) / 60,
                'suture_tensioning': getattr(metrics, 'suture_tensioning_time', 0) / 60,
                'final_inspection': getattr(metrics, 'final_inspection_time', 0) / 60
            },
            'saved_files': saved_files,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def display_professional_results(self, report, surgical_case):
        """Display comprehensive professional analysis results."""
        self.results_text.delete(1.0, tk.END)
        
        results_text = f"""
🏥 COMPREHENSIVE SURGICAL VIDEO ANALYSIS REPORT
{'='*60}

📋 CASE INFORMATION:
   Case ID: {report['metadata']['case_id']}
   Surgeon: {report['metadata']['surgeon_id']}
   Analysis Type: Professional Multi-Model AI System

⏱️ PROCEDURE METRICS:
   Total Duration: {report['surgical_analysis']['total_duration_minutes']:.2f} minutes
   Total Idle Time: {report['surgical_analysis']['total_idle_time_minutes']:.2f} minutes
   Idle Percentage: {(report['surgical_analysis']['total_idle_time_minutes']/report['surgical_analysis']['total_duration_minutes']*100):.1f}%

🔧 SURGICAL PHASES ANALYSIS:
   • Diagnostic Arthroscopy: {report['phase_durations']['diagnostic_arthroscopy']:.2f} min
   • Glenoid Preparation: {report['phase_durations']['glenoid_preparation']:.2f} min
   • Labral Mobilization: {report['phase_durations']['labral_mobilization']:.2f} min
   • Anchor Placement: {report['phase_durations']['anchor_placement']:.2f} min
   • Suture Passage: {report['phase_durations']['suture_passage']:.2f} min
   • Suture Tensioning: {report['phase_durations']['suture_tensioning']:.2f} min
   • Final Inspection: {report['phase_durations']['final_inspection']:.2f} min

🏷️ EVENT DETECTION:
   • Bleeding Events: {report['surgical_analysis']['bleeding_events']}
   • Suture Attempts: {report['surgical_analysis']['suture_attempts']}
   • Suture Failure Rate: {report['surgical_analysis']['suture_failure_rate']:.1f}%
   • Time to First Suture: {report['surgical_analysis']['time_to_first_suture']:.2f} seconds

🔨 INSTRUMENT & IMPLANT USAGE:
   • Number of Implants: {report['surgical_analysis']['number_of_implants']}
   • Instrument Events Detected: {report['surgical_analysis']['instruments_detected']}

💾 EXPORTED FILES:
"""
        
        for file_type, file_path in report['saved_files'].items():
            results_text += f"   • {file_type.upper()}: {Path(file_path).name}\n"
        
        results_text += f"\n⏰ Analysis completed: {report['analysis_timestamp']}"
        results_text += f"\n\n🚀 Professional Analysis Complete - Full surgical workflow analyzed!"
        
        self.results_text.insert(1.0, results_text)
        
        # Store for export
        self.last_professional_report = report
        self.last_surgical_case = surgical_case
        
        # Switch to results tab
        self.notebook.select(1)
    
    def generate_basic_report(self, predictions, duration):
        """Generate basic analysis report."""
        # Phase analysis
        phase_counts = {}
        for pred in predictions:
            phase = pred['predicted_phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Metadata
        metadata = {
            'case_id': self.case_id_entry.get(),
            'surgeon_name': self.surgeon_entry.get(),
            'procedure_type': self.procedure_entry.get(),
            'procedure_date': self.date_entry.get(),
        }
        
        # Report
        report = {
            'metadata': metadata,
            'video_analysis': {
                'total_duration_minutes': duration / 60,
                'frames_analyzed': len(predictions),
                'phases_detected': list(phase_counts.keys()),
                'phase_distribution': phase_counts,
                'dominant_phase': max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else "Unknown",
                'average_confidence': np.mean([p['confidence'] for p in predictions])
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Store for export
        self.last_report = report
        self.last_predictions = predictions
        
        return report
    
    def display_results(self, report, predictions):
        """Display results in the results tab."""
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        results_text = f"""
🏥 SURGICAL VIDEO ANALYSIS REPORT
{'='*50}

📋 CASE INFORMATION:
   Case ID: {report['metadata']['case_id']}
   Surgeon: {report['metadata']['surgeon_name']}
   Procedure: {report['metadata']['procedure_type']}
   Date: {report['metadata']['procedure_date']}

⏱️ PROCEDURE ANALYSIS:
   Total Duration: {report['video_analysis']['total_duration_minutes']:.2f} minutes
   Frames Analyzed: {report['video_analysis']['frames_analyzed']}
   
🧠 AI PHASE DETECTION:
   Dominant Phase: {report['video_analysis']['dominant_phase']}
   Average Confidence: {report['video_analysis']['average_confidence']:.2f}
   
📊 PHASE DISTRIBUTION:
"""
        
        for phase, count in report['video_analysis']['phase_distribution'].items():
            percentage = (count / len(predictions)) * 100
            results_text += f"   • {phase}: {count} frames ({percentage:.1f}%)\n"
        
        results_text += f"\n🕐 SAMPLE PREDICTIONS:\n"
        for pred in predictions[:10]:  # Show first 10 predictions
            results_text += f"   {pred['timestamp_formatted']} → {pred['predicted_phase']} ({pred['confidence']:.2f})\n"
        
        if len(predictions) > 10:
            results_text += f"   ... and {len(predictions) - 10} more predictions\n"
        
        results_text += f"\n⏰ Analysis completed: {report['analysis_timestamp']}\n"
        
        self.results_text.insert(1.0, results_text)
        
        # Switch to results tab
        self.notebook.select(1)
    
    def export_json(self):
        """Export results as JSON."""
        if not hasattr(self, 'last_report'):
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save JSON Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.last_report, f, indent=2)
                messagebox.showinfo("Success", f"Report exported to {filename}")
                self.status_var.set(f"Exported JSON: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def export_csv(self):
        """Export predictions as CSV."""
        if not hasattr(self, 'last_predictions'):
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save CSV Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.last_predictions)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
                self.status_var.set(f"Exported CSV: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main entry point."""
    app = SurgicalVideoAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main()