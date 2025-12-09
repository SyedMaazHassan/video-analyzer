#!/usr/bin/env python3
"""
Modern Surgical Video Analysis GUI
A professional, contemporary interface for AI-powered surgical video analysis.
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
import os
import shutil

# Import professional AI system
sys.path.append(str(Path(__file__).parent / "surgical_ai_system"))
from inference.master_inference_engine import MasterInferenceEngine
from config.system_config import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modern color scheme (matching web UI)
COLORS = {
    'bg_dark': '#0f172a',
    'bg_medium': '#1e293b',
    'bg_light': '#f8fafc',
    'bg_card': '#ffffff',
    'primary': '#3b82f6',
    'primary_dark': '#2563eb',
    'secondary': '#64748b',
    'success': '#22c55e',
    'success_dark': '#16a34a',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'danger_dark': '#dc2626',
    'text_primary': '#0f172a',
    'text_secondary': '#64748b',
    'text_muted': '#94a3b8',
    'border': '#e2e8f0',
    'purple': '#a855f7',
    'purple_dark': '#9333ea',
}


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


class ModernButton(tk.Canvas):
    """Modern styled button with hover effects and icons."""

    def __init__(self, parent, text, command=None, icon=None,
                 bg_color=None, hover_color=None, width=180, height=40, **kwargs):
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bg=parent.cget('bg'), **kwargs)

        self.bg_color = bg_color or COLORS['primary']
        self.hover_color = hover_color or COLORS['primary_dark']
        self.command = command
        self.text = text
        self.icon = icon
        self.width = width
        self.height = height

        self.draw_button(self.bg_color)

        # Bind events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def draw_button(self, color):
        self.delete("all")
        # Draw rounded rectangle
        radius = 10
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, radius, fill=color, outline="")
        # Draw text with icon
        display_text = f"{self.icon} {self.text}" if self.icon else self.text
        self.create_text(self.width//2, self.height//2, text=display_text,
                        fill="white", font=("Inter", 11, "bold"))

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r,
            x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y2-r, x1, y1+r, x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, e):
        self.draw_button(self.hover_color)
        self.config(cursor="hand2")

    def _on_leave(self, e):
        self.draw_button(self.bg_color)

    def _on_click(self, e):
        if self.command:
            self.command()


class ModernEntry(tk.Frame):
    """Modern styled entry with label."""

    def __init__(self, parent, label_text, placeholder="", **kwargs):
        super().__init__(parent, bg=COLORS['bg_light'])

        # Label
        self.label = tk.Label(
            self,
            text=label_text,
            font=("Inter", 10, "bold"),
            fg=COLORS['text_primary'],
            bg=COLORS['bg_light']
        )
        self.label.pack(anchor="w", pady=(0, 6))

        # Entry frame for border
        entry_frame = tk.Frame(self, bg=COLORS['border'], bd=0)
        entry_frame.pack(fill="x")

        # Entry
        self.entry = tk.Entry(
            entry_frame,
            font=("Inter", 11),
            relief="flat",
            bd=0,
            bg=COLORS['bg_card'],
            fg=COLORS['text_primary'],
            insertbackground=COLORS['primary'],
            **kwargs
        )
        self.entry.pack(fill="x", ipady=10, ipadx=12, padx=1, pady=1)

        # Placeholder
        if placeholder:
            self.entry.insert(0, placeholder)
            self.entry.configure(fg=COLORS['text_muted'])
            self.entry.bind("<FocusIn>", self._clear_placeholder)
            self.entry.bind("<FocusOut>", self._add_placeholder)
            self.placeholder = placeholder

    def _clear_placeholder(self, event):
        if self.entry.get() == getattr(self, 'placeholder', ''):
            self.entry.delete(0, "end")
            self.entry.configure(fg=COLORS['text_primary'])

    def _add_placeholder(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.configure(fg=COLORS['text_muted'])

    def get(self):
        value = self.entry.get()
        return "" if value == getattr(self, 'placeholder', "") else value


class MetricCard(tk.Frame):
    """Modern metric card widget."""

    def __init__(self, parent, icon, label, value, color=COLORS['primary']):
        super().__init__(parent, bg=COLORS['bg_card'], bd=0, relief="flat")

        # Configure padding
        self.configure(padx=20, pady=16)

        # Icon and label row
        header = tk.Frame(self, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 8))

        # Icon badge
        icon_label = tk.Label(header, text=icon, font=("Inter", 14),
                             fg=color, bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        # Label
        label_widget = tk.Label(header, text=label.upper(),
                               font=("Inter", 9, "bold"),
                               fg=COLORS['text_secondary'], bg=COLORS['bg_card'])
        label_widget.pack(side="left", padx=(8, 0))

        # Value
        self.value_label = tk.Label(self, text=value, font=("Inter", 28, "bold"),
                                   fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        self.value_label.pack(anchor="w")

    def update_value(self, value):
        self.value_label.configure(text=value)


class SurgicalVideoAnalyzerGUI:
    """Modern GUI for surgical video analysis."""

    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()

        # Professional AI system components
        self.inference_engine = None
        self.config_manager = None
        self.video_path = None

        # Initialize professional configuration system
        self.initialize_professional_system()

    def setup_window(self):
        """Configure the main window."""
        self.root.title("Surgical Analysis Platform")
        self.root.geometry("1100x800")
        self.root.configure(bg=COLORS['bg_light'])
        self.root.resizable(True, True)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1100 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1100x800+{x}+{y}")

        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Modern.TNotebook", background=COLORS['bg_light'], borderwidth=0)
        style.configure("Modern.TNotebook.Tab", padding=[20, 12], font=("Inter", 10, "bold"))
        style.map("Modern.TNotebook.Tab",
                 background=[("selected", COLORS['bg_card']), ("!selected", COLORS['bg_light'])],
                 foreground=[("selected", COLORS['primary']), ("!selected", COLORS['text_secondary'])])

    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Header
        self.create_header()

        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS['bg_light'])
        main_frame.pack(fill="both", expand=True, padx=24, pady=24)

        # Metrics row
        self.create_metrics_row(main_frame)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style="Modern.TNotebook")
        self.notebook.pack(fill="both", expand=True, pady=(20, 0))

        # Analysis Tab
        self.create_analysis_tab()

        # Results Tab
        self.create_results_tab()

        # Downloads Tab
        self.create_downloads_tab()

        # Status Bar
        self.create_status_bar()

    def create_header(self):
        """Create modern header section."""
        header_frame = tk.Frame(self.root, bg=COLORS['bg_dark'], height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        # Content container
        content = tk.Frame(header_frame, bg=COLORS['bg_dark'])
        content.pack(fill="both", expand=True, padx=24)

        # Icon badge
        icon_frame = tk.Frame(content, bg=COLORS['primary'], width=48, height=48)
        icon_frame.pack(side="left", pady=16)
        icon_frame.pack_propagate(False)

        icon_label = tk.Label(icon_frame, text="AI", font=("Inter", 16, "bold"),
                             fg="white", bg=COLORS['primary'])
        icon_label.place(relx=0.5, rely=0.5, anchor="center")

        # Title section
        title_frame = tk.Frame(content, bg=COLORS['bg_dark'])
        title_frame.pack(side="left", padx=(16, 0), pady=16)

        title_label = tk.Label(
            title_frame,
            text="Surgical Analysis Platform",
            font=("Inter", 20, "bold"),
            fg="white",
            bg=COLORS['bg_dark']
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            title_frame,
            text="AI-Powered Surgical Video Analysis & Reporting",
            font=("Inter", 11),
            fg=COLORS['text_muted'],
            bg=COLORS['bg_dark']
        )
        subtitle_label.pack(anchor="w")

    def create_metrics_row(self, parent):
        """Create metrics cards row."""
        metrics_frame = tk.Frame(parent, bg=COLORS['bg_light'])
        metrics_frame.pack(fill="x", pady=(0, 20))

        # Configure grid
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1, uniform="metrics")

        # Metric cards
        self.cases_card = MetricCard(metrics_frame, "Folder", "Cases Analyzed", "0", COLORS['primary'])
        self.cases_card.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.duration_card = MetricCard(metrics_frame, "Clock", "Avg Duration", "- min", COLORS['success'])
        self.duration_card.grid(row=0, column=1, sticky="ew", padx=10)

        self.phases_card = MetricCard(metrics_frame, "Layers", "Phases Detected", "0", COLORS['warning'])
        self.phases_card.grid(row=0, column=2, sticky="ew", padx=10)

        self.confidence_card = MetricCard(metrics_frame, "Target", "Model Accuracy", "77.9%", COLORS['purple'])
        self.confidence_card.grid(row=0, column=3, sticky="ew", padx=(10, 0))

    def create_analysis_tab(self):
        """Create the main analysis tab."""
        analysis_frame = tk.Frame(self.notebook, bg=COLORS['bg_light'])
        self.notebook.add(analysis_frame, text="  Analysis  ")

        # Scrollable content
        canvas = tk.Canvas(analysis_frame, bg=COLORS['bg_light'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['bg_light'])

        scrollable_frame.bind("<Configure>",
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

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
        section_frame = tk.Frame(parent, bg=COLORS['bg_card'], bd=0)
        section_frame.pack(fill="x", pady=(0, 16), padx=4)

        # Section content
        content = tk.Frame(section_frame, bg=COLORS['bg_card'])
        content.pack(fill="x", padx=24, pady=20)

        # Header
        header = tk.Frame(content, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 16))

        icon_label = tk.Label(header, text="Video", font=("Inter", 12),
                             fg=COLORS['primary'], bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        title = tk.Label(header, text="Video Selection", font=("Inter", 14, "bold"),
                        fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        title.pack(side="left", padx=(8, 0))

        # Video path display
        self.video_path_var = tk.StringVar(value="No video selected")
        video_label = tk.Label(
            content,
            textvariable=self.video_path_var,
            font=("Inter", 11),
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_card'],
            wraplength=600
        )
        video_label.pack(pady=(0, 16))

        # Select button
        select_btn = ModernButton(
            content,
            text="Select Video File",
            icon="Choose",
            command=self.select_video_file,
            bg_color=COLORS['success'],
            hover_color=COLORS['success_dark'],
            width=200
        )
        select_btn.pack()

    def create_metadata_section(self, parent):
        """Create metadata input section."""
        section_frame = tk.Frame(parent, bg=COLORS['bg_card'], bd=0)
        section_frame.pack(fill="x", pady=(0, 16), padx=4)

        content = tk.Frame(section_frame, bg=COLORS['bg_card'])
        content.pack(fill="x", padx=24, pady=20)

        # Header
        header = tk.Frame(content, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 16))

        icon_label = tk.Label(header, text="Form", font=("Inter", 12),
                             fg=COLORS['primary'], bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        title = tk.Label(header, text="Case Information", font=("Inter", 14, "bold"),
                        fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        title.pack(side="left", padx=(8, 0))

        # Two-column layout
        columns = tk.Frame(content, bg=COLORS['bg_card'])
        columns.pack(fill="x")

        left_frame = tk.Frame(columns, bg=COLORS['bg_card'])
        right_frame = tk.Frame(columns, bg=COLORS['bg_card'])
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right_frame.pack(side="right", fill="both", expand=True, padx=(12, 0))

        # Form fields
        # Override bg color for entries
        for frame in [left_frame, right_frame]:
            frame.configure(bg=COLORS['bg_light'])

        self.case_id_entry = ModernEntry(left_frame, "Case ID", "e.g., CASE_001")
        self.case_id_entry.pack(fill="x", pady=(0, 12))

        self.surgeon_entry = ModernEntry(left_frame, "Surgeon Name", "Dr. Smith")
        self.surgeon_entry.pack(fill="x")

        self.procedure_entry = ModernEntry(right_frame, "Procedure Type", "Labral Repair")
        self.procedure_entry.pack(fill="x", pady=(0, 12))

        self.date_entry = ModernEntry(right_frame, "Procedure Date",
                                      datetime.now().strftime("%Y-%m-%d"))
        self.date_entry.pack(fill="x")

    def create_analysis_section(self, parent):
        """Create analysis control section."""
        section_frame = tk.Frame(parent, bg=COLORS['bg_card'], bd=0)
        section_frame.pack(fill="x", pady=(0, 16), padx=4)

        content = tk.Frame(section_frame, bg=COLORS['bg_card'])
        content.pack(fill="x", padx=24, pady=20)

        # Header
        header = tk.Frame(content, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 16))

        icon_label = tk.Label(header, text="AI", font=("Inter", 12),
                             fg=COLORS['primary'], bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        title = tk.Label(header, text="AI Analysis", font=("Inter", 14, "bold"),
                        fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        title.pack(side="left", padx=(8, 0))

        # Model status
        self.model_status_var = tk.StringVar(value="Ready to load AI model")
        status_label = tk.Label(
            content,
            textvariable=self.model_status_var,
            font=("Inter", 11),
            fg=COLORS['text_secondary'],
            bg=COLORS['bg_card']
        )
        status_label.pack(pady=(0, 16))

        # Buttons frame
        buttons_frame = tk.Frame(content, bg=COLORS['bg_card'])
        buttons_frame.pack()

        # Load model button
        load_model_btn = ModernButton(
            buttons_frame,
            text="Load AI Model",
            icon="Load",
            command=self.load_model,
            bg_color=COLORS['primary'],
            hover_color=COLORS['primary_dark'],
            width=180
        )
        load_model_btn.pack(side="left", padx=(0, 12))

        # Analyze button
        self.analyze_btn = ModernButton(
            buttons_frame,
            text="Start Analysis",
            icon="Start",
            command=self.start_analysis,
            bg_color=COLORS['danger'],
            hover_color=COLORS['danger_dark'],
            width=180
        )
        self.analyze_btn.pack(side="left")

        # Progress label
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            content,
            textvariable=self.progress_var,
            font=("Inter", 10),
            fg=COLORS['success'],
            bg=COLORS['bg_card']
        )
        self.progress_label.pack(pady=(16, 0))

    def create_results_tab(self):
        """Create results display tab."""
        results_frame = tk.Frame(self.notebook, bg=COLORS['bg_light'])
        self.notebook.add(results_frame, text="  Results  ")

        # Results section
        section_frame = tk.Frame(results_frame, bg=COLORS['bg_card'], bd=0)
        section_frame.pack(fill="both", expand=True, padx=4, pady=4)

        content = tk.Frame(section_frame, bg=COLORS['bg_card'])
        content.pack(fill="both", expand=True, padx=24, pady=20)

        # Header
        header = tk.Frame(content, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 16))

        icon_label = tk.Label(header, text="Report", font=("Inter", 12),
                             fg=COLORS['primary'], bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        title = tk.Label(header, text="Analysis Results", font=("Inter", 14, "bold"),
                        fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        title.pack(side="left", padx=(8, 0))

        # Text area
        text_frame = tk.Frame(content, bg=COLORS['bg_medium'])
        text_frame.pack(fill="both", expand=True)

        self.results_text = tk.Text(
            text_frame,
            font=("JetBrains Mono", 11),
            bg=COLORS['bg_medium'],
            fg="#e2e8f0",
            relief="flat",
            bd=0,
            padx=20,
            pady=20,
            insertbackground="#e2e8f0"
        )
        results_scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")

        # Export buttons
        export_frame = tk.Frame(content, bg=COLORS['bg_card'])
        export_frame.pack(fill="x", pady=(16, 0))

        export_json_btn = ModernButton(
            export_frame,
            text="Export JSON",
            icon="Save",
            command=self.export_json,
            bg_color=COLORS['purple'],
            hover_color=COLORS['purple_dark'],
            width=160
        )
        export_json_btn.pack(side="left", padx=(0, 12))

        export_csv_btn = ModernButton(
            export_frame,
            text="Export CSV",
            icon="Table",
            command=self.export_csv,
            bg_color=COLORS['warning'],
            hover_color="#d97706",
            width=160
        )
        export_csv_btn.pack(side="left", padx=(0, 12))

        export_txt_btn = ModernButton(
            export_frame,
            text="Export TXT Report",
            icon="Doc",
            command=self.export_txt,
            bg_color=COLORS['secondary'],
            hover_color="#475569",
            width=180
        )
        export_txt_btn.pack(side="left")

    def create_downloads_tab(self):
        """Create downloads tab for model outputs and evaluation data."""
        downloads_frame = tk.Frame(self.notebook, bg=COLORS['bg_light'])
        self.notebook.add(downloads_frame, text="  Downloads  ")

        # Evaluation Results Section
        eval_section = tk.Frame(downloads_frame, bg=COLORS['bg_card'], bd=0)
        eval_section.pack(fill="x", padx=4, pady=4)

        eval_content = tk.Frame(eval_section, bg=COLORS['bg_card'])
        eval_content.pack(fill="x", padx=24, pady=20)

        # Header
        header = tk.Frame(eval_content, bg=COLORS['bg_card'])
        header.pack(fill="x", pady=(0, 16))

        icon_label = tk.Label(header, text="Charts", font=("Inter", 12),
                             fg=COLORS['primary'], bg=COLORS['bg_card'])
        icon_label.pack(side="left")

        title = tk.Label(header, text="Model Evaluation Results", font=("Inter", 14, "bold"),
                        fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        title.pack(side="left", padx=(8, 0))

        # Description
        desc = tk.Label(eval_content,
                       text="Download the AI model's evaluation metrics and performance data:",
                       font=("Inter", 11), fg=COLORS['text_secondary'], bg=COLORS['bg_card'],
                       justify="left")
        desc.pack(anchor="w", pady=(0, 16))

        # CSV download buttons
        csv_buttons_frame = tk.Frame(eval_content, bg=COLORS['bg_card'])
        csv_buttons_frame.pack(fill="x")

        # Classification Report CSV
        class_btn = ModernButton(
            csv_buttons_frame,
            text="Classification Report",
            icon="CSV",
            command=lambda: self.download_eval_file("classification_report.csv"),
            bg_color=COLORS['success'],
            hover_color=COLORS['success_dark'],
            width=200
        )
        class_btn.pack(side="left", padx=(0, 12))

        # Per-Case Results CSV
        case_btn = ModernButton(
            csv_buttons_frame,
            text="Per-Case Results",
            icon="CSV",
            command=lambda: self.download_eval_file("per_case_results.csv"),
            bg_color=COLORS['primary'],
            hover_color=COLORS['primary_dark'],
            width=180
        )
        case_btn.pack(side="left", padx=(0, 12))

        # Confusion Matrix PNG
        matrix_btn = ModernButton(
            csv_buttons_frame,
            text="Confusion Matrix",
            icon="IMG",
            command=lambda: self.download_eval_file("confusion_matrix.png"),
            bg_color=COLORS['purple'],
            hover_color=COLORS['purple_dark'],
            width=180
        )
        matrix_btn.pack(side="left")

        # Download All button
        all_frame = tk.Frame(eval_content, bg=COLORS['bg_card'])
        all_frame.pack(fill="x", pady=(16, 0))

        download_all_btn = ModernButton(
            all_frame,
            text="Download All Evaluation Files",
            icon="Pack",
            command=self.download_all_eval_files,
            bg_color=COLORS['danger'],
            hover_color=COLORS['danger_dark'],
            width=280
        )
        download_all_btn.pack(anchor="w")

        # Model Performance Summary
        perf_section = tk.Frame(downloads_frame, bg=COLORS['bg_card'], bd=0)
        perf_section.pack(fill="x", padx=4, pady=(16, 4))

        perf_content = tk.Frame(perf_section, bg=COLORS['bg_card'])
        perf_content.pack(fill="x", padx=24, pady=20)

        # Header
        perf_header = tk.Frame(perf_content, bg=COLORS['bg_card'])
        perf_header.pack(fill="x", pady=(0, 16))

        perf_icon = tk.Label(perf_header, text="Stats", font=("Inter", 12),
                            fg=COLORS['warning'], bg=COLORS['bg_card'])
        perf_icon.pack(side="left")

        perf_title = tk.Label(perf_header, text="Model Performance Summary",
                             font=("Inter", 14, "bold"),
                             fg=COLORS['text_primary'], bg=COLORS['bg_card'])
        perf_title.pack(side="left", padx=(8, 0))

        # Performance stats
        stats_text = """
Overall Accuracy: 77.9%

Best Performing Phases:
  - Suture Tensioning: 93.8% F1-Score
  - Anchor Placement: 87.4% F1-Score
  - Portal Placement: 86.9% F1-Score
  - Suture Passage: 87.2% F1-Score

Areas for Improvement:
  - Glenoid Preparation: 4.8% F1-Score (limited training data)
  - Final Inspection: 42.3% F1-Score (short phase duration)
  - Labral Mobilization: 46.3% F1-Score

Tested on 4 surgical videos (5,584 frames)
"""
        stats_label = tk.Label(perf_content, text=stats_text, font=("JetBrains Mono", 10),
                              fg=COLORS['text_secondary'], bg=COLORS['bg_card'],
                              justify="left", anchor="w")
        stats_label.pack(anchor="w")

    def create_status_bar(self):
        """Create status bar."""
        status_frame = tk.Frame(self.root, bg=COLORS['border'], height=36)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Inter", 10),
            fg=COLORS['text_secondary'],
            bg=COLORS['border']
        )
        status_label.pack(side="left", padx=16, pady=8)

    def download_eval_file(self, filename):
        """Download a specific evaluation file."""
        source_path = Path("model-inference/evaluation_results") / filename

        if not source_path.exists():
            messagebox.showerror("Error", f"File not found: {filename}")
            return

        # Ask for save location
        if filename.endswith('.csv'):
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        elif filename.endswith('.png'):
            filetypes = [("PNG files", "*.png"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*")]

        dest_path = filedialog.asksaveasfilename(
            title=f"Save {filename}",
            defaultextension=Path(filename).suffix,
            initialfile=filename,
            filetypes=filetypes
        )

        if dest_path:
            try:
                shutil.copy2(source_path, dest_path)
                messagebox.showinfo("Success", f"File saved to:\n{dest_path}")
                self.status_var.set(f"Downloaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def download_all_eval_files(self):
        """Download all evaluation files to a folder."""
        source_dir = Path("model-inference/evaluation_results")

        if not source_dir.exists():
            messagebox.showerror("Error", "Evaluation results folder not found")
            return

        # Ask for destination folder
        dest_dir = filedialog.askdirectory(title="Select folder to save evaluation files")

        if dest_dir:
            try:
                dest_path = Path(dest_dir) / "evaluation_results"
                dest_path.mkdir(parents=True, exist_ok=True)

                files_copied = 0
                for file in source_dir.iterdir():
                    if file.is_file():
                        shutil.copy2(file, dest_path / file.name)
                        files_copied += 1

                messagebox.showinfo("Success",
                                   f"Copied {files_copied} files to:\n{dest_path}")
                self.status_var.set(f"Downloaded {files_copied} evaluation files")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy files: {str(e)}")

    def export_txt(self):
        """Export results as TXT report."""
        results = self.results_text.get(1.0, tk.END).strip()

        if not results:
            messagebox.showwarning("Warning", "No analysis results to export.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save TXT Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(results)
                messagebox.showinfo("Success", f"Report exported to {filename}")
                self.status_var.set(f"Exported TXT: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

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
            config_path = Path("surgical_ai_system/config/surgical_ai_config.yaml")
            self.config_manager = ConfigurationManager(str(config_path))

            models_dir = Path("surgical_ai_system/trained_models")
            models_dir.mkdir(parents=True, exist_ok=True)

            fallback_models_dir = Path("trained_models")
            if fallback_models_dir.exists():
                for model_file in fallback_models_dir.glob("*.pth"):
                    dest_path = models_dir / model_file.name
                    if not dest_path.exists():
                        shutil.copy2(model_file, dest_path)

                for config_file in fallback_models_dir.glob("*.json"):
                    dest_path = models_dir / config_file.name
                    if not dest_path.exists():
                        shutil.copy2(config_file, dest_path)

            self.status_var.set("Professional AI system initialized")

        except Exception as e:
            logger.warning(f"Could not fully initialize professional system: {e}")
            self.status_var.set("System initialized in basic mode")

    def load_model(self):
        """Load the professional AI models."""
        try:
            self.model_status_var.set("Loading professional AI models...")
            self.status_var.set("Loading comprehensive AI system...")

            models_dir = Path("surgical_ai_system/trained_models")
            config_path = Path("surgical_ai_system/config/surgical_ai_config.yaml")

            self.inference_engine = MasterInferenceEngine(
                config_path=str(config_path) if config_path.exists() else None,
                models_dir=str(models_dir)
            )

            self.model_status_var.set("Professional AI system loaded successfully!")
            self.status_var.set("AI models loaded - Ready for analysis")

        except Exception as e:
            logger.error(f"Failed to load professional system: {e}")

            try:
                self.model_status_var.set("Loading fallback model...")

                model_path = Path("surgical_ai_system/trained_models/phase_detector.pth")
                config_path = Path("surgical_ai_system/trained_models/model_configs.json")

                if not model_path.exists() or not config_path.exists():
                    messagebox.showerror("Error", "No AI models found. Please run training first.")
                    return

                with open(config_path, 'r') as f:
                    self.config = json.load(f)

                self.model = SimplePhaseModel(len(self.config['phase_labels']))
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()

                self.model_status_var.set("AI model loaded (phase detection)")
                self.status_var.set("Phase detector model loaded successfully")

            except Exception as e2:
                messagebox.showerror("Error", f"Failed to load any AI model: {str(e2)}")
                self.model_status_var.set("Failed to load AI models")

    def start_analysis(self):
        """Start video analysis in a separate thread."""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return

        if not self.inference_engine and not hasattr(self, 'model'):
            messagebox.showwarning("Warning", "Please load the AI model first.")
            return

        if not all([self.case_id_entry.get(), self.surgeon_entry.get()]):
            messagebox.showwarning("Warning", "Please fill in at least Case ID and Surgeon Name.")
            return

        self.progress_var.set("Starting analysis...")

        analysis_thread = threading.Thread(target=self.run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis(self):
        """Run the actual video analysis."""
        try:
            if self.inference_engine:
                self.run_professional_analysis()
            else:
                self.run_basic_analysis()

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set("Analysis failed"))

    def run_professional_analysis(self):
        """Run comprehensive analysis using professional inference engine."""
        self.root.after(0, lambda: self.progress_var.set("Preparing comprehensive AI analysis..."))

        case_metadata = {
            'case_id': self.case_id_entry.get(),
            'surgeon_id': self.surgeon_entry.get(),
            'procedure_type': self.procedure_entry.get(),
            'procedure_date': self.date_entry.get()
        }

        self.root.after(0, lambda: self.progress_var.set("Running multi-model AI analysis..."))

        try:
            def update_progress(message):
                self.root.after(0, lambda: self.progress_var.set(message))

            surgical_case = self.inference_engine.analyze_video(
                self.video_path, case_metadata, update_progress)

            self.root.after(0, lambda: self.progress_var.set("Generating comprehensive report..."))

            output_dir = Path("results") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_files = self.inference_engine.save_comprehensive_results(str(output_dir))

            results = self.generate_professional_report(surgical_case, saved_files)

            self.root.after(0, lambda: self.display_professional_results(results, surgical_case))
            self.root.after(0, lambda: self.progress_var.set("Comprehensive analysis complete!"))

            # Update metrics cards
            self.root.after(0, lambda: self.update_metrics(surgical_case))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Analysis failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_var.set("Analysis failed"))

    def run_basic_analysis(self):
        """Run basic analysis using simple phase detection model."""
        self.root.after(0, lambda: self.progress_var.set("Loading video..."))

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        predictions = []
        frame_count = 0
        sample_rate = 60

        self.root.after(0, lambda: self.progress_var.set("AI analyzing video frames..."))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                progress_pct = (frame_count / total_frames) * 100
                self.root.after(0, lambda p=progress_pct: self.progress_var.set(f"Analyzing... {p:.1f}%"))

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

        self.root.after(0, lambda: self.progress_var.set("Generating report..."))
        results = self.generate_basic_report(predictions, duration)

        self.root.after(0, lambda: self.display_results(results, predictions))
        self.root.after(0, lambda: self.progress_var.set("Basic analysis complete!"))

    def update_metrics(self, surgical_case):
        """Update metrics cards with analysis results."""
        self.cases_card.update_value("1")
        duration_min = surgical_case.video_duration / 60 if surgical_case.video_duration else 0
        self.duration_card.update_value(f"{duration_min:.1f} min")
        self.phases_card.update_value(str(len(surgical_case.phases)))

    def generate_professional_report(self, surgical_case, saved_files):
        """Generate comprehensive report from professional analysis."""
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
COMPREHENSIVE SURGICAL VIDEO ANALYSIS REPORT
{'='*60}

CASE INFORMATION:
   Case ID: {report['metadata']['case_id']}
   Surgeon: {report['metadata']['surgeon_id']}
   Analysis Type: Professional Multi-Model AI System

PROCEDURE METRICS:
   Total Duration: {report['surgical_analysis']['total_duration_minutes']:.2f} minutes
   Total Idle Time: {report['surgical_analysis']['total_idle_time_minutes']:.2f} minutes

SURGICAL PHASES ANALYSIS:
   - Diagnostic Arthroscopy: {report['phase_durations']['diagnostic_arthroscopy']:.2f} min
   - Glenoid Preparation: {report['phase_durations']['glenoid_preparation']:.2f} min
   - Labral Mobilization: {report['phase_durations']['labral_mobilization']:.2f} min
   - Anchor Placement: {report['phase_durations']['anchor_placement']:.2f} min
   - Suture Passage: {report['phase_durations']['suture_passage']:.2f} min
   - Suture Tensioning: {report['phase_durations']['suture_tensioning']:.2f} min
   - Final Inspection: {report['phase_durations']['final_inspection']:.2f} min

EVENT DETECTION:
   - Bleeding Events: {report['surgical_analysis']['bleeding_events']}
   - Suture Attempts: {report['surgical_analysis']['suture_attempts']}
   - Suture Failure Rate: {report['surgical_analysis']['suture_failure_rate']:.1f}%

INSTRUMENT & IMPLANT USAGE:
   - Number of Implants: {report['surgical_analysis']['number_of_implants']}
   - Instrument Events Detected: {report['surgical_analysis']['instruments_detected']}

EXPORTED FILES:
"""

        for file_type, file_path in report['saved_files'].items():
            results_text += f"   - {file_type.upper()}: {Path(file_path).name}\n"

        results_text += f"\nAnalysis completed: {report['analysis_timestamp']}"

        self.results_text.insert(1.0, results_text)

        self.last_professional_report = report
        self.last_surgical_case = surgical_case

        self.notebook.select(1)

    def generate_basic_report(self, predictions, duration):
        """Generate basic analysis report."""
        phase_counts = {}
        for pred in predictions:
            phase = pred['predicted_phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        metadata = {
            'case_id': self.case_id_entry.get(),
            'surgeon_name': self.surgeon_entry.get(),
            'procedure_type': self.procedure_entry.get(),
            'procedure_date': self.date_entry.get(),
        }

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

        self.last_report = report
        self.last_predictions = predictions

        return report

    def display_results(self, report, predictions):
        """Display results in the results tab."""
        self.results_text.delete(1.0, tk.END)

        results_text = f"""
SURGICAL VIDEO ANALYSIS REPORT
{'='*50}

CASE INFORMATION:
   Case ID: {report['metadata']['case_id']}
   Surgeon: {report['metadata']['surgeon_name']}
   Procedure: {report['metadata']['procedure_type']}
   Date: {report['metadata']['procedure_date']}

PROCEDURE ANALYSIS:
   Total Duration: {report['video_analysis']['total_duration_minutes']:.2f} minutes
   Frames Analyzed: {report['video_analysis']['frames_analyzed']}

AI PHASE DETECTION:
   Dominant Phase: {report['video_analysis']['dominant_phase']}
   Average Confidence: {report['video_analysis']['average_confidence']:.2f}

PHASE DISTRIBUTION:
"""

        for phase, count in report['video_analysis']['phase_distribution'].items():
            percentage = (count / len(predictions)) * 100
            results_text += f"   - {phase}: {count} frames ({percentage:.1f}%)\n"

        results_text += f"\nSAMPLE PREDICTIONS:\n"
        for pred in predictions[:10]:
            results_text += f"   {pred['timestamp_formatted']} -> {pred['predicted_phase']} ({pred['confidence']:.2f})\n"

        if len(predictions) > 10:
            results_text += f"   ... and {len(predictions) - 10} more predictions\n"

        results_text += f"\nAnalysis completed: {report['analysis_timestamp']}\n"

        self.results_text.insert(1.0, results_text)
        self.notebook.select(1)

    def export_json(self):
        """Export results as JSON."""
        if not hasattr(self, 'last_report') and not hasattr(self, 'last_professional_report'):
            messagebox.showwarning("Warning", "No analysis results to export.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save JSON Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                report = getattr(self, 'last_professional_report', None) or self.last_report
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
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
