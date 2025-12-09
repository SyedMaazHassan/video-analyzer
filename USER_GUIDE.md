# Surgical Video Analysis Platform - User Guide

Welcome to the Surgical Video Analysis Platform! This guide will walk you through everything you need to know to analyze surgical videos using our AI-powered system.

---

## Table of Contents

1. [What This Software Does](#what-this-software-does)
2. [System Requirements](#system-requirements)
3. [Setting Up the Environment](#setting-up-the-environment-first-time-only) (First Time Only)
4. [Getting Started](#getting-started)
5. [Using the Web Interface](#using-the-web-interface)
6. [Using the Desktop Application](#using-the-desktop-application)
7. [Understanding the Analysis Results](#understanding-the-analysis-results)
8. [Downloading Reports and Data](#downloading-reports-and-data)
9. [Troubleshooting](#troubleshooting)

---

## What This Software Does

The Surgical Video Analysis Platform uses artificial intelligence to automatically analyze surgical procedure videos. It can:

- **Detect Surgical Phases**: Automatically identify different stages of a surgery (e.g., Portal Placement, Diagnostic Arthroscopy, Anchor Placement, etc.)
- **Track Events**: Detect important surgical events like bleeding, suture attempts, and instrument usage
- **Generate Reports**: Create detailed reports with timing information, phase breakdowns, and performance metrics
- **Visualize Timelines**: Show a visual timeline of the entire surgical procedure

The AI model has been trained specifically for **labral repair** (shoulder arthroscopy) procedures and achieves **77.9% overall accuracy** in phase detection.

---

## System Requirements

- **Operating System**: macOS, Windows, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: At least 8GB recommended
- **Storage**: At least 2GB free space for the application and models

---

## Setting Up the Environment (First Time Only)

If this is your first time using the software, follow these steps to set up the environment. If you've already set up the environment, skip to [Getting Started](#getting-started).

### Step 1: Install Python (if not already installed)

**On Mac:**

Option A - Using Homebrew (Recommended):
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11
```

Option B - Download from Python.org:
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or later
3. Run the installer and follow the prompts

**Verify Python is installed:**
```bash
python3 --version
```
You should see something like `Python 3.11.x`

### Step 2: Open Terminal and Navigate to Project

```bash
# Open Terminal (Cmd + Space, type "Terminal", press Enter)

# Navigate to where you downloaded/saved the project
cd /path/to/video-analyzer

# Example:
cd ~/Downloads/video-analyzer
```

### Step 3: Create the Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects on your computer.

```bash
# Create a new virtual environment called "venv_desktop"
python3 -m venv venv_desktop
```

This creates a folder called `venv_desktop` in your project directory.

### Step 4: Activate the Virtual Environment

```bash
source venv_desktop/bin/activate
```

You'll know it worked when you see `(venv_desktop)` at the beginning of your terminal prompt:
```
(venv_desktop) your-computer:video-analyzer username$
```

### Step 5: Install Required Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This may take several minutes as it downloads and installs:
- PyTorch (AI/machine learning framework)
- OpenCV (video processing)
- Flask (web server)
- And other dependencies

**Note:** If you see warnings (yellow text), that's usually fine. Only red error messages indicate problems.

### Step 6: Verify Installation

```bash
# Test that the main packages are installed
python -c "import torch; import cv2; import flask; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!", you're ready to go!

### Troubleshooting Setup Issues

**"pip: command not found"**
```bash
# Try using pip3 instead
pip3 install -r requirements.txt
```

**"Permission denied" errors**
```bash
# Add --user flag
pip install --user -r requirements.txt
```

**Installation fails or freezes**
```bash
# Try installing packages one at a time
pip install torch torchvision
pip install opencv-python-headless
pip install flask pandas numpy
pip install -r requirements.txt
```

**"No module named tkinter" (Desktop app only)**

On Mac:
```bash
brew install python-tk@3.11
```

---

## Getting Started

Once your environment is set up, follow these steps each time you want to use the software.

### Step 1: Open Terminal

**On Mac:**
- Press `Cmd + Space` to open Spotlight
- Type "Terminal" and press Enter

### Step 2: Navigate to the Project Folder

```bash
cd /path/to/video-analyzer
```

Replace `/path/to/video-analyzer` with the actual location where you saved the project. For example:
```bash
cd ~/Documents/video-analyzer
```

### Step 3: Activate the Virtual Environment

```bash
source venv_desktop/bin/activate
```

You should see `(venv_desktop)` appear at the beginning of your terminal prompt, indicating the environment is active.

### Step 4: Choose Your Interface

You have two options:

**Option A: Web Interface** (recommended for beginners)
```bash
python app/web_app.py
```
Then open your web browser and go to: `http://127.0.0.1:5005`

**Option B: Desktop Application**
```bash
python surgical_video_gui.py
```
This will open a desktop window application.

---

## Using the Web Interface

The web interface is the easiest way to use the platform. Here's how to navigate it:

### Dashboard Overview

When you open the web interface, you'll see:

- **Metrics Cards** (top): Shows summary statistics
  - Total Cases: Number of analyzed cases
  - Avg Duration: Average procedure time
  - Bleeding Events: Average per case
  - Suture Success Rate: Percentage of successful sutures

- **Navigation Tabs**: Click to switch between sections
  - Cases: View all analyzed cases
  - Analytics: Charts and graphs
  - Surgeons: Manage surgeon profiles
  - Upload Case: Add new surgical videos
  - Add Surgeon: Register new surgeons

### Uploading a New Case

1. Click the **"Upload Case"** tab
2. Fill in the form:
   - **Case ID**: Enter a unique identifier (e.g., "CASE001")
   - **Surgeon**: Select from the dropdown
   - **Procedure Type**: Usually "Labral Repair"
   - **Video File**: Click to select your surgical video (MP4, AVI, MOV, or MKV)
   - **Notes**: Add any additional notes (optional)
3. Click **"Upload and Process Case"**
4. Wait approximately 15-30 seconds for the AI to analyze the video
5. Go to the **"Cases"** tab to see your analyzed case

### Viewing Case Results

1. Click the **"Cases"** tab
2. Find your case in the list
3. Click **"View"** to see detailed results including:
   - Timeline visualization
   - Phase breakdown with durations
   - Detected events
   - Resource usage

### Downloading Reports

From the case details view:
- Click **"View Text Report"** to see the full report
- Click **"Download PDF Report"** to save as PDF

---

## Using the Desktop Application

The desktop application provides similar functionality in a standalone window:

### Main Tabs

1. **Analysis Tab**: This is where you start
   - Select a video file
   - Enter case information
   - Run AI analysis

2. **Results Tab**: View analysis results and export data
   - See the detailed report
   - Export as JSON, CSV, or TXT

3. **Downloads Tab**: Access model evaluation data
   - Download classification reports
   - Download per-case results
   - Download confusion matrix

### Analyzing a Video

1. In the **Analysis** tab, click **"Select Video File"**
2. Choose your surgical video from your computer
3. Fill in the case information:
   - Case ID
   - Surgeon Name
   - Procedure Type
   - Date
4. Click **"Load AI Model"** (wait for it to load)
5. Click **"Start Analysis"**
6. Wait for the analysis to complete (progress shown on screen)
7. View results in the **Results** tab

### Exporting Results

In the **Results** tab, you can:
- **Export JSON**: Full structured data for technical use
- **Export CSV**: Spreadsheet-compatible frame-by-frame predictions
- **Export TXT Report**: Plain text report for documentation

---

## Understanding the Analysis Results

### Surgical Phases Detected

The AI recognizes these surgical phases:

| Phase | Description |
|-------|-------------|
| Portal Placement | Initial port creation for camera and instruments |
| Diagnostic Arthroscopy | Camera inspection of the joint |
| Glenoid Preparation | Preparing the bone surface |
| Labral Mobilization | Freeing the labral tissue |
| Anchor Placement | Inserting suture anchors |
| Suture Passage | Threading sutures through tissue |
| Suture Tensioning | Tightening and securing sutures |
| Final Inspection | Final check of the repair |


### Metrics Explained

- **Total Duration**: Total length of the procedure in minutes
- **Phase Durations**: Time spent in each surgical phase
- **Suture Success Rate**: Percentage of successful suture attempts
- **Idle Time**: Time when no active surgical action is detected

---

## Downloading Reports and Data

### From the Web Interface

| Data Type | How to Download | Location |
|-----------|-----------------|----------|
| PDF Report | Click "Download PDF Report" in case details | Saves to your Downloads folder |
| Text Report | Click "View Text Report" then copy/save | Displays in browser |
| Chart Images | Click "Download Chart" on Analytics tab | Saves as PNG |

### From the Desktop Application

| Data Type | How to Download | Button |
|-----------|-----------------|--------|
| JSON Report | Results tab | "Export JSON" |
| CSV Data | Results tab | "Export CSV" |
| Text Report | Results tab | "Export TXT Report" |
| Classification Report | Downloads tab | "Classification Report" |
| Per-Case Results | Downloads tab | "Per-Case Results" |
| Confusion Matrix | Downloads tab | "Confusion Matrix" |
| All Evaluation Files | Downloads tab | "Download All Evaluation Files" |

---

## Troubleshooting

### "Command not found" Error

If you see this error, make sure you:
1. Have Python installed
2. Are in the correct directory
3. Have activated the virtual environment

```bash
# Check Python is installed
python3 --version

# If not found on Mac, install with:
brew install python3
```

### "Module not found" Error

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Web Interface Won't Load

1. Make sure the server is running (check Terminal for errors)
2. Try a different browser
3. Clear your browser cache
4. Check if port 5005 is available:
```bash
lsof -i :5005
```

### Video Won't Upload

- Ensure the video format is supported (MP4, AVI, MOV, MKV)
- Check if the file size isn't too large (recommended under 2GB)
- Make sure you have enough disk space

### Analysis Takes Too Long

- Longer videos take more time to analyze
- A 40-minute video typically takes 1-2 minutes to process
- Check if your computer has enough RAM available

### AI Model Won't Load

```bash
# Check if model files exist
ls model-inference/

# If missing, the model needs to be retrained or downloaded
```

---



## Quick Reference

### Starting the Web Interface
```bash
cd /path/to/video-analyzer
source venv_desktop/bin/activate
python app/web_app.py
```
Then open: `http://127.0.0.1:5005`

### Starting the Desktop App
```bash
cd /path/to/video-analyzer
source venv_desktop/bin/activate
python surgical_video_gui.py
```

### Stopping the Application
- **Web Interface**: Press `Ctrl + C` in Terminal
- **Desktop App**: Close the window or press `Cmd + Q`
