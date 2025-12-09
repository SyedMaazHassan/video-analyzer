# Surgical Video Analysis Platform

AI-powered surgical video analysis for labral repair procedures. Automatically detect surgical phases, track events, and generate detailed reports.

---

## Quick Start (Mac/Linux)

```bash
# 1. Open Terminal and navigate to project
cd /path/to/video-analyzer

# 2. Activate virtual environment
source venv_desktop/bin/activate

# 3. Start the application (choose one):

# Option A: Web Interface (Recommended)
python app/web_app.py
# Then open browser to: http://127.0.0.1:5005

# Option B: Desktop Application
python surgical_video_gui.py
```

---

## What This Software Does

- **Detects Surgical Phases**: Automatically identifies 8 phases of labral repair surgery
- **Tracks Events**: Detects bleeding events, suture attempts, instrument usage
- **Generates Reports**: Creates PDF and text reports with detailed metrics
- **Visualizes Timelines**: Shows visual timeline of the entire procedure

**Model Accuracy: 77.9%** (tested on 5,584 frames from 4 surgical videos)

---

## Documentation

| Document | What It Covers |
|----------|----------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Step-by-step instructions for using the software |
| [MODEL_EVALUATION_RESULTS.md](MODEL_EVALUATION_RESULTS.md) | Detailed AI model performance metrics |

> Legacy documentation is available in `docs/legacy/`

---

## Features

| Feature | Web Interface | Desktop App |
|---------|---------------|-------------|
| Upload surgical videos | Yes | Yes |
| AI phase detection | Yes | Yes |
| Timeline visualization | Yes | Yes |
| PDF report generation | Yes | Yes |
| CSV data export | Yes | Yes |
| Model evaluation downloads | - | Yes |
| Surgeon management | Yes | - |
| Analytics charts | Yes | - |

---

## Surgical Phases Detected

1. Portal Placement
2. Diagnostic Arthroscopy
3. Glenoid Preparation
4. Labral Mobilization
5. Anchor Placement
6. Suture Passage
7. Suture Tensioning
8. Final Inspection

---

## Project Structure

```
video-analyzer/
├── app/                      # Web application
│   ├── web_app.py           # Flask server
│   └── templates/           # HTML templates
├── surgical_video_gui.py    # Desktop application
├── model-inference/         # AI model files
│   ├── best_model.pt       # Trained model
│   └── evaluation_results/ # Performance data
├── data/                    # Database and results
├── USER_GUIDE.md           # User documentation
└── MODEL_EVALUATION_RESULTS.md  # Model performance
```

---

## System Requirements

- Python 3.8+
- macOS, Windows, or Linux
- 8GB RAM recommended
- 2GB free disk space

---

## Troubleshooting

**"Command not found" error:**
```bash
# Make sure Python is installed
python3 --version

# On Mac, install if needed:
brew install python3
```

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Find what's using port 5005
lsof -i :5005

# Kill the process
kill -9 <PID>
```

---

## Getting Help

1. Read the [USER_GUIDE.md](USER_GUIDE.md) for detailed instructions
2. Check [MODEL_EVALUATION_RESULTS.md](MODEL_EVALUATION_RESULTS.md) to understand the AI performance
3. Look in `docs/legacy/` for additional technical documentation
