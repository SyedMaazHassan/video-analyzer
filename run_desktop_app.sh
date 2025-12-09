#!/bin/bash
# Launcher script for Desktop Application

cd "$(dirname "$0")"

# Activate virtual environment
source venv_desktop/bin/activate

# Run desktop app
python app/desktop_app.py



