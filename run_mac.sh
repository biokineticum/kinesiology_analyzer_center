#!/bin/bash
# Move to the script's directory
cd "$(dirname "$0")"

# Detect and activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the PySide6 native desktop application
python standalone/launch_desktop.py
