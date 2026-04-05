#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Stock Predictor Launcher ---

# Define the name of the virtual environment directory
VENV_DIR="ml"

# --- 1. Set Up Virtual Environment ---
echo "🔧 Checking virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Virtual environment not found. Creating one named '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
else
    echo "✅ Virtual environment '$VENV_DIR' found."
fi

# --- 2. Activate Virtual Environment ---
echo "🔌 Activating virtual environment..."
source $VENV_DIR/bin/activate


# --- 5. Run the Application ---
echo "🚀 Starting the Streamlit application..."
echo "   The app will be available at http://localhost:8501"
echo "   Press CTRL+C in the terminal to stop the server."
echo "----------------------------------------------------"

streamlit run frontend/app.py