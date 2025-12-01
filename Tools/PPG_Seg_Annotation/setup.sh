#!/bin/bash

# Define project directory
PROJECT_DIR="PPG_Annotator_Mac"

echo "=========================================="
echo "   PPG Annotator - MacOS Setup Assistant"
echo "=========================================="

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Directory '$PROJECT_DIR' not found."
    echo "Please run 'python3 build_project.py' first."
    exit 1
fi

cd "$PROJECT_DIR"

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 could not be found."
    exit 1
fi

# 2. Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔹 Creating virtual environment..."
    python3 -m venv venv
else
    echo "🔹 Virtual environment found."
fi

# 3. Activate Virtual Environment
echo "🔹 Activating environment..."
source venv/bin/activate

# 4. Install Dependencies
echo "🔹 Installing dependencies (this may take a minute)..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

# 5. Run the Application
echo "=========================================="
echo "   🚀 Launching Application..."
echo "=========================================="

# We execute python directly
python main.py
