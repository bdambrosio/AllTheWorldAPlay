#!/bin/bash

# Basic error handling
set -e  # Exit on error

# Activate virtual environment
if [ ! -d "owl" ]; then
    echo "Error: owl directory not found. Are you in the src directory?"
    exit 1
fi
source owl/bin/activate

# Set environment variables
export PYTHONPATH="$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/owl/lib/python3.12/site-packages/PyQt5/Qt5/lib"
export QT_PLUGIN_PATH="$(pwd)/owl/lib/python3.12/site-packages/PyQt5/Qt5/plugins"
export CUDA_VISIBLE_DEVICES="0"

# Start main.py using uvicorn
cd sim/webworld/src
echo "Starting main.py..."
uvicorn main:app --reload --port 8000 &
MAIN_PID=$!

# Start stable diffusion server
cd ../../../utils
echo "Starting image server..."
uvicorn sd3_serve:app --host 0.0.0.0 --port 5008 &
SD_PID=$!

# Start React app
cd ../sim/webworld
echo "Starting React app..."
npm start &
REACT_PID=$!

# Cleanup on exit
trap 'kill $MAIN_PID $SD_PID $REACT_PID 2>/dev/null' EXIT

# Keep script running
wait 