#!/bin/bash
# Script to run ROS2 commands with the owl virtual environment active

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
OWL_VENV_DIR="$WORKSPACE_DIR/src/owl"

# Check if owl venv exists
if [ ! -d "$OWL_VENV_DIR" ]; then
    echo "❌ Owl virtual environment not found at: $OWL_VENV_DIR"
    echo "Please ensure the owl venv is properly installed."
    exit 1
fi

# Source ROS2 environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "✅ ROS2 Jazzy environment sourced"
else
    echo "❌ ROS2 Jazzy not found. Please install ROS2 Jazzy."
    exit 1
fi

# Activate owl virtual environment
if [ -f "$OWL_VENV_DIR/bin/activate" ]; then
    source "$OWL_VENV_DIR/bin/activate"
    echo "✅ Owl virtual environment activated"
    echo "   Python: $(which python3)"
    echo "   Virtual env: $VIRTUAL_ENV"
else
    echo "❌ Owl venv activation script not found at: $OWL_VENV_DIR/bin/activate"
    exit 1
fi

# Set additional environment variables
export PYTHONPATH="/opt/ros/jazzy/lib/python3.12/site-packages:/opt/ros/jazzy/local/lib/python3.12/dist-packages:$WORKSPACE_DIR/ros:$WORKSPACE_DIR/src"
export ROS_DISTRO="jazzy"

# Check if LLM API is available
echo "🔍 Checking LLM API availability..."
python3 -c "
try:
    from src.utils.llm_api import LLM
    print('✅ LLM API available')
except ImportError as e:
    print(f'⚠️  LLM API not available: {e}')
    print('   Will use mock responses')
" 2>/dev/null

echo ""
echo "🚀 Ready to run ROS2 commands with owl venv!"
echo ""

# If arguments are provided, run them as a command
if [ $# -gt 0 ]; then
    echo "Running: $@"
    exec "$@"
else
    # If no arguments, start an interactive shell with the environment set up
    echo "Starting interactive shell with owl venv and ROS2..."
    echo "You can now run ROS2 commands like:"
    echo "  ros2 run cognitive_framework llm_service_node"
    echo "  ros2 run cognitive_framework simple_llm_action_example"
    echo ""
    bash
fi 