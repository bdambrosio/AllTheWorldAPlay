#!/bin/bash
set -e

echo "=== ROS2 Cognitive Framework Test Runner ==="

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS2 environment not sourced. Please run:"
    echo "   source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "✓ ROS2 $ROS_DISTRO detected"

# Check if we're in a ROS2 workspace
if [ ! -f "setup.py" ]; then
    echo "❌ Not in cognitive framework directory. Please cd to the ros/ directory"
    exit 1
fi

# Check for required fix in package.xml
if grep -q "<n>cognitive_framework</n>" package.xml; then
    echo "❌ package.xml needs manual fix:"
    echo "   Please change <n>cognitive_framework</n> to <name>cognitive_framework</name>"
    echo "   in package.xml before building"
    exit 1
fi

# Find or create workspace
if [ -z "$COLCON_PREFIX_PATH" ]; then
    echo "Setting up workspace..."
    cd ..
    mkdir -p cognitive_ws/src
    cd cognitive_ws/src
    ln -sf ../../ros ./cognitive_framework
    cd ..
    WORKSPACE_ROOT="$(pwd)"
else
    WORKSPACE_ROOT="$(dirname $COLCON_PREFIX_PATH)"
fi

echo "Using workspace: $WORKSPACE_ROOT"

# Build the package
echo "Building cognitive_framework package..."
cd "$WORKSPACE_ROOT"
colcon build --packages-select cognitive_framework --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✓ Build successful"

# Source the workspace
source install/setup.bash

# Launch the system in background
echo "Launching cognitive system..."
ros2 launch cognitive_framework cognitive_system.launch.py &
LAUNCH_PID=$!

# Give nodes time to start
sleep 3

# Run the test
echo "Running system test..."
python3 src/cognitive_framework/test_cognitive_system.py &
TEST_PID=$!

# Wait for test to complete
wait $TEST_PID

# Cleanup
echo "Shutting down cognitive system..."
kill $LAUNCH_PID 2>/dev/null || true
sleep 2

# Kill any remaining nodes
pkill -f "cognitive_framework" 2>/dev/null || true

echo "✓ Test complete" 