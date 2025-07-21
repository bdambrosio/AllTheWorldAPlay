# Development Setup Guide for Cursor/VSCode

This guide helps you configure Cursor (or VSCode) for ROS2 cognitive framework development.

## 🚀 Quick Setup

**Option 1: Automatic Environment (Recommended)**
```bash
# Add to your ~/.bashrc or ~/.zshrc
echo "source $(pwd)/ros/.rosrc" >> ~/.bashrc
source ~/.bashrc
```

**Option 2: Manual Setup Each Session**
```bash
source /opt/ros/jazzy/setup.bash
```

## 🔧 IDE Configuration

### Files Created

I've set up the following files for your development environment:

1. **`.vscode/settings.json`** - Configures Python paths and ROS2 environment
2. **`.vscode/launch.json`** - Debug configurations for each node
3. **`.vscode/tasks.json`** - Common ROS2 commands as tasks
4. **`ros/.rosrc`** - Environment setup script

### Available Debug Configurations

In Cursor, press `F5` or go to "Run and Debug" and choose:

**Individual Nodes:**
- **Debug Sense Node** - Debug the perception node (console input available)
- **Debug Memory Node** - Debug the memory consolidation node  
- **Debug Action Node** - Debug the decision-making node
- **Test Cognitive System** - Run the full system test

**Multi-Node Launchers:**
- **Launch All Cognitive Nodes** - Start all three nodes in one session (console input ready!)
- **Debug All Cognitive Nodes** - Start all three with individual debug sessions

**Recommended for Console Input Testing:**
- Use **"Launch All Cognitive Nodes"** for simplest setup
- Use **"Debug All Cognitive Nodes"** for breakpoint debugging

### Available Tasks

Press `Ctrl+Shift+P` → "Tasks: Run Task" and choose:

- **ROS2: Build Cognitive Framework** - Build the package
- **ROS2: Launch Cognitive System** - Start all nodes
- **ROS2: Test Cognitive System** - Run tests
- **ROS2: Monitor Sense/Memory/Action Data** - Watch topic output

## 🐍 Python Import Issues Fixed

The configuration automatically handles:

✅ **rclpy imports** - ROS2 Python client library  
✅ **std_msgs imports** - Standard message types  
✅ **Node imports** - ROS2 node base classes  
✅ **IntelliSense** - Code completion for ROS2 APIs  
✅ **Debugging** - Breakpoints work in all nodes  

## 🔍 Testing Your Setup

### Test 1: Import Test
```bash
# Should work without errors:
python3 -c "import rclpy; from rclpy.node import Node; print('✓ ROS2 imports working')"
```

### Test 2: Quick Framework Test
```bash
# In terminal:
source ros/.rosrc
ros2-cognitive-test
```

### Test 3: IDE Debug Test
1. Open `ros/cognitive_framework/sense_node.py` in Cursor
2. Set a breakpoint in `sense_callback()` 
3. Press `F5` → "Debug Sense Node"
4. Should hit breakpoint within ~0.1 seconds

## 🎯 Development Workflow

### Typical Development Session:

1. **Start Cursor** - Environment auto-loads with settings
2. **Edit code** - IntelliSense works for ROS2 APIs
3. **Test changes** - Use tasks or debug configurations
4. **Monitor data** - Use topic monitoring tasks

### Common Commands:

```bash
# Quick environment setup
source ros/.rosrc

# Build after changes
colcon build --packages-select cognitive_framework

# Test the system
ros2-cognitive-test

# Launch for debugging
ros2-cognitive-launch
```

## 🔧 Troubleshooting

### "Import rclpy" errors:
- ✅ Check: ROS2 environment sourced
- ✅ Check: `.vscode/settings.json` has correct Python paths
- ✅ Check: Using system Python, not virtual environment

### IntelliSense not working:
- ✅ Reload Cursor window (`Ctrl+Shift+P` → "Developer: Reload Window")
- ✅ Check Python interpreter path in bottom status bar
- ✅ Verify `.vscode/settings.json` paths match your system

### Debug configurations not working:
- ✅ Check: `.vscode/launch.json` has ROS2 environment variables
- ✅ Check: Python path points to system Python with ROS2
- ✅ Try: "Python Debugger: Current File" as fallback

### Tasks failing:
- ✅ Check: ROS2 Jazzy is installed and accessible
- ✅ Check: Working directory is correct in terminal
- ✅ Try: Run commands manually first to verify

## 📁 File Structure Summary

```
.vscode/
├── settings.json     # IDE configuration
├── launch.json       # Debug configurations  
└── tasks.json        # Common commands

ros/
├── .rosrc           # Environment setup script
├── cognitive_framework/  # Your ROS2 package
│   ├── sense_node.py
│   ├── memory_node.py
│   └── action_node.py
└── README.md        # Package documentation
```

Your Cursor IDE is now fully configured for ROS2 cognitive framework development! 🎉 