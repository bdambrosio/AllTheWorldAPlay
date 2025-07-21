# Quick Start Guide - ROS2 Cognitive Framework with Console Input

## ✅ Problem Fixed

The `RuntimeError: Context must be initialized before it can be shutdown` error has been **fixed**! 

Both test scripts now handle ROS2 environment issues gracefully and shut down cleanly.

## 🚀 Quick Test (30 seconds)

### 1. Test the Framework (No Input Required)
```bash
# Test the basic functionality
source /opt/ros/jazzy/setup.bash
python3 ros/test_cognitive_system.py
```
**Expected Output:** Health report showing no nodes (since none are running)

### 2. Test Console Input Feature
```bash
# Terminal 1: Start the cognitive system
source /opt/ros/jazzy/setup.bash
ros2 launch cognitive_framework cognitive_system.launch.py

# Terminal 2: Monitor processing (optional)
source /opt/ros/jazzy/setup.bash  
python3 ros/test_console_input.py

# In Terminal 1 (sense_node output), type:
Hello world
What can you do?
```

**Expected Output:**
- 🎤 Text input acknowledgment
- 🧠 High importance memory processing (0.8+)
- ⚡ Action responses to your input

## 🎯 Individual Node Testing

```bash
# Terminal 1: Sense node (type your input here!)
source /opt/ros/jazzy/setup.bash
ros2 run cognitive_framework sense_node

# Terminal 2: Memory processing  
source /opt/ros/jazzy/setup.bash
ros2 run cognitive_framework memory_node

# Terminal 3: Action decisions
source /opt/ros/jazzy/setup.bash  
ros2 run cognitive_framework action_node
```

## 📋 What Works Now

✅ **Clean startup/shutdown** - No more RuntimeError  
✅ **Console input detection** - Type messages in sense_node terminal  
✅ **High importance processing** - Console input gets 0.8-1.0 priority  
✅ **Real-time feedback** - See cognitive processing as you type  
✅ **Graceful interruption** - Ctrl+C or timeout works cleanly  
✅ **Environment validation** - Clear error messages if ROS2 not sourced  

## 🔧 Troubleshooting

### "ROS2 environment not sourced"
```bash
source /opt/ros/jazzy/setup.bash
```

### "Failed to publish log message to rosout" 
- ✅ **Harmless warning** during shutdown - ignore it

### Console input not working
- ✅ Type in the **sense_node terminal** (not test terminals)
- ✅ Press **Enter** after typing  
- ✅ Look for "Console Text Sensor ready" message

### No cognitive responses
- ✅ Start all three nodes (sense, memory, action)
- ✅ Try longer, more interesting messages
- ✅ Check logs for processing activity

## 🎉 Success Indicators

You'll know it's working when you see:

1. **Sense Node**: `Text input received: "your message"`
2. **Memory Node**: `importance: 0.85` for console input  
3. **Action Node**: Actions triggered by your input
4. **Test Monitor**: Real-time processing updates

**Your cognitive framework is ready for interactive testing!** 🧠💬 