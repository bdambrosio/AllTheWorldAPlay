# Console Input Demo Guide

This guide demonstrates the interactive console input feature of the cognitive framework.

## 🚀 Quick Demo

### Step 1: Start the Cognitive System
```bash
# Option A: Launch all nodes
source /opt/ros/jazzy/setup.bash
ros2 launch cognitive_framework cognitive_system.launch.py

# Option B: Individual terminals (if you want to see each node's output)
# Terminal 1:
ros2 run cognitive_framework sense_node

# Terminal 2: 
ros2 run cognitive_framework memory_node

# Terminal 3:
ros2 run cognitive_framework action_node
```

### Step 2: Start Console Input Monitor (Optional)
```bash
# New terminal - monitors cognitive processing of your input
source /opt/ros/jazzy/setup.bash
python3 ros/test_console_input.py
```

### Step 3: Interact with the System

**Go to the sense_node terminal** (you'll see: "Console Text Sensor ready...")

**Type these test messages:**

```
Hello cognitive system
> You should see: [INFO] Text input received: "Hello cognitive system"

What is your purpose?
> Watch the logs for memory processing and action responses

I am feeling curious today
> Notice how console input gets high importance scores

Emergency situation!
> See how longer/urgent messages get processed differently

help
> Try short commands

Tell me about the weather
> Experiment with different types of input
```

## 🔍 What to Watch For

### In Sense Node Terminal:
- ✅ `Text input received: "your message"`
- ✅ `Published sense data #X [Console input: "your message"]`

### In Memory Node Terminal:
- ✅ High importance scores (0.8-1.0) for console input
- ✅ `Processed memory entry` with console content
- ✅ Memory consolidation when importance > 0.7

### In Action Node Terminal:
- ✅ Action responses to your input
- ✅ Higher priority actions for console input
- ✅ Goal updates based on user interaction

### In Console Input Monitor:
- 🎤 `SENSE detected console input: "your message"`
- 🧠 `MEMORY processed console input with importance: 0.85`
- ⚡ `ACTION triggered: investigate_important_stimulus`

## 🎯 Demo Scenarios

### Scenario 1: Simple Greeting
```
Input: "Hello"
Expected: High importance, memory storage, potential greeting response
```

### Scenario 2: Question
```
Input: "What can you do?"
Expected: Very high importance, goal creation, exploratory actions
```

### Scenario 3: Command
```
Input: "Stop what you are doing"
Expected: Immediate action response, goal interruption
```

### Scenario 4: Information
```
Input: "The temperature is 25 degrees"
Expected: Environmental context update, memory consolidation
```

## 🔧 Troubleshooting

### Console input not detected:
- ✅ Check: Are you typing in the **sense_node terminal**?
- ✅ Check: Are you pressing **Enter** after typing?
- ✅ Check: Is the sense node running and showing "Console Text Sensor ready"?

### No memory processing:
- ✅ Check: Is memory_node running and subscribed to sense_data?
- ✅ Check: Are you seeing memory updates in the logs?

### No action responses:
- ✅ Check: Is action_node running?
- ✅ Check: Try longer/more urgent messages for higher importance

### Input appearing twice:
- ✅ Normal: Input appears when received and when published in sense data

## 🎉 Success Indicators

You'll know it's working when you see:
1. **Input acknowledgment** in sense node
2. **High importance scores** in memory processing  
3. **Action responses** to your input
4. **Real-time processing** as you type

The cognitive system is now interactive and responsive to your text input! 🧠💬 