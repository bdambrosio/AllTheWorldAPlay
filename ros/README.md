# Cognitive Framework ROS2 Package

A basic cognitive architecture implemented using ROS2, featuring three core components:

- **Sense Node**: Perception and sensory input processing
- **Memory Node**: Information storage and consolidation  
- **Action Node**: Decision making and execution

## Architecture

```
sense_node → /cognitive/sense_data → memory_node → /cognitive/memory_data → action_node
     ↓                                                                            ↑
     └─────────────────────────→ /cognitive/sense_data ────────────────────────┘
                                                      ↓
                                              /cognitive/action_data
```

## Prerequisites

- ROS2 (tested with Humble/Iron)
- Python 3.8+
- rclpy
- std_msgs

## Installation

1. **Navigate to your ROS2 workspace**:
   ```bash
   cd ~/ros2_ws/src  # or your workspace location
   ```

2. **Clone/copy this package**:
   ```bash
   # If part of AllTheWorldAPlay, it's already in ros/
   ln -s /path/to/AllTheWorldAPlay/ros ./cognitive_framework
   ```

3. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select cognitive_framework
   source install/setup.bash
   ```

## Usage

### Option 1: Launch All Nodes (Recommended)

```bash
# Launch the complete cognitive system
ros2 launch cognitive_framework cognitive_system.launch.py

# With custom namespace and log level
ros2 launch cognitive_framework cognitive_system.launch.py namespace:=my_agent log_level:=debug
```

### Option 2: Run Individual Nodes

```bash
# Terminal 1: Sense node (with console input)
ros2 run cognitive_framework sense_node

# Terminal 2: Memory node  
ros2 run cognitive_framework memory_node

# Terminal 3: Action node
ros2 run cognitive_framework action_node

# Terminal 4: Monitor console input processing (optional)
python3 ros/test_console_input.py
```

## Monitoring the System

### View Topics
```bash
# List all cognitive topics
ros2 topic list | grep cognitive

# Monitor sense data (high frequency)
ros2 topic echo /cognitive/sense_data

# Monitor memory updates
ros2 topic echo /cognitive/memory_data

# Monitor action commands
ros2 topic echo /cognitive/action_data
```

### Check Node Status
```bash
# List running nodes
ros2 node list

# Get node info
ros2 node info /sense_node
ros2 node info /memory_node  
ros2 node info /action_node
```

## 🎤 Interactive Console Input

The sense node includes a **console text sensor** that reads user input from the terminal:

### How to Use:
1. **Start the cognitive system** (any method above)
2. **Go to the sense_node terminal** (where you see "Console Text Sensor ready...")
3. **Type messages and press Enter** - they'll be processed by the cognitive system
4. **Watch the processing** in real-time through the logs

### Example Session:
```bash
# In sense_node terminal:
Hello cognitive system
> [INFO] Text input received: "Hello cognitive system"
> [INFO] Published sense data #45 [Console input: "Hello cognitive system"]

What is 2+2?
> [INFO] Text input received: "What is 2+2?"
> [INFO] Published sense data #67 [Console input: "What is 2+2?"]
```

Console input gets **high importance** (0.8-1.0) in memory processing and triggers cognitive responses!

## Message Flow

1. **Sense Node** publishes hybrid sensor data at 10Hz (simulated + console input):
   ```json
   {
     "timestamp": "2025-01-09T...",
     "sequence_id": 123,
     "sensor_type": "hybrid", 
     "data": {
       "visual": "visual_input_123",
       "audio": "audio_input_123",
       "text": {
         "simulated": "text_input_123",
         "console_input": "Hello cognitive system",
         "last_console_input": "Hello cognitive system"
       },
       "console_text_sensor": {
         "new_input": true,
         "current_input": "Hello cognitive system",
         "input_timestamp": "2025-01-09T..."
       },
       "environment": {
         "light_level": 0.65,
         "noise_level": 0.23
       }
     }
   }
   ```

2. **Memory Node** processes and stores sensory data, publishes memory updates:
   ```json
   {
     "type": "memory_update",
     "memory_entry": {...},
     "working_memory": {...},
     "memory_stats": {
       "short_term_count": 45,
       "long_term_count": 12
     }
   }
   ```

3. **Action Node** receives both streams, makes decisions, publishes actions:
   ```json
   {
     "action_id": 67,
     "source": "deliberative",
     "action": {
       "type": "explorative",
       "action": "pursue_opportunity",
       "priority": 0.78
     }
   }
   ```

## Customization

### Modify Sensing
Edit `cognitive_framework/sense_node.py`:
- Change sensor simulation in `sense_callback()`
- Adjust publishing frequency in timer creation
- Add real sensor interfaces

### Modify Memory Processing  
Edit `cognitive_framework/memory_node.py`:
- Adjust memory consolidation logic in `consolidate_memory()`
- Change importance calculation in `_calculate_importance()`
- Modify memory structures (STM/LTM/WM)

### Modify Decision Making
Edit `cognitive_framework/action_node.py`:
- Update reactive rules in `_check_reactive_actions()`
- Modify situation assessment in `_assess_situation()`
- Change action selection logic in `_select_action()`

## Integration with AllTheWorldAPlay

This ROS2 framework is designed to eventually replace or augment the monolithic cognitive system in AllTheWorldAPlay's `agh.py`. Key integration points:

1. **Replace DriveSignalManager** → distributed across memory/action nodes
2. **Replace MemoryConsolidator** → memory_node  
3. **Replace decision making** → action_node
4. **Maintain LLM integration** → add to action_node for reasoning
5. **Preserve character state** → distribute across all nodes

## Troubleshooting

### Nodes not communicating
- Check topic names: `ros2 topic list`
- Verify QoS settings match between publishers/subscribers
- Ensure all nodes are in same namespace

### High CPU usage
- Reduce sensing frequency in sense_node timer
- Increase decision timer interval in action_node
- Adjust memory consolidation frequency

### No actions being generated
- Check action_node logs for decision threshold issues
- Verify both sense and memory data are being received
- Lower `decision_threshold` in action_node

## Future Extensions

- Custom message types instead of JSON strings
- Parameter configuration via ROS2 parameters
- Service interfaces for querying node state
- Integration with real sensors/actuators
- LLM integration for reasoning
- Persistent memory storage 