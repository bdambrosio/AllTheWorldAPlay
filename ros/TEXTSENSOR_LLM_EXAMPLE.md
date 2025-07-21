# TextSensor-to-LLM-Action Pattern Example

This example demonstrates the specific pattern requested: **Action node receives textSensor input and calls LLM with that content**.

## 🎯 **The Pattern**

```
User Input → TextSensor → Sense Message → Action Node → LLM → Action Response
```

**Detailed Flow:**
1. User types in sense node terminal
2. Sense node publishes message with `textSensor` data  
3. Action node receives sense message
4. Action node extracts `textSensor` content
5. Action node calls LLM with that text content
6. LLM processes the text and returns analysis
7. Action node creates action based on LLM response
8. Action node publishes the action

## 🚀 **Quick Demo**

### **Method 1: VSCode Launch (Recommended)**

1. **Press `F5` in VSCode**
2. **Choose: "Text-to-LLM-Action Example"**
3. **Wait for all nodes to start**
4. **Type messages in the Sense Node terminal**
5. **Watch the pattern flow in all terminals!**

### **Method 2: Manual Launch**

```bash
# Terminal 1: Start LLM service
source /opt/ros/jazzy/setup.bash
ros2 run cognitive_framework llm_service_node

# Terminal 2: Start sense node (with textSensor)
ros2 run cognitive_framework sense_node

# Terminal 3: Start simple LLM action example
ros2 run cognitive_framework simple_llm_action_example

# Terminal 4: Monitor the pattern flow
python3 ros/test_textsensor_llm_action.py
```

## 📋 **Code Walkthrough**

### **Key Code in `simple_llm_action_example.py`:**

```python
def sense_data_callback(self, msg):
    """
    This is the main callback that demonstrates the requested pattern.
    """
    # Parse the sense data
    sense_data = json.loads(msg.data)
    
    # Extract textSensor information
    console_text_sensor = sense_data.get('data', {}).get('console_text_sensor', {})
    
    # Check if we have new text input
    if console_text_sensor.get('new_input', False):
        current_input = console_text_sensor.get('current_input', '')
        
        if current_input.strip():
            # 🎯 THIS IS THE KEY PATTERN: Call LLM with textSensor content
            self._process_text_input_with_llm(current_input, sense_data)

def _process_text_input_with_llm(self, text_input: str, sense_data: dict):
    """Process textSensor input using LLM and create appropriate action."""
    
    # Create system prompt for the LLM
    system_prompt = """You are a cognitive action system. A user has provided text input to your sensors.
Analyze their input and suggest a specific action the system should take."""
    
    # Create user prompt with the textSensor content
    user_prompt = f"""User input from textSensor: "{text_input}"
    
What action should the cognitive system take in response to this input?"""
    
    # BLOCKING call - wait for LLM response
    response = self.llm_client.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=100,
        temperature=0.7,
        timeout=10.0
    )
    
    if response.success:
        # Create action based on LLM response
        action = self._create_action_from_llm_response(
            text_input, 
            response.text, 
            'blocking_llm',
            response.processing_time
        )
        
        # Publish the action
        self._publish_action(action)
```

## 📊 **Expected Output**

When you type **"What should I focus on?"** in the sense node terminal:

```
# Sense Node Terminal:
[sense_node] 🎤 Text input received: "What should I focus on?"
[sense_node] Published sense data #47 [Console input: "What should I focus on?"]

# LLM Service Terminal:
[llm_service_node] 📥 Received LLM request req_12: "User input from textSensor: 'What should I focus on?'..."
[llm_service_node] 📤 Completed LLM request req_12 in 1.2s

# Action Node Terminal:
[simple_llm_action_example] 📥 Received textSensor input: "What should I focus on?"
[simple_llm_action_example] 🧠 Processing textSensor input with LLM: "What should I focus on?"
[simple_llm_action_example] ⏳ Making BLOCKING LLM call...
[simple_llm_action_example] ✅ LLM response received: "Focus on analyzing incoming sensory data and prioritizing high-importance information for decision making."
[simple_llm_action_example] 🤖 Published llm_guided_action action #3: Response to "What should I focus on?" (priority: 0.80)

# Pattern Monitor Terminal:
[tester] 🎯 STEP 1-2: TextSensor Input Detected [flow_0]
[tester]     📝 User input: "What should I focus on?"
[tester] 🎯 STEP 3-5: LLM Request Sent [flow_0]
[tester]     🤖 Action node extracted: "What should I focus on?"
[tester] 🎯 STEP 6-7: LLM Response Received [flow_0]
[tester]     ✅ Success: True
[tester]     ⏱️  Processing time: 1.20s
[tester] 🎯 STEP 8-9: Action Created & Published [flow_0]
[tester]     🎬 Action type: llm_guided_action (blocking_llm)
[tester] 🎉 COMPLETE FLOW SUMMARY:
[tester]     📝 User Input: "What should I focus on?"
[tester]     🤖 LLM Analysis: "Focus on analyzing incoming sensory data and prioritizing high-importance..."
[tester]     🎬 Resulting Action: llm_guided_action (ID: 3)
[tester]     ⏱️  Total Flow: 1.45s
[tester] ✅ Pattern completed successfully!
```

## 🔍 **Key Features Demonstrated**

### **1. TextSensor Content Extraction**
```python
# Extract textSensor data from sense message
console_text_sensor = sense_data.get('data', {}).get('console_text_sensor', {})
if console_text_sensor.get('new_input', False):
    text_input = console_text_sensor.get('current_input', '')
```

### **2. Blocking LLM Call**
```python
# Blocking call - action node waits for LLM response
response = self.llm_client.generate(
    prompt=user_prompt,
    system_prompt=system_prompt,
    timeout=10.0
)
```

### **3. Non-Blocking LLM Call (also shown)**
```python
# Non-blocking call with callback for background analysis
future = self.llm_client.generate_async(
    prompt=analysis_prompt,
    callback=analysis_callback
)
```

### **4. Action Creation from LLM Response**
```python
action = {
    'action_id': self.action_counter,
    'type': 'llm_guided_action',
    'trigger': {
        'source': 'textSensor',
        'input': original_input
    },
    'llm_analysis': {
        'response': llm_response,
        'processing_time_seconds': processing_time
    },
    'action_plan': {
        'llm_guidance': llm_response,
        'priority': 0.8
    }
}
```

## 🎮 **Try These Example Inputs**

Type these in the sense node terminal to see different LLM responses:

- **"What should I focus on?"** - Gets strategic guidance
- **"I'm feeling confused"** - Gets emotional support response  
- **"How can I help?"** - Gets proactive action suggestions
- **"Tell me about the environment"** - Gets environmental analysis
- **"What's my next step?"** - Gets procedural guidance

## 🔧 **Customization**

To customize the pattern for your needs:

1. **Modify the system prompt** in `_blocking_llm_example()` to change LLM behavior
2. **Add different action types** in `_create_action_from_llm_response()`
3. **Change LLM parameters** (temperature, max_tokens) for different styles
4. **Add filtering** to only process certain types of text input
5. **Add context** by including other sensor data in the LLM prompt

## 🎯 **Files Involved**

- **`simple_llm_action_example.py`** - The main example action node
- **`test_textsensor_llm_action.py`** - Pattern flow monitor
- **`llm_service_node.py`** - LLM service wrapper
- **`llm_client.py`** - Client library for LLM access
- **`sense_node.py`** - Provides textSensor input

## ✅ **Success Criteria**

You've successfully demonstrated the pattern when you see:

1. ✅ User input appears in sense node
2. ✅ Action node receives and processes the input
3. ✅ LLM request is sent with textSensor content
4. ✅ LLM response is received
5. ✅ Action is created based on LLM analysis
6. ✅ Action is published with LLM guidance
7. ✅ Pattern monitor shows complete flow

**🎉 Your textSensor-to-LLM-action pattern is working perfectly!** 