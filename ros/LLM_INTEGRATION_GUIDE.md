# LLM Integration Guide for ROS2 Cognitive Framework

This guide shows how to integrate your existing `llm_api.py` with the ROS2 cognitive framework using both **blocking** and **non-blocking** patterns.

## 🎯 **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Cognitive      │    │   LLM Service    │    │    Your LLM     │
│  Nodes          │────│   Node           │────│    API          │
│  (Memory/Action)│    │  (Thread Pool)   │    │  (llm_api.py)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
    Non-blocking              Concurrent              Blocking
    LLM requests             Processing              LLM calls
```

**Benefits:**
- ✅ **Multiple nodes can call LLM concurrently** without blocking each other
- ✅ **Your existing `llm_api.py` is reentrant** - multiple threads can use it safely
- ✅ **Blocking calls happen in background threads** - don't freeze ROS2 nodes
- ✅ **Future-style interface** - get tokens to wait on when you need results

## 🚀 **Quick Start**

### 1. Start the LLM Service

```bash
# Terminal 1: Start LLM service (handles your llm_api.py)
source /opt/ros/jazzy/setup.bash
ros2 run cognitive_framework llm_service_node

# Terminal 2: Start your cognitive nodes
ros2 run cognitive_framework action_node
```

### 2. Basic Usage in Any Node

```python
from cognitive_framework.llm_client import LLMClient

class YourCognitiveNode(Node):
    def __init__(self):
        super().__init__('your_node')
        
        # Initialize LLM client
        self.llm_client = LLMClient(self)
    
    def some_callback(self, msg):
        # BLOCKING call - wait for result
        response = self.llm_client.generate("What should I do?")
        if response.success:
            print(f"LLM says: {response.text}")
```

## 📋 **Usage Patterns**

### **Pattern 1: Blocking Call (Wait for Result)**

```python
def handle_immediate_decision(self, user_input: str):
    """Use when you need the LLM result immediately."""
    
    response = self.llm_client.generate(
        prompt=f"User says: '{user_input}'. How should I respond?",
        system_prompt="You are a helpful cognitive assistant.",
        max_tokens=100,
        temperature=0.7,
        timeout=10.0
    )
    
    if response.success:
        self.get_logger().info(f"LLM guidance: {response.text}")
        # Use response.text to make immediate decision
        return self.create_action_from_llm(response.text)
    else:
        self.get_logger().error(f"LLM failed: {response.error}")
        return self.fallback_action()
```

### **Pattern 2: Non-Blocking with Future (Check When Ready)**

```python
def start_background_analysis(self, data):
    """Start analysis but continue other work."""
    
    # Start LLM request
    future = self.llm_client.generate_async(
        prompt=f"Analyze this data: {data}",
        max_tokens=150,
        timeout=20.0
    )
    
    # Store future for later
    self.pending_analysis = future
    
    # Continue with other work immediately
    self.continue_other_processing()

def check_analysis_later(self):
    """Check if analysis is ready."""
    
    if hasattr(self, 'pending_analysis') and self.pending_analysis.is_ready():
        response = self.pending_analysis.result()
        if response.success:
            self.get_logger().info(f"Analysis complete: {response.text}")
            self.apply_analysis_results(response.text)
        
        # Clean up
        del self.pending_analysis
```

### **Pattern 3: Fire-and-Forget with Callback**

```python
def start_background_reasoning(self, context):
    """Start reasoning, get callback when done."""
    
    def reasoning_callback(response):
        if response.success:
            self.get_logger().info(f"Background reasoning: {response.text}")
            # Create new goals based on LLM insight
            self.create_strategic_goals(response.text)
        else:
            self.get_logger().warning(f"Reasoning failed: {response.error}")
    
    # Fire and forget - callback will handle result
    self.llm_client.generate_async(
        prompt=f"Strategic analysis of: {context}",
        system_prompt="Provide strategic cognitive insights.",
        callback=reasoning_callback
    )
    
    # Continue immediately, callback will be called when ready
```

### **Pattern 4: Batch Processing with Multiple Futures**

```python
def analyze_multiple_memories(self, memories):
    """Analyze multiple memories concurrently."""
    
    futures = []
    for memory in memories:
        future = self.llm_client.generate_async(
            prompt=f"Analyze memory: {memory['content']}",
            max_tokens=80
        )
        futures.append((memory['id'], future))
    
    # Store for later collection
    self.memory_analysis_futures = futures

def collect_memory_analyses(self):
    """Collect completed analyses."""
    
    if not hasattr(self, 'memory_analysis_futures'):
        return
    
    completed = []
    for memory_id, future in self.memory_analysis_futures:
        if future.is_ready():
            response = future.result()
            if response.success:
                completed.append({
                    'memory_id': memory_id,
                    'analysis': response.text
                })
    
    # Process completed analyses
    if completed:
        self.integrate_memory_analyses(completed)
    
    # Remove completed futures
    self.memory_analysis_futures = [
        (mid, fut) for mid, fut in self.memory_analysis_futures
        if not fut.is_ready()
    ]
```

## 🔧 **Advanced Features**

### **Timeout Handling**

```python
# Set timeout for specific request
response = self.llm_client.generate(
    prompt="Complex analysis...",
    timeout=30.0  # 30 second timeout
)

# Check for timeout
if not response.success and "timeout" in response.error:
    self.get_logger().warning("LLM request timed out, using fallback")
    return self.fallback_logic()
```

### **Request Cancellation**

```python
# Start request
future = self.llm_client.generate_async("Long analysis...")

# Cancel if conditions change
if some_condition_changed:
    cancelled = self.llm_client.cancel_request(future.request_id)
    if cancelled:
        self.get_logger().info("Cancelled LLM request")
```

### **Service Health Monitoring**

```python
def check_llm_availability(self):
    """Check if LLM service is available."""
    
    if self.llm_client.is_service_available():
        self.get_logger().info("✅ LLM service is available")
        return True
    else:
        self.get_logger().warning("⚠️ LLM service not available")
        return False

def get_llm_status(self):
    """Get status of pending requests."""
    
    pending = self.llm_client.get_pending_requests()
    self.get_logger().info(f"📊 {len(pending)} LLM requests pending")
    
    for request_id, status in pending.items():
        elapsed = status['elapsed_time']
        self.get_logger().info(f"  Request {request_id}: {elapsed:.1f}s elapsed")
```

## 🎯 **Real-World Example: Enhanced Action Node**

See `action_node_with_llm.py` for a complete example that demonstrates:

- **Console input analysis** (blocking LLM call for immediate response)
- **Memory analysis** (non-blocking with callback)
- **Strategic reasoning** (fire-and-forget with periodic timing)
- **Goal creation** from LLM insights
- **Fallback handling** when LLM unavailable

## 🚀 **Running with LLM Integration**

### **Update your launch to include LLM service:**

```bash
# Terminal 1: Complete system with LLM
source /opt/ros/jazzy/setup.bash
ros2 run cognitive_framework llm_service_node &
ros2 run cognitive_framework sense_node &
ros2 run cognitive_framework memory_node &
ros2 run cognitive_framework action_node_with_llm &

# Type in sense_node terminal - watch LLM-enhanced responses!
```

### **Expected Output:**

```
# Console input: "What should I focus on?"
[action_node] 🧠 Analyzing console input with LLM: "What should I focus on?"
[llm_service] 📥 Received LLM request req_1: "Human input: 'What should I focus on?'..."
[llm_service] 📤 Completed LLM request req_1 in 1.2s
[action_node] ✅ LLM analysis: Focus on integrating recent sensory data...
[action_node] 🤖 Executed llm_blocking action #15: respond_to_user (priority: 0.90)
```

## 🎉 **Benefits Achieved**

✅ **Your `llm_api.py` works unchanged** - just wrapped in a service  
✅ **Multiple cognitive nodes can use LLM concurrently**  
✅ **Blocking calls don't freeze your ROS2 nodes** - they run in threads  
✅ **Non-blocking patterns available** for background reasoning  
✅ **Future-style tokens** for flexible waiting patterns  
✅ **Automatic error handling** and fallback support  
✅ **Performance monitoring** and request tracking  

Your cognitive framework now has **intelligent LLM-enhanced decision making** while maintaining full concurrency! 🧠🚀 