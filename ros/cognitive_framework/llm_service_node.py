#!/home/bruce/Downloads/AllTheWorldAPlay/src/owl/bin/python3
"""
LLM Service Node

This node provides a ROS2 service interface to the LLM API, allowing multiple
cognitive nodes to make concurrent LLM calls without blocking each other.

The service handles requests in a thread pool, enabling true concurrency.
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import sys
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime

# Add the parent directory to sys.path to import llm_api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.utils.llm_api import LLM
    from src.utils.Messages import SystemMessage, UserMessage, AssistantMessage
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LLM API not available: {e}")
    print("   LLM service will return mock responses")
    LLM_AVAILABLE = False


# Custom service definition using String messages (avoiding custom interfaces for simplicity)
class LLMRequest:
    """Simple LLM request structure."""
    def __init__(self, messages, bindings, max_tokens: int = 150, temperature: float = 0.7, stops: list = ['</end>']):
        self.bindings = bindings
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stops = stops
        self.timestamp = datetime.now().isoformat()


class LLMResponse:
    """Simple LLM response structure."""
    def __init__(self, response: str, success: bool = True, error: str = "", request_id: str = ""):
        self.response = response
        self.success = success
        self.error = error
        self.request_id = request_id
        self.timestamp = datetime.now().isoformat()


class LLMServiceNode(Node):
    """
    ROS2 service node that provides LLM API access to other nodes.
    
    Features:
    - Concurrent request handling using thread pool
    - Request queuing and prioritization
    - Error handling and fallback responses
    - Performance monitoring
    """
    
    def __init__(self):
        super().__init__('llm_service_node')
        
        # Use reentrant callback group for concurrent processing
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscriber for LLM requests
        self.request_subscriber = self.create_subscription(
            String,
            '/cognitive/llm_request',
            self.handle_llm_request,
            qos_profile=10,
            callback_group=self.callback_group
        )
        
        # Publisher for LLM responses (request_id -> response mapping)
        self.response_publisher = self.create_publisher(
            String,
            '/cognitive/llm_response',
            qos_profile=10
        )
        
        # Thread pool for concurrent LLM processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='llm_worker')
        
        # Initialize LLM if available
        self.llm = None
        if LLM_AVAILABLE:
            try:
                # Try to initialize LLM - check what parameters your LLM class accepts
                import inspect
                llm_init_signature = inspect.signature(LLM.__init__)
                llm_params = list(llm_init_signature.parameters.keys())
                
                self.get_logger().info(f'LLM __init__ accepts parameters: {llm_params}')
                
                # Initialize LLM based on available parameters
                if 'server_name' in llm_params:
                    self.llm = LLM(server_name='vllm')
                    self.get_logger().info('✅ LLM API initialized with vllm server')
                elif 'server' in llm_params:
                    self.llm = LLM(server='vllm')
                    self.get_logger().info('✅ LLM API initialized with vllm server (legacy parameter)')
                else:
                    self.llm = LLM()
                    self.get_logger().info('✅ LLM API initialized with default parameters')
                    
            except Exception as e:
                self.get_logger().error(f'❌ Failed to initialize LLM: {e}')
                self.llm = None
        
        # Request tracking
        self.request_counter = 0
        self.active_requests = {}
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        # Status publisher for monitoring
        self.status_timer = self.create_timer(10.0, self.publish_status)
        
        self.get_logger().info('🤖 LLM Service Node initialized - ready for concurrent requests')
    
    def handle_llm_request(self, msg):
        """
        Handle incoming LLM requests by submitting them to the thread pool.
        
        This method returns immediately, while the actual LLM call happens asynchronously.
        Response is published to the response topic when ready.
        """
        try:
            # Parse request JSON
            request_data = json.loads(msg.data)
            request_id = request_data.get('request_id', f'req_{self.request_counter}')
            self.request_counter += 1
            
            # Create LLM request object
            llm_request = LLMRequest(
                messages=request_data.get('messages', []),
                bindings=request_data.get('bindings', {}),
                max_tokens=request_data.get('max_tokens', 150),
                temperature=request_data.get('temperature', 0.7),
                stops=request_data.get('stops', ['</end>'])
            )
            
            # Log the first message as a preview
            preview_text = str(llm_request.messages[0])[:50] if llm_request.messages else "empty"
            self.get_logger().info(f'📥 Received LLM request {request_id}: "{preview_text}..."')
            
            # Submit to thread pool for async processing
            future = self.thread_pool.submit(self._process_llm_request, request_id, llm_request)
            self.active_requests[request_id] = {
                'future': future,
                'start_time': time.time(),
                'request': llm_request
            }
            
            self.request_stats['total_requests'] += 1
            
        except Exception as e:
            error_msg = f'Error handling LLM request: {str(e)}'
            self.get_logger().error(f'❌ {error_msg}')
            
            # Publish error response
            error_response = {
                'request_id': request_data.get('request_id', 'unknown'),
                'response': '',
                'success': False,
                'error': error_msg,
                'processing_time': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            error_msg_obj = String()
            error_msg_obj.data = json.dumps(error_response)
            self.response_publisher.publish(error_msg_obj)
    
    def _process_llm_request(self, request_id: str, llm_request: LLMRequest) -> None:
        """
        Process an LLM request in a background thread.
        
        This method does the actual blocking LLM call and publishes the response.
        """
        start_time = time.time()
        
        try:
            if self.llm is not None and LLM_AVAILABLE:
                # Real LLM call
                messages = []
                for n, message in enumerate(llm_request.messages):
                    if n == 0:
                        messages.append(SystemMessage(content=message))
                    else:
                        messages.append(UserMessage(content=message))
                
                # This is the blocking call, but it's in a separate thread
                response_text = self.llm.ask(llm_request.bindings, messages, temp=llm_request.temperature, max_tokens=llm_request.max_tokens, stops=llm_request.stops)
                
                llm_response = LLMResponse(
                    response=response_text,
                    success=True,
                    request_id=request_id
                )
                
                self.request_stats['successful_requests'] += 1
                
            else:
                # Mock response when LLM not available
                mock_input = str(llm_request.messages[-1]) if llm_request.messages else "empty input"
                response_text = f"[MOCK] Cognitive response to: {mock_input}"
                llm_response = LLMResponse(
                    response=response_text,
                    success=True,
                    request_id=request_id
                )
                
                # Simulate processing time
                time.sleep(0.5)
                self.request_stats['successful_requests'] += 1
            
        except Exception as e:
            error_msg = f'LLM processing error: {str(e)}'
            self.get_logger().error(f'❌ {error_msg}')
            
            llm_response = LLMResponse(
                response="",
                success=False,
                error=error_msg,
                request_id=request_id
            )
            
            self.request_stats['failed_requests'] += 1
        
        # Calculate timing
        processing_time = time.time() - start_time
        self._update_avg_response_time(processing_time)
        
        # Publish response
        response_msg = String()
        response_msg.data = json.dumps({
            'request_id': request_id,
            'response': llm_response.response,
            'success': llm_response.success,
            'error': llm_response.error,
            'processing_time': processing_time,
            'timestamp': llm_response.timestamp
        })
        
        self.response_publisher.publish(response_msg)
        
        # Clean up tracking
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        self.get_logger().info(
            f'📤 Completed LLM request {request_id} in {processing_time:.2f}s'
        )
    
    def _update_avg_response_time(self, processing_time: float):
        """Update rolling average response time."""
        total = self.request_stats['total_requests']
        if total > 0:
            current_avg = self.request_stats['avg_response_time']
            # Weighted average favoring recent requests
            weight = min(0.1, 1.0 / total)
            self.request_stats['avg_response_time'] = (
                current_avg * (1 - weight) + processing_time * weight
            )
    
    def publish_status(self):
        """Publish service status for monitoring."""
        status = {
            'service': 'llm_service',
            'timestamp': datetime.now().isoformat(),
            'active_requests': len(self.active_requests),
            'stats': self.request_stats,
            'llm_available': LLM_AVAILABLE and self.llm is not None
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        # Note: We'd need a status topic, for now just log periodically
        
        if len(self.active_requests) > 0:
            self.get_logger().info(
                f'📊 LLM Service Status: {len(self.active_requests)} active, '
                f'{self.request_stats["total_requests"]} total, '
                f'{self.request_stats["avg_response_time"]:.2f}s avg'
            )
    
    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('🛑 Shutting down LLM service...')
        
        # Wait for active requests to complete (with timeout)
        if self.active_requests:
            self.get_logger().info(f'Waiting for {len(self.active_requests)} active requests...')
            for request_id, request_info in self.active_requests.items():
                try:
                    request_info['future'].result(timeout=5.0)
                except Exception:
                    pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=10.0)
        
        super().destroy_node()


def main(args=None):
    """Main entry point for the LLM service node."""
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor for concurrent request handling
    executor = MultiThreadedExecutor(num_threads=6)
    
    llm_service_node = LLMServiceNode()
    
    try:
        rclpy.spin(llm_service_node, executor=executor)
    except KeyboardInterrupt:
        llm_service_node.get_logger().info('LLM Service Node shutting down...')
    finally:
        llm_service_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 