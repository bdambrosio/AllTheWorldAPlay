#!/usr/bin/env python3
"""
LLM Client

This module provides easy-to-use client classes for accessing the LLM service
from any ROS2 cognitive node. Supports both blocking and non-blocking (Future-based) usage.

Usage Examples:

# Blocking call:
client = LLMClient(node)
response = client.generate("What is consciousness?")

# Non-blocking call:
client = LLMClient(node)
future = client.generate_async("What is consciousness?")
# ... do other work ...
response = future.result()  # Wait when you need the result

# Fire and forget with callback:
def handle_response(response):
    print(f"LLM said: {response}")
    
client.generate_async("Tell me a joke", callback=handle_response)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading
import uuid
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from concurrent.futures import Future
from dataclasses import dataclass


@dataclass 
class LLMResponse:
    """Response from LLM service."""
    text: str
    success: bool
    error: str = ""
    processing_time: float = 0.0
    request_id: str = ""
    timestamp: str = ""


class LLMFuture(Future):
    """
    Extended Future that provides LLM-specific functionality.
    
    This allows non-blocking access to LLM results with additional convenience methods.
    """
    
    def __init__(self, request_id: str, timeout: float = 30.0):
        super().__init__()
        self.request_id = request_id
        self.timeout = timeout
        self.start_time = time.time()
    
    def is_ready(self) -> bool:
        """Check if the response is ready without blocking."""
        return self.done()
    
    def wait_with_timeout(self, timeout: Optional[float] = None) -> Optional[LLMResponse]:
        """
        Wait for result with timeout.
        
        Args:
            timeout: Max time to wait in seconds (uses default if None)
            
        Returns:
            LLMResponse if ready, None if timeout
        """
        try:
            timeout = timeout or self.timeout
            return self.result(timeout=timeout)
        except TimeoutError:
            return None
        except Exception:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the request."""
        return {
            'request_id': self.request_id,
            'is_ready': self.done(),
            'elapsed_time': time.time() - self.start_time,
            'timeout': self.timeout
        }


class LLMClient:
    """
    Client for making LLM requests with both blocking and non-blocking interfaces.
    
    This class handles the ROS2 service communication and provides convenient
    methods for cognitive nodes to interact with the LLM service.
    """
    
    def __init__(self, node: Node, service_timeout: float = 30.0):
        """
        Initialize LLM client.
        
        Args:
            node: The ROS2 node that will use this client
            service_timeout: Default timeout for LLM requests
        """
        self.node = node
        self.service_timeout = service_timeout
        
        # Create request publisher (using topic-based approach for better async support)
        self.request_publisher = node.create_publisher(
            String, 
            '/cognitive/llm_request',
            qos_profile=10
        )
        
        # Subscribe to responses
        self.response_subscriber = node.create_subscription(
            String,
            '/cognitive/llm_response', 
            self._handle_response,
            qos_profile=10
        )
        
        # Track pending requests
        self.pending_requests: Dict[str, LLMFuture] = {}
        self.response_callbacks: Dict[str, Callable] = {}
        
        node.get_logger().info('🤖 LLM Client initialized')
    

    
    def _handle_response(self, msg: String) -> None:
        """Handle incoming LLM responses."""
        try:
            response_data = json.loads(msg.data)
            request_id = response_data.get('request_id', '')
            
            if request_id in self.pending_requests:
                # Create response object
                llm_response = LLMResponse(
                    text=response_data.get('response', ''),
                    success=response_data.get('success', False),
                    error=response_data.get('error', ''),
                    processing_time=response_data.get('processing_time', 0.0),
                    request_id=request_id,
                    timestamp=response_data.get('timestamp', '')
                )
                
                # Complete the future
                future = self.pending_requests[request_id]
                future.set_result(llm_response)
                
                # Call callback if registered
                if request_id in self.response_callbacks:
                    try:
                        self.response_callbacks[request_id](llm_response)
                    except Exception as e:
                        self.node.get_logger().error(f'❌ Error in response callback: {e}')
                    finally:
                        del self.response_callbacks[request_id]
                
                # Clean up
                del self.pending_requests[request_id]
                
                self.node.get_logger().info(f'✅ Received LLM response for {request_id}')
                
        except Exception as e:
            self.node.get_logger().error(f'❌ Error handling LLM response: {e}')
    
    def generate_async(self, 
                      prompt: str, 
                      system_prompt: str = "",
                      bindings: Dict[str, Any] = None,
                      max_tokens: int = 150,
                      temperature: float = 0.7,
                      stops: list = None,
                      callback: Optional[Callable[[LLMResponse], None]] = None,
                      timeout: float = None) -> LLMFuture:
        """
        Generate LLM response asynchronously (non-blocking).
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system/instruction prompt
            bindings: Optional bindings dictionary for LLM variables
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stops: Optional list of stop tokens
            callback: Optional callback function called when response is ready
            timeout: Request timeout (uses default if None)
            
        Returns:
            LLMFuture object that can be used to check status and get result
        """
        request_id = str(uuid.uuid4())
        timeout = timeout or self.service_timeout
        
        # Convert prompt/system_prompt to messages format expected by LLM API
        messages = []
        if system_prompt:
            messages.append(system_prompt)  # System message first
        messages.append(prompt)  # User message
        
        # Create request in format expected by service node
        request_data = {
            'request_id': request_id,
            'messages': messages,
            'bindings': bindings or {},
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stops': stops or ['</end>']
        }
        
        # Create future
        future = LLMFuture(request_id, timeout)
        self.pending_requests[request_id] = future
        
        # Register callback if provided
        if callback:
            self.response_callbacks[request_id] = callback
        
        # Send request
        request_msg = String()
        request_msg.data = json.dumps(request_data)
        
        try:
            # Publish request - this is non-blocking
            self.request_publisher.publish(request_msg)
            
            self.node.get_logger().info(f'📤 Sent async LLM request {request_id}: "{prompt[:50]}..."')
            
        except Exception as e:
            # Handle service call failure
            error_response = LLMResponse(
                text="",
                success=False,
                error=f"Service call failed: {str(e)}",
                request_id=request_id
            )
            future.set_result(error_response)
            del self.pending_requests[request_id]
            
            self.node.get_logger().error(f'❌ Failed to send LLM request: {e}')
        
        return future
    
    def generate(self, 
                prompt: str,
                system_prompt: str = "",
                bindings: Dict[str, Any] = None,
                max_tokens: int = 150,
                temperature: float = 0.7,
                stops: list = None,
                timeout: float = None) -> LLMResponse:
        """
        Generate LLM response synchronously (blocking).
        
        This is a convenience method that calls generate_async() and waits for the result.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system/instruction prompt
            bindings: Optional bindings dictionary for LLM variables
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stops: Optional list of stop tokens
            timeout: Request timeout (uses default if None)
            
        Returns:
            LLMResponse with the generated text
        """
        future = self.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            bindings=bindings,
            max_tokens=max_tokens,
            temperature=temperature,
            stops=stops,
            timeout=timeout
        )
        
        try:
            # Block until result is available
            return future.result(timeout=timeout or self.service_timeout)
        except TimeoutError:
            # Clean up on timeout
            if future.request_id in self.pending_requests:
                del self.pending_requests[future.request_id]
            
            return LLMResponse(
                text="",
                success=False,
                error="Request timed out",
                request_id=future.request_id
            )
    
    def get_pending_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all pending requests."""
        return {
            request_id: future.get_status() 
            for request_id, future in self.pending_requests.items()
        }
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.
        
        Args:
            request_id: ID of the request to cancel
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            cancelled = future.cancel()
            if cancelled:
                del self.pending_requests[request_id]
                if request_id in self.response_callbacks:
                    del self.response_callbacks[request_id]
            return cancelled
        return False
    
    def is_service_available(self) -> bool:
        """Check if the LLM service is currently available."""
        # For topic-based approach, we can check if we have subscribers
        return self.request_publisher.get_subscription_count() > 0


# Convenience functions for quick usage
def create_llm_client(node: Node, **kwargs) -> LLMClient:
    """Create an LLM client for the given node."""
    return LLMClient(node, **kwargs)


def quick_generate(node: Node, prompt: str, **kwargs) -> str:
    """
    Quick one-shot LLM generation (creates client, makes request, returns text).
    
    Useful for simple cases where you don't need to reuse the client.
    Supports all parameters: system_prompt, bindings, max_tokens, temperature, stops, timeout
    """
    client = LLMClient(node)
    response = client.generate(prompt, **kwargs)
    return response.text if response.success else f"Error: {response.error}" 