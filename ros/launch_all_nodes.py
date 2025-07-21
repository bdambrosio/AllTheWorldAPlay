#!/usr/bin/env python3
"""
Launch All Cognitive Nodes

This script starts all three cognitive framework nodes in separate processes,
allowing them to run concurrently while being managed from a single VSCode debug session.
"""

import rclpy
import subprocess
import threading
import time
import signal
import sys
import os
from pathlib import Path


class CognitiveNodeLauncher:
    """Manages launching and monitoring multiple ROS2 nodes."""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down nodes...")
        self.running = False
        self._stop_all_nodes()
        sys.exit(0)
    
    def _start_node(self, node_name, script_path):
        """Start a single ROS2 node in a subprocess."""
        try:
            # Get the current environment and ensure ROS2 is sourced
            env = os.environ.copy()
            
            # Command to run the node
            cmd = [sys.executable, script_path]
            
            print(f"🚀 Starting {node_name}...")
            
            # For SENSE node, allow direct console interaction
            # For other nodes, capture output for monitoring
            if node_name == "SENSE":
                # Let sense node use the main terminal for input
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=None,  # Use main terminal
                    stderr=None,  # Use main terminal  
                    stdin=None    # Use main terminal
                )
            else:
                # Capture output for memory and action nodes
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Start a thread to monitor this process output
                monitor_thread = threading.Thread(
                    target=self._monitor_process_output,
                    args=(node_name, process),
                    daemon=True
                )
                monitor_thread.start()
            
            self.processes.append((node_name, process))
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {node_name}: {e}")
            return None
    
    def _monitor_process_output(self, node_name, process):
        """Monitor and display output from a node process."""
        try:
            for line in iter(process.stdout.readline, ''):
                if not self.running:
                    break
                # Prefix each line with the node name for clarity
                print(f"[{node_name}] {line.strip()}")
        except Exception as e:
            if self.running:  # Only show error if we're still supposed to be running
                print(f"❌ Error monitoring {node_name}: {e}")
    
    def _stop_all_nodes(self):
        """Stop all running node processes."""
        print("🛑 Stopping all nodes...")
        
        for node_name, process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    print(f"   Stopping {node_name}...")
                    process.terminate()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print(f"   Force killing {node_name}...")
                        process.kill()
                        process.wait()
                        
            except Exception as e:
                print(f"❌ Error stopping {node_name}: {e}")
        
        self.processes.clear()
        print("✅ All nodes stopped")
    
    def launch_all_nodes(self):
        """Launch all three cognitive framework nodes."""
        
        # Check if ROS2 environment is available
        if 'ROS_DISTRO' not in os.environ:
            print("❌ ROS2 environment not sourced!")
            print("Please run: source /opt/ros/jazzy/setup.bash")
            return False
        
        print("🧠 Starting Cognitive Framework - All Nodes")
        print("=" * 60)
        
        # Get paths to the node scripts
        base_path = Path(__file__).parent / "cognitive_framework"
        
        nodes = [
            ("SENSE", base_path / "sense_node.py"),
            ("MEMORY", base_path / "memory_node.py"), 
            ("ACTION", base_path / "action_node.py"),
            ("LLM_SERVICE", base_path / "llm_service_node.py")
        ]
        
        # Start all nodes
        for node_name, script_path in nodes:
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                return False
                
            process = self._start_node(node_name, str(script_path))
            if process is None:
                print(f"❌ Failed to start {node_name} node")
                self._stop_all_nodes()
                return False
                
            # Small delay between starting nodes
            time.sleep(0.5)
        
        print("=" * 60)
        print("🎯 All nodes started successfully!")
        print()
        print("💬 CONSOLE INPUT READY:")
        print("   • SENSE node is ready for your input in THIS terminal")
        print("   • Just type messages and press Enter")
        print("   • Your input will flow through: SENSE → MEMORY → ACTION")
        print()
        print("📊 MONITORING:")
        print("   • [MEMORY] and [ACTION] logs will appear above")
        print("   • [SENSE] logs appear mixed with your input")
        print()
        print("🛑 SHUTDOWN: Press Ctrl+C to stop all nodes")
        print("=" * 60)
        print()
        print("🎤 Ready for console input (type below):")
        
        return True
    
    def monitor_nodes(self):
        """Monitor all nodes and handle user input."""
        try:
            # Main monitoring loop
            while self.running:
                # Check if any processes have died
                for node_name, process in self.processes[:]:  # Copy list to avoid modification during iteration
                    if process.poll() is not None:  # Process has terminated
                        print(f"⚠️  {node_name} node has stopped (exit code: {process.returncode})")
                        self.processes.remove((node_name, process))
                
                # If all processes are dead, exit
                if not self.processes:
                    print("❌ All nodes have stopped")
                    break
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
        finally:
            self.running = False
            self._stop_all_nodes()


def main():
    """Main entry point."""
    
    print("🧠 Cognitive Framework - Multi-Node Launcher")
    print()
    
    # Initialize ROS2
    try:
        rclpy.init()
    except Exception as e:
        print(f"❌ Failed to initialize ROS2: {e}")
        print("Make sure ROS2 is installed and environment is sourced:")
        print("  source /opt/ros/jazzy/setup.bash")
        return 1
    
    try:
        # Create launcher and start nodes
        launcher = CognitiveNodeLauncher()
        
        if not launcher.launch_all_nodes():
            return 1
        
        # Monitor nodes and handle input
        launcher.monitor_nodes()
        
    finally:
        # Clean shutdown
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
    
    print("✅ Cognitive Framework shutdown complete")
    return 0


if __name__ == '__main__':
    sys.exit(main()) 