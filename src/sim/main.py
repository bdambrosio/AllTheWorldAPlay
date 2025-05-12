import os, json, math, time, requests, sys
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uuid
import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import base64
from io import BytesIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.llm_api import generate_image
from pathlib import Path
from utils.llm_api import LLM
from utils.llm_api import IMAGE_PATH
import asyncio
from contextlib import asynccontextmanager
import zmq
import zmq.asyncio
import subprocess
import websockets
import matplotlib.pyplot as plt
import logging
import signal
import copy
from fastapi.responses import StreamingResponse

# Create replay directory if it doesn't exist
replay_dir = Path.home() / '.local/share' / 'alltheworldaplay' / 'logs'
replay_dir.mkdir(parents=True, exist_ok=True)
replay_file = None
capture_file = None
# Small grey image (1x1 pixel) in base64
MINI_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Add near the top with other globals
replay_events = []
current_replay_index = 0
is_replay_mode = False

# used to control the speed of the replay
EVENT_DELAYS = {
    'replay_event': 0.5,      # UI events
    'show_update': 0.3,       # Text updates
    'world_update': 0.5,      # World state changes
    'character_update': 0.4,  # Character updates
    'command_ack': 0.1,       # Command acknowledgments
    'default': 0.2           # Any other event type
}

def cleanup():
    global replay_file, capture_file
    if replay_file:
        replay_file.close()
        replay_file = None
    if capture_file:
        capture_file.close()
        capture_file = None

def signal_handler(signum, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions - but code only handles one session right now
sessions: Dict[str, int] = {}

class WebSocketManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._message_queue = asyncio.Queue()
        
    @asynccontextmanager
    async def send_lock(self):
        try:
            await self._lock.acquire()
            yield
        finally:
            self._lock.release()
            
    async def send_message(self, websocket, message):
        async with self.send_lock():
            await websocket.send_json(message)
            # Small delay to ensure message processing
            await asyncio.sleep(0.05)
            
    async def process_queue(self, websocket):
        while True:
            message = await self._message_queue.get()
            await self.send_message(websocket, message)
            self._message_queue.task_done()
            
    def queue_message(self, message):
        self._message_queue.put_nowait(message)

class SimulationManager:
    def __init__(self):
        self.context = zmq.asyncio.Context()
        self.command_socket = self.context.socket(zmq.PUSH)
        self.result_socket = self.context.socket(zmq.PULL)
        self.process = None
        
    async def start(self):
        # Setup ZMQ ports
        self.command_socket.bind("tcp://127.0.0.1:5555")
        self.result_socket.bind("tcp://127.0.0.1:5556")
        
        # Start simulation process
        sim_path = Path(__file__).parent / "simulation.py"
        self.process = subprocess.Popen(["python", "-Xfrozen_modules=off", str(sim_path)])
        
        # Wait briefly for process to start
        await asyncio.sleep(10)
        
    async def stop(self):
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
        self.command_socket.close()
        self.result_socket.close()
        self.context.term()
        
    async def send_command(self, command):
        await self.command_socket.send_json(command)
        
    async def receive_results(self):
        while True:
            try:
                result = await self.result_socket.recv_json()
                yield result
            except Exception as e:
                print(f"Error receiving results: {e}")
                break


sim_manager = SimulationManager()
ws_manager = WebSocketManager()
# Add a root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok"}

# Add session creation endpoint
@app.get("/api/session/new")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = None
    return {"session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sessions[session_id] = None
    
    # Start the message queue processing task
    queue_task = asyncio.create_task(ws_manager.process_queue(websocket))
    
    async def send_command_ack(command: str):
        """Send command completion acknowledgement"""
        await websocket.send_json({
            "type": "command_ack",
            "command": command
        })

    async def handle_heartbeat():
        if session_id in sessions and sessions[session_id]:
            sim = sessions[session_id]
        return {
            'type': 'heartbeat_response',
            'processing': False,
            'timestamp': time.time()
        }

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            
            # Add replay event handling
            if data.get('type') == 'replay_event':
                # Write to replay file
                capture_file.write(json.dumps({
                    'type': 'replay_event',
                    'action': data['action'],
                    'arg': data['arg'],
                    'timestamp': data['timestamp']
                }, indent=2) + "\n")
                continue
                
            # Add heartbeat handling
            if data.get('type') == 'heartbeat':
                response = await handle_heartbeat()
                await websocket.send_json(response)
                continue
                
            if data.get('type') == 'command':
                if data.get('action') == 'start_replay':
                    # Load replay file and start replay mode
                    global replay_events, current_replay_index, is_replay_mode
                    replay_events = load_replay_file(replay_dir / 'replay.json')
                    current_replay_index = 0
                    is_replay_mode = True
                    await send_command_ack('start_replay')
                elif is_replay_mode:
                    if data.get('action') == 'step':
                        # Step: send next event
                        if current_replay_index < len(replay_events):
                            print(f"Sending step event: {replay_events[current_replay_index]}")
                            await websocket.send_json(replay_events[current_replay_index])
                            current_replay_index += 1
                            await send_command_ack('step')
                        else:
                            print("No more replay events")
                            await send_command_ack('step')
                            is_replay_mode = False
                    elif data.get('action') == 'run':
                        # Run: send events with delays until paused or end
                        while current_replay_index < len(replay_events):
                            event = replay_events[current_replay_index]
                            await websocket.send_json(event)
                            current_replay_index += 1
                            # Get delay for this event type, or use default
                            delay = EVENT_DELAYS.get(event.get('type'), EVENT_DELAYS['default'])
                            await asyncio.sleep(delay)
                    elif data.get('action') == 'pause':
                        # Pause: just acknowledge
                        await send_command_ack('pause')
                else:
                    await sim_manager.send_command(data)
                
            elif data['action'] == 'load_known_actors':
                await sim_manager.send_command(data)
                
            elif data['action'] == 'save_known_actors':
                await sim_manager.send_command(data)
                
    except WebSocketDisconnect:
        queue_task.cancel()
        print(f"Client disconnected: {session_id}")
        # Stop simulation if running
        if session_id in sessions:
            sim = sessions[session_id]
            if sim and sim.simulation:
                sim.simulation.running = False
                sim.simulation.paused = True
            # Clean up session
            del sessions[session_id]
            
    except Exception as e:
        queue_task.cancel()
        print(f"Error in websocket handler: {e}")
        # Clean up on other errors too
        if session_id in sessions:
            del sessions[session_id]


def process_ws_message(message):
    # Convert websocket message to simulation command format
    try:
        data = json.loads(message)
        # Assuming the websocket message already has the correct format
        # Adjust this conversion based on your specific message formats
        return data
    except json.JSONDecodeError:
        return {
            'type': 'error',
            'error': 'Invalid JSON message'
        }

async def handle_websocket(websocket, sim_manager):  # Add sim_manager parameter
    async for message in websocket:
        command = process_ws_message(message)
        await sim_manager.send_command(command)

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    global sim_manager, ws_manager, capture_file
    await sim_manager.start()
    
    # Open replay_capture file
    capture_file = open(replay_dir / 'record.json', 'w')
    
    async def handle_simulation_results():
        async for result in sim_manager.receive_results():
            ws_message = convert_sim_to_ws_message(result)
            ws_manager.queue_message(ws_message)
                        # Minimize images in the copy
            capture_file_result = copy.deepcopy(ws_message)
            if capture_file_result.get('type') == 'character_update' and 'image' in capture_file_result.get('data', {}):
                capture_file_result['data']['image'] = MINI_IMAGE
            elif capture_file_result.get('type') == 'world_update' and 'image' in capture_file_result.get('data', {}):
                capture_file_result['data']['image'] = MINI_IMAGE

            # Record the ws_message instead of the raw result
            capture_file.write(json.dumps(capture_file_result, indent=2)+"\n")
            await asyncio.sleep(0.1)
    
    app.state.result_task = asyncio.create_task(handle_simulation_results())

@app.on_event("shutdown")
async def shutdown_event():
    global capture_file
    if hasattr(app.state, 'result_task'):
        app.state.result_task.cancel()
    cleanup()
    await sim_manager.stop()

def convert_sim_to_ws_message(sim_result):
    """Convert simulation message format to websocket format"""
    print(f"Converting message: {sim_result.get('type')}")  # Debug logging
    
    # Handle show_update messages from the simulation
    if sim_result.get('type') == 'show_update':
        if sim_result['message']['name'] is None or sim_result['message']['name'] == '':
            return {
                'type': 'show_update',
                'text': sim_result['message']['text']
            }
        else:
            return {
                'type': 'show_update',
                'text': f"{sim_result['message']['name']}: {sim_result['message']['text']}"
            }
    # Handle world_update messages
    elif sim_result.get('type') == 'world_update':
        return {
            'type': 'world_update',
            'name': 'World',
            'data': sim_result['world']  # Unpack the nested world data
        }
    # Handle character_update messages
    elif sim_result.get('type') == 'character_update':
        return {
            'type': 'character_update',
            'name': sim_result['character']['name'],
            'data': sim_result['character']  # The full character data including image if present
        }
    # Handle character details messages
    elif sim_result.get('type') == 'character_details':
        return {
            'type': 'character_details',
            'name': sim_result['name'],
            'details': sim_result.get('details', {})  # Use get() with default empty dict
        }
    # Handle error messages
    elif sim_result.get('type') == 'error':
        return {
            'type': 'error',
            'error': sim_result.get('error', 'Unknown error')
        }
    # Pass through other message types as-is
    return sim_result

def load_replay_file(file_path: Path) -> list:
    events = []
    current_event = []
    brace_count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                current_event.append(line)
                # Count opening and closing braces
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and current_event:  # We've found a complete JSON object
                    try:
                        event = json.loads(''.join(current_event))
                        events.append(event)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(f"Problematic JSON: {''.join(current_event)}")
                    current_event = []
    return events


