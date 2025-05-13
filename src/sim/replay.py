import os, json, math, time, requests, sys
import traceback

import zmq
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
import subprocess
import websockets
import logging
import signal
import copy

# Create replay directory if it doesn't exist
replay_dir = Path.home() / '.local/share' / 'alltheworldaplay' / 'replays'
replay_dir.mkdir(parents=True, exist_ok=True)
replay_file = None
capture_file = None
# Small grey image (1x1 pixel) in base64
MINI_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Add near the top with other globals
replay_events = []
current_replay_index = 0
is_replay_mode = False
is_replay_running = False

# used to control the speed of the replay
EVENT_DELAYS = {
    'replay_event': 0.5,      # UI events
    'show_update': 1.0,       # Text updates
    'world_update': 0.5,      # World state changes
    'character_update': 0.4,  # Character updates
    'command_ack': 0.1,       # Command acknowledgments
    'default': 0.2           # Any other event type
}

# Add a dictionary to store the most recent character details
character_details_cache = {}

# Add this function to store character details as they're encountered
def cache_character_details(event):
    if event.get('type') == 'character_details' and 'name' in event and 'details' in event:
        character_details_cache[event['name']] = event

def cleanup():
    global replay_file
    if replay_file:
        replay_file.close()
        replay_file = None

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
    global replay_events, current_replay_index, is_replay_mode, is_replay_running
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
        return {
            'type': 'heartbeat_response',
            'processing': False,
            'timestamp': time.time()
        }

    try:
        while True:
            data = json.loads(await websocket.receive_text())                
            # Add heartbeat handling
            if data.get('type') == 'heartbeat':
                response = await handle_heartbeat()
                await websocket.send_json(response)
                continue
                
            if data.get('type') == 'command':
                command = data.get('command') or data.get('action', '')
                if command == 'initialize':
                    json_files = [f.name for f in replay_dir.glob('*.json')]
                    await websocket.send_json({
                        "type": "initialize",
                        "plays": json_files
                    })
                    continue

                # Handle load: load selected replay file for playback
                elif command == 'load_play' and data.get('play'):
                    selected_file = data['play']
                    replay_events = load_replay_file(replay_dir / selected_file)
                    current_replay_index = 0
                    is_replay_mode = True
                    await send_command_ack('load_play')
                    continue
                
                elif data.get('action') == 'start_replay':
                    # Load replay file and start replay mode
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
                            # Cache character details for later use
                            cache_character_details(replay_events[current_replay_index])
                            current_replay_index += 1
                            await send_command_ack('step')
                        else:
                            print("No more replay events")
                            await send_command_ack('step')
                            is_replay_mode = False
                    elif data.get('action') == 'run':
                        # Cancel any existing run task
                        global replay_task
                        if 'replay_task' in globals() and replay_task and not replay_task.done():
                            replay_task.cancel()
                        
                        # Define the run loop as a separate function
                        async def run_replay_loop():
                            global current_replay_index, is_replay_running
                            is_replay_running = True
                            try:
                                while current_replay_index < len(replay_events) and is_replay_running:
                                    event = replay_events[current_replay_index]
                                    await websocket.send_json(event)
                                    # Cache character details for later use
                                    cache_character_details(event)
                                    current_replay_index += 1
                                    delay = EVENT_DELAYS.get(event.get('type'), EVENT_DELAYS['default'])
                                    await asyncio.sleep(delay)
                            except asyncio.CancelledError:
                                pass
                            finally:
                                is_replay_running = False
                        
                        # Start the task
                        replay_task = asyncio.create_task(run_replay_loop())
                        await send_command_ack('run')
                    elif data.get('action') == 'pause':
                        # Set flag to stop the run loop
                        global is_replay_running
                        is_replay_running = False
                        # Also cancel the task for immediate effect
                        if 'replay_task' in globals() and replay_task and not replay_task.done():
                            replay_task.cancel()
                            

                    # Next, add a handler for get_character_details commands during replay
           
                    # Handle get_character_details specifically during replay
                    elif command == 'get_character_details' and data.get('name'):
                        character_name = data['name']
                        if character_name in character_details_cache:
                        # Return cached character details
                            await websocket.send_json(character_details_cache[character_name])
                            continue
                        else:
                            # No cached details, send an empty response
                            await websocket.send_json({
                                "type": "character_details",
                                "name": character_name,
                                "details": {}
                            })
                        continue
                 
    except WebSocketDisconnect:
        queue_task.cancel()
        print(f"Client disconnected: {session_id}")
        # Just clean up session
        if session_id in sessions:
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

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
async def shutdown_event():
    cleanup()


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


