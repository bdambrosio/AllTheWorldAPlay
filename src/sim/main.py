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

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
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
            
            # Add heartbeat handling
            if data.get('type') == 'heartbeat':
                response = await handle_heartbeat()
                await websocket.send_json(response)
                continue
                
            if data.get('type') == 'command':
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
    global sim_manager, ws_manager
    await sim_manager.start()
    
    async def handle_simulation_results():
        async for result in sim_manager.receive_results():
            ws_message = convert_sim_to_ws_message(result)
            #print(f"Sending message to websocket: {ws_message}")
            ws_manager.queue_message(ws_message)
            await asyncio.sleep(0.1)
    
    app.state.result_task = asyncio.create_task(handle_simulation_results())

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'result_task'):
        app.state.result_task.cancel()
    await sim_manager.stop()

def convert_sim_to_ws_message(sim_result):
    """Convert simulation message format to websocket format"""
    print(f"Converting message: {sim_result.get('type')}")  # Debug logging
    
    # Handle show_update messages from the simulation
    if sim_result.get('type') == 'show_update':
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

               