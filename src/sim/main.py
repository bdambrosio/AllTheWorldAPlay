import os, json, math, time, requests, sys
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uuid
import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from sim.simulation import SimulationWrapper
import base64
from io import BytesIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.llm_api import generate_image
from pathlib import Path
from utils.llm_api import LLM
from utils.llm_api import IMAGE_PATH
import asyncio
from contextlib import asynccontextmanager

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions: Dict[str, SimulationWrapper] = {}

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

def generate_and_encode_image(char):
    """Generate and base64 encode an image for a character"""
    try:
        description = char.generate_image_description()
        if description:
            image_path = generate_image(description)
            if image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    print(f"Image size: {len(image_data)} bytes")  # Debug log
                    return base64.b64encode(image_data).decode()
    except Exception as e:
        print(f"Error generating image for {char.name}: {e}")
    return None

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

async def process_messages(websocket, sim):
    while True:
        try:
            if not sim.simulation.context.message_queue.empty():
                message = sim.simulation.context.message_queue.get_nowait()
                await websocket.send_json({
                    "type": "show_update",
                    "text": f"{message['name']}: {message['text']}"  # Format as string
                })
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error processing messages: {e}")
            break

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sessions[session_id] = None
    message_task = None
    
    async def send_command_ack(command: str):
        """Send command completion acknowledgement"""
        await websocket.send_json({
            "type": "command_ack",
            "command": command
        })

    async def update_world(name, world_data):
            # Send character update
            context_data = sim.simulation.context.to_json()
            image_path = context_data['image']
            if image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    context_data['image'] = base64.b64encode(image_data).decode()
            await websocket.send_text(json.dumps({
                'type': 'world_update',
                'name': 'World',
                'data': context_data
            }))
            await websocket.send_text(json.dumps({
                'type': 'context_update',
                'text': context_data['show']
            }))


    async def update_character(name, char_data):
        # Send character update only
        await websocket.send_text(json.dumps({
            'type': 'character_update',
            'name': name,
            'data': char_data
        }))

    async def log_callback(message):
        await websocket.send_text(json.dumps({
            'type': 'show_update',
            'text': message
        }))

    async def run_simulation(sim):
        if sim is None:
            return
        print(f"Starting simulation run with context id: {id(sim.simulation.context)}")  # Debug
        try:
            while sim.simulation.running and not sim.simulation.paused:
                await sim.simulation.step(
                    char_update_callback=update_character,
                    world_update_callback=update_world
                )
                await asyncio.sleep(0.1)
        finally:
            print("Cancelling message task")  # Debug
            if message_task:
                message_task.cancel()
    
    async def load_and_start_play(websocket, play_path):
        nonlocal message_task  # Access outer message_task
        sessions[session_id] = SimulationWrapper(play_path)
        sim = sessions[session_id]
        
        # Create message task after sim exists
        message_task = asyncio.create_task(
            process_messages(websocket, sim)
        )
        
        await websocket.send_text(json.dumps({
            'type': 'play_loaded',
            'name': play_name
        }))
        
        # Rest of initialization...
        context_data = sim.simulation.context.to_json()
        image_path = context_data['image']
        if image_path:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                context_data['image'] = base64.b64encode(image_data).decode()
        await websocket.send_text(json.dumps({
            'type': 'world_update',
            'name': 'World',
            'data': context_data
        }))
        await asyncio.sleep(0.1)
        await sim.simulation.update_actors(char_update_callback=update_character, world_update_callback=update_world, log_callback=log_callback)
        await asyncio.sleep(0.1)
        await send_command_ack('load_play')

    # Add heartbeat handler but keep all existing code
    async def handle_heartbeat():
        if session_id in sessions and sessions[session_id]:
            sim = sessions[session_id]
            return await sim.simulation.handle_heartbeat()
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
                action = data.get('action')
                sim = sessions[session_id]
                
                if action == 'run' and sim is not None:
                    sim.simulation.running = True
                    sim.simulation.paused = False
                    asyncio.create_task(run_simulation(sim))
                
                elif action == 'pause' and sim is not None:
                    sim.simulation.paused = True
                    sim.simulation.running = False
                
                elif action == 'initialize':
                    try:
                        # Get path to plays directory relative to main.py
                        main_dir = Path(__file__).parent
                        plays_dir = (main_dir / '../plays').resolve()
                        
                        # List .py files, excluding system files
                        play_files = [f.name for f in plays_dir.glob('*.py') 
                                    if f.is_file() and not f.name.startswith('__')]
                        
                        await websocket.send_text(json.dumps({
                            'type': 'play_list',
                            'plays': play_files
                        }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            'type': 'play_error',
                            'error': f'Failed to list plays: {str(e)}'
                        }))
                    await send_command_ack('initialize')    
                
                elif action == 'load_play':
                    play_name = data.get('play')
                    if sim is not None and sim.simulation:
                        await websocket.send_text(json.dumps({
                            'type': 'confirm_reload',
                            'message': 'This will reset the current simulation. Continue?'
                        }))
                    else:
                        try:
                            main_dir = Path(__file__).parent
                            play_path = (main_dir / '../plays' / play_name).resolve()
                            await load_and_start_play(websocket, play_path)
                            # Add completion message
                            await websocket.send_text(json.dumps({
                            'type': 'command_complete',
                                'command': 'load_play'
                            }))
                        except Exception as e:
                            traceback.print_exc()
                            await websocket.send_text(json.dumps({
                                'type': 'play_error',
                                'error': f'Failed to load play: {str(e)}'
                            }))
                        await send_command_ack('load_play')
                
                elif action == 'confirm_load_play':
                    play_name = data.get('play')
                    try:
                        main_dir = Path(__file__).parent
                        play_path = (main_dir / '../plays' / play_name).resolve()
                        await load_and_start_play(websocket, play_path)
                        # Add completion message here too
                        await websocket.send_text(json.dumps({
                            'type': 'command_complete',
                            'command': 'load_play'
                        }))
                    except Exception as e:
                        sim = None
                        await websocket.send_text(json.dumps({
                            'type': 'play_error',
                            'error': f'Failed to load play: {str(e)}'
                        }))
                    await send_command_ack('load_play')
                elif action == 'step':
                    if sim is None:
                        return
                    
                    # Pass callback to step
                    await sim.simulation.step(char_update_callback=update_character, world_update_callback=update_world)
                    
                    # Send completion status
                    await websocket.send_text(json.dumps({
                        'type': 'command_complete',
                        'command': 'step'
                    }))
                    
                    await websocket.send_text(json.dumps({
                        'type': 'status_update',
                        'status': {
                            'running': sim.simulation.running,
                            'paused': sim.simulation.paused
                        }
                    }))
                    await send_command_ack('step')
                elif action == 'inject':
                    target_name = data.get('target')
                    text = data.get('text')
                    try:
                        if target_name == 'World':
                            # Stub for now
                            return
                        # Format message as "character-name, message" like worldsim
                        formatted_message = f"{target_name}, {text}"
                        await sim.simulation.inject(target_name, text, update_callback=update_character)
                        await websocket.send_text(json.dumps({
                            'type': 'command_complete',
                            'command': 'inject'
                        }))
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"Error handling inject: {e}")
                    await send_command_ack('inject')
                # Send updated state
                if sim is not None:
                    char_states = sim.simulation.get_character_states()
                    await websocket.send_json({
                        "type": "character_update",
                        "data": char_states
                    })
                    await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        if message_task:
            message_task.cancel()
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
        if message_task:
            message_task.cancel()
        print(f"Error in websocket handler: {e}")
        # Clean up on other errors too
        if session_id in sessions:
            del sessions[session_id]
                
@app.get("/api/character/{name}/details")
async def get_character_details(name: str, session_id: str):
    """Get detailed character state for explorer"""
    try:
        if not session_id:
            raise HTTPException(
                status_code=400, 
                detail="Missing session ID"
            )
            
        if session_id not in sessions:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {session_id} not found"
            )
            
        sim = sessions[session_id]
        if not sim:
            raise HTTPException(
                status_code=404, 
                detail="No active simulation"
            )
            
        details = sim.simulation.get_character_details(name)
        if not details:
            raise HTTPException(
                status_code=404,
                detail=f"Character {name} not found"
            )
            
        return details
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting character details: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
                