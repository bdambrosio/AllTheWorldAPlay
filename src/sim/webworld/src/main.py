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
from utils.llm_api import IMAGE_PATH

import asyncio

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

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sessions[session_id] = None
    
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

    async def update_character(name, char_data):
                        # Send character update
                        await websocket.send_text(json.dumps({
                            'type': 'character_update',
                            'name': name,
                            'data': char_data
                        }))
                        # Send show text to middle panel
                        if char_data.get('show'):
                            await websocket.send_text(json.dumps({
                                'type': 'show_update',
                                'text': f"{name}: {char_data['show']}\n"
                            }))
 


    async def run_simulation(sim):
        if sim is None:
            return
        while sim.simulation.running and not sim.simulation.paused:
            await sim.simulation.step(char_update_callback=update_character, world_update_callback=update_world)
            await asyncio.sleep(0.1)
    
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            
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
                        plays_dir = (main_dir / '../../../plays').resolve()
                        
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
                
                elif action == 'load_play':
                    play_name = data.get('play')
                    if sim is not None and sim.simulation:  # If simulation exists
                        await websocket.send_text(json.dumps({
                            'type': 'confirm_reload',
                            'message': 'This will reset the current simulation. Continue?'
                        }))
                    else:  # No existing simulation
                        try:
                            main_dir = Path(__file__).parent
                            play_path = (main_dir / '../../../plays' / play_name).resolve()
                            sessions[session_id] = SimulationWrapper(play_path)
                            sim = sessions[session_id]
                            await websocket.send_text(json.dumps({
                                'type': 'play_loaded',
                                'name': play_name
                            }))
                            await asyncio.sleep(0.1)
                            #image_path = sim.simulation.context.image(filepath='worldsim.png')
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
                            await sim.simulation.step(char_update_callback=update_character, world_update_callback=update_world)
                            await asyncio.sleep(0.1)
                        except Exception as e:
                            print(f"Error in simulation step: {e}")
                            print(f"Traceback:\n{traceback.format_exc()}")
                            await websocket.send_text(json.dumps({
                                'type': 'play_error',
                                'error': f'Failed to load play: {str(e)}'
                            }))
  
                elif action == 'confirm_load_play':
                    play_name = data.get('play')
                    try:
                        main_dir = Path(__file__).parent
                        play_path = (main_dir / '../../../plays' / play_name).resolve()
                        sim = SimulationWrapper(play_path, world_update_callback=update_world)
                        sessions[session_id] = sim
                        await websocket.send_text(json.dumps({
                            'type': 'play_loaded',
                            'name': play_name
                        }))
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
                        await sim.simulation.step(char_update_callback=update_character, world_update_callback=update_world)
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        sim = None
                        await websocket.send_text(json.dumps({
                            'type': 'play_error',
                            'error': f'Failed to load play: {str(e)}'
                        }))
                
                elif action == 'step':
                    # Create callback for character updates
                    if sim is None:
                        return
                  
                    # Pass callback to step
                    await sim.simulation.step(char_update_callback=update_character, world_update_callback=update_world)
                    
                    # Send final status
                    await websocket.send_text(json.dumps({
                        'type': 'status_update',
                        'status': {
                            'running': sim.simulation.running,
                            'paused': sim.simulation.paused
                        }
                    }))
                    await asyncio.sleep(0.1)
                    
                elif action == 'inject':
                    target = data.get('target')
                    text = data.get('text')
                    try:
                        await sim.simulation.inject(target, text, update_callback=update_character)
                    except Exception as e:
                        print(f"Error handling inject: {e}")
                
                # Send updated state
                if sim is not None:
                    state = {
                        'type': 'state_update',
                        'characters': {
                            char.name: char.to_json() for char in sim.simulation.characters
                        },
                        'status': {
                            'running': sim.simulation.running,
                            'paused': sim.simulation.paused
                        }
                    }
                    await websocket.send_text(json.dumps(state))
                    await asyncio.sleep(0.1)
    except WebSocketDisconnect:
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
        print(f"Error in websocket handler: {e}")
        # Clean up on other errors too
        if session_id in sessions:
            del sessions[session_id]
        