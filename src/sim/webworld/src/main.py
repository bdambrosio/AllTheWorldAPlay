import os, json, math, time, requests, sys
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
    sessions[session_id] = SimulationWrapper()
    return {"session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await websocket.accept()
        
        if session_id not in sessions:
            await websocket.close(code=1008)  # Policy violation
            return
            
        sim = sessions[session_id]
        
        while True:
            try:
                data = json.loads(await websocket.receive_text())
                
                if data.get('type') == 'command':
                    action = data.get('action')
                    
                    if action == 'step':
                        # Create callback for character updates
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
                        
                        # Pass callback to step
                        await sim.simulation.step(update_character)
                        
                        # Send final status
                        await websocket.send_text(json.dumps({
                            'type': 'status_update',
                            'status': {
                                'running': sim.simulation.running,
                                'paused': sim.simulation.paused
                            }
                        }))
                        
                    elif action == 'run':
                        sim.simulation.run()
                    elif action == 'pause':
                        sim.simulation.pause()
                    elif action == 'inject':
                        sim.simulation.inject(data.get('text', ''))
                    
                    # Send updated state
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
                    
            except Exception as e:
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': f"Command error: {str(e)}"
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': "Invalid message format"
                }))
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011)  # Internal error
        except:
            pass
        