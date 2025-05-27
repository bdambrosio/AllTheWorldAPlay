import os, json, time, requests, sys
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import base64
from io import BytesIO
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import signal

# Create replay directory if it doesn't exist
main_dir = Path(__file__).parent
replays_dir = (main_dir / '../plays/replays/').resolve()
replays_dir.mkdir(parents=True, exist_ok=True)
replay_file = None
# Small grey image (1x1 pixel) in base64
MINI_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Store active sessions - but code only handles one session right now
sessions: Dict[str, int] = {}
replay_events:Dict[str, list] = {}
current_replay_index:Dict[str, int] = {}
is_replay_mode:Dict[str, bool] = {}
is_replay_running:Dict[str, bool] = {}
speech_enabled:Dict[str, bool] = {}
character_details_cache:Dict[str, dict] = {}
image_cache:Dict[str, dict] = {}
speech_complete_event:Dict[str, asyncio.Event] = {}

# Add to globals
replay_task: Dict[str, asyncio.Task] = {}

def init_session_state(session_id):
    sessions[session_id] = session_id
    replay_events[session_id] = []
    current_replay_index[session_id] = 0
    is_replay_mode[session_id] = False
    is_replay_running[session_id] = False
    speech_enabled[session_id] = True
    character_details_cache[session_id] = {}
    image_cache[session_id] = {}
    speech_complete_event[session_id] = asyncio.Event()
    replay_task[session_id] = None

def release_session_state(session_id):
    if session_id in replay_events:
        del replay_events[session_id]
    if session_id in current_replay_index:
        del current_replay_index[session_id]
    if session_id in is_replay_mode:
        del is_replay_mode[session_id]
    if session_id in is_replay_running:
        del is_replay_running[session_id]
    if session_id in speech_enabled:
        del speech_enabled[session_id]
    if session_id in character_details_cache:
        del character_details_cache[session_id]
    if session_id in image_cache:
        del image_cache[session_id]
    if session_id in speech_complete_event:
        del speech_complete_event[session_id]
    if session_id in replay_task:
        del replay_task[session_id]

    if session_id in character_details_cache:
        del character_details_cache[session_id]
    if session_id in image_cache:
        del image_cache[session_id]
    if session_id in speech_complete_event:
        del speech_complete_event[session_id]
    if session_id in replay_task:
        task = replay_task[session_id]
        if task and not task.done():
            task.cancel()
            del replay_task[session_id]


SPEECH_TIMEOUT = 30  # seconds

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

# Add this function to store character details as they're encountered
async def post_process_event(session_id, event):
    if event.get('type') == 'character_details' and 'name' in event and 'details' in event:
        character_details_cache[session_id][event['name']] = event

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


async def handle_event_image(session_id, event):
    if event.get('type') == 'character_update' and event.get('data') and event['data'].get('name'):
        char_name = event['data']['name']
        if event['data'].get('image'):
            image_cache[session_id][char_name] = event['data']['image']
        elif char_name in image_cache[session_id]:
            event['data']['image'] = image_cache[session_id][char_name]
        return event

    elif event.get('type') == 'world_update':
        if event.get('data'):
            world_name = 'world' #event['data']['name']
            if event['data'].get('image'):
                image_cache[session_id][world_name] = event['data']['image']
            elif world_name in image_cache[session_id]:
                event['data']['image'] = image_cache[session_id][world_name]
        return event

    elif event.get('type') == 'speak':
        if not speech_enabled[session_id]:
            return None
        try:
            message = event.get('message')
            text = message.get('text')
            audio_data = event.get('audio')
            speak_event = {
                'type': 'speak',
                'message': {
                    'name': message['name'],
                    'text': message['text'],
                },
                'audio': audio_data,
                'audio_format': 'mp3',
                'needs_speech_complete': True
            }
            event = speak_event
        except Exception as e:
            print(f"Error processing speech: {e}")
        return event
    elif event.get('type') == 'play_list':
        return None

    return event

# Add session creation endpoint
@app.get("/api/session/new")
async def create_session():
    global image_cache
    session_id = str(uuid.uuid4())
    init_session_state(session_id)
    return {"session_id": session_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    global image_cache, replay_events, current_replay_index, is_replay_mode, is_replay_running
    await websocket.accept()
    init_session_state(session_id)
    # Send initial mode information
    await websocket.send_json({
        "type": "mode_update",
        "mode": "replay"
    })
    await websocket.send_json({
            'type': 'show_update',
            'text': """Welcome to replay mode. Select a replay file to load and start playing.
Voiced files may take a moment to load.
---------------------------------------------------------------------------
Replays are recordings of previous real-time performances, and differ from them in several ways
Since these are pre-recorded, you cannot interact through character chat or Director's Chair.
Images are low quality and rarely updated, to reduce file space.
The Voice button is only effective if the replay includes audio (you may need to toggle it to hear the audio).

However, most UI elements are present and fully functional, such as character tab selection and the explore character feature.

Contact bruce@tuuyi.com for more info.
---------------------------------------------------------------------------

"""})
    await asyncio.sleep(.1)
    
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

    async def send_prologue(session_id, websocket):
        await websocket.send_json({
            'type': 'show_update',
            'text': """Welcome to replay mode. Select a replay file to load and start playing.
Voiced files may take a moment to load.
---------------------------------------------------------------------------
Replays are recordings of previous real-time performances, and differ from them in several ways
Since these are pre-recorded, you cannot interact through character chat or Director's Chair.
Images are low quality and rarely updated, to reduce file space.
The Voice button is only effective if the replay includes audio (you may need to toggle it to hear the audio).

However, most UI elements are present and fully functional, such as character tab selection and the explore character feature.

Contact bruce@tuuyi.com for more info.
---------------------------------------------------------------------------

"""})
    await asyncio.sleep(.1)
    try:
        while True:
            try:
                data = json.loads(await websocket.receive_text())
            except Exception as e:
                print(f"Error receiving or parsing websocket data: {e}")
                break  # Exit the loop to trigger cleanup
                
            # Add heartbeat handling
            if data.get('type') == 'heartbeat':
                response = await handle_heartbeat()
                await websocket.send_json(response)
                continue
                
            if data.get('type') == 'command':
                command = data.get('command') or data.get('action', '')
                if command == 'initialize':
                    try:
                        #await send_prologue(session_id, websocket)
                        json_files = [f.name for f in replays_dir.glob('*.json')]
                        await websocket.send_json({
                            "type": "play_list",
                            "plays": json_files
                        })
                    except Exception as e:
                        print(f"Error listing replay files: {e}")
                        release_session_state(session_id)
                        break
                    continue

                # Handle load: load selected replay file for playback
                elif command == 'load_play' and data.get('play'):
                    try:
                        selected_file = data['play']
                        replay_events[session_id] = load_replay_file(replays_dir / selected_file)
                        current_replay_index[session_id] = 0
                        is_replay_mode[session_id] = True
                        await send_command_ack('load_play')
                    except Exception as e:
                        print(f"Error loading replay file: {e}")
                        release_session_state(session_id)
                        break
                    continue
                
                elif data.get('action') == 'start_replay':
                    # Load replay file and start replay mode
                    replay_events[session_id] = load_replay_file(replays_dir / 'replay.json')
                    current_replay_index[session_id] = 0
                    is_replay_mode[session_id] = True
                    await send_command_ack('start_replay')

                elif data.get('action') == 'speech_complete':
                    speech_complete_event[session_id].set()
                    continue

                elif data.get('action') == 'step':
                    # Step: send next event
                    if current_replay_index[session_id] < len(replay_events[session_id]):
                        print(f"Sending step event: {replay_events[session_id][current_replay_index[session_id]].get('type')}")
                        event = await handle_event_image(session_id, replay_events[session_id][current_replay_index[session_id]])
                        if event:
                            await websocket.send_json(event)
                        # Cache character details for later use
                        if event:
                            await post_process_event(session_id, event)
                        current_replay_index[session_id] += 1
                        await send_command_ack('step')
                    else:
                        print("No more replay events")
                        await send_command_ack('step')
                        is_replay_mode[session_id] = False
                        continue

                elif data.get('action') == 'run':
                    # Cancel any existing run task
                    if replay_task.get(session_id) and not replay_task[session_id].done():
                        replay_task[session_id].cancel()
                        
                    # Define the run loop as a separate function
                    async def run_replay_loop():
                        global current_replay_index, is_replay_running
                        is_replay_running[session_id] = True
                        try:
                            while current_replay_index[session_id] < len(replay_events[session_id]) and is_replay_running[session_id]:
                                print(f"Sending step event: {replay_events[session_id][current_replay_index[session_id]].get('type')}")
                                event = await handle_event_image(session_id, replay_events[session_id][current_replay_index[session_id]])
                                if event:
                                    await websocket.send_json(event)
                                    # If this event needs speech completion, wait for it
                                    if event.get('needs_speech_complete'):
                                        speech_complete_event[session_id].clear()
                                        try:
                                            await asyncio.wait_for(speech_complete_event[session_id].wait(), timeout=SPEECH_TIMEOUT)
                                        except asyncio.CancelledError:
                                            pass
                                        except asyncio.TimeoutError:
                                            print("Speech completion timeout")
                                    
                                    # Cache character details for later use
                                    await post_process_event(session_id, event)
                                    delay = EVENT_DELAYS.get(event.get('type'), EVENT_DELAYS['default'])
                                    await asyncio.sleep(delay)
                                else:
                                    print(f"Skipping event: {replay_events[session_id][current_replay_index[session_id]].get('type')}")
                                current_replay_index[session_id] += 1
                            print("Reached end of replay")
                        except asyncio.CancelledError:
                            pass
                        finally:
                            is_replay_running[session_id] = False
                        
                    # Start the task
                    replay_task[session_id] = asyncio.create_task(run_replay_loop())
                    await send_command_ack('run')

                elif data.get('action') == 'pause':
                    # Set flag to stop the run loop
                    is_replay_running[session_id] = False
                    # Also cancel the task for immediate effect
                    if replay_task.get(session_id) and not replay_task[session_id].done():
                        replay_task[session_id].cancel()
                            
                elif data.get('action') == 'toggle_speech':
                    global speech_enabled
                    speech_enabled[session_id] = not speech_enabled[session_id]
                    await websocket.send_json({
                        'type': 'speech_toggle',
                        'enabled': speech_enabled[session_id]
                    })
                    # Add command acknowledgment
                    await send_command_ack('toggle_speech')
                    continue

                # Next, add a handler for get_character_details commands during replay
           
                # Handle get_character_details specifically during replay
                elif command == 'get_character_details' and data.get('name'):
                    character_name = data['name']
                    if character_name in character_details_cache[session_id]:
                    # Return cached character details
                        await websocket.send_json(character_details_cache[session_id][character_name])
                        continue
                    else:
                        # No cached details, send an empty response
                        await websocket.send_json({
                            "type": "character_details",
                            "name": character_name,
                            "details": {}
                        })
                    continue
                 
                elif data.get('type') == 'speech_complete':
                    speech_complete_event[session_id].set()
                    continue

    except WebSocketDisconnect:
        queue_task.cancel()
        print(f"Client disconnected: {session_id}")
        release_session_state(session_id)
        if session_id in sessions:
            del sessions[session_id]
            release_session_state(session_id)
            
    except Exception as e:
        queue_task.cancel()
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"Error in websocket handler: {e}")
        release_session_state(session_id)
        if session_id in sessions:
            del sessions[session_id]
            release_session_state(session_id)
    finally:
        # Always ensure state is reset on exit
        release_session_state(session_id)

 


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
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    current_event.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and current_event:
                        try:
                            event = json.loads(''.join(current_event))
                            events.append(event)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            print(f"Problematic JSON: {''.join(current_event)}")
                        current_event = []
    except Exception as e:
        print(f"Error opening or reading replay file: {e}")
        # Optionally: raise or return empty list
        return []
    return events


