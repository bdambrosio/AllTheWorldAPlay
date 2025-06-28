import os, json, time, requests, sys
import traceback
import logging
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
from ReplayStateManager import ReplayStateManager, ReplayState

logger = logging.getLogger(__name__)

SPEECH_TIMEOUT = 30  # seconds

# Create replay directory if it doesn't exist
main_dir = Path(__file__).parent
replays_dir = (main_dir / '../plays/replays/').resolve()
replays_dir.mkdir(parents=True, exist_ok=True)
replay_file = None
# Small grey image (1x1 pixel) in base64
MINI_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# Store active sessions - but code only handles one session right now
sessions: Dict[str, int] = {}
replay_events: Dict[str, list] = {}
current_replay_index: Dict[str, int] = {}
# Scene navigation helpers
scene_markers: Dict[str, list] = {}          # list of indices that start a scene
current_scene_idx: Dict[str, int] = {}       # pointer within scene_markers

is_replay_mode: Dict[str, bool] = {}
speech_enabled: Dict[str, bool] = {}
character_details_cache: Dict[str, dict] = {}
image_cache: Dict[str, dict] = {}

state_managers: Dict[str, ReplayStateManager] = {}

def init_session_state(session_id, websocket_send_func=None):
    sessions[session_id] = session_id
    replay_events[session_id] = []
    current_replay_index[session_id] = 0
    scene_markers[session_id] = []
    current_scene_idx[session_id] = 0
    is_replay_mode[session_id] = False
    speech_enabled[session_id] = True
    character_details_cache[session_id] = {}
    image_cache[session_id] = {}
    if websocket_send_func is not None:
        state_managers[session_id] = ReplayStateManager(session_id, websocket_send_func)

def release_session_state(session_id):
    if session_id in replay_events:
        del replay_events[session_id]
    if session_id in current_replay_index:
        del current_replay_index[session_id]
    if session_id in is_replay_mode:
        del is_replay_mode[session_id]
    if session_id in scene_markers:
        del scene_markers[session_id]
    if session_id in current_scene_idx:
        del current_scene_idx[session_id]
    if session_id in speech_enabled:
        del speech_enabled[session_id]
    if session_id in character_details_cache:
        del character_details_cache[session_id]
    if session_id in image_cache:
        del image_cache[session_id]
    if session_id in state_managers:
        state_managers[session_id].cleanup()
        del state_managers[session_id]



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
    init_session_state(session_id, None)
    return {"session_id": session_id}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    global image_cache, replay_events, current_replay_index, is_replay_mode
    await websocket.accept()
    
    async def websocket_send_func(message):
        await websocket.send_json(message)
    
    init_session_state(session_id, websocket_send_func)
    state_manager = state_managers[session_id]
    
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

For more information, see https://github.com/bdambrosio/AllTheWorldAPlay.git
* this will updated soon to https://github.com/bdambrosio/ImprovAi.git

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

    async def page_end(session_id):
        if current_replay_index[session_id] >= len(replay_events[session_id]):
            return True
        event = replay_events[session_id][current_replay_index[session_id]]
        if event.get('type') == 'show_update' and event.get('text') and (event.get('text', '').startswith('----- Act') or event.get('text', '').startswith(' -----scene-')):
            return True
        return False

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
    
    async def run_replay_loop(session_id, break_on_page_end=True):
        state_manager = state_managers[session_id]
        logger.info(f"run_replay_loop starting, break_on_page_end={break_on_page_end}, state={state_manager.state.value}")
        try:
            # Ensure we're in PROCESSING state at the start of the loop
            if state_manager.state == ReplayState.IDLE:
                logger.warning(f"run_replay_loop called but state is {state_manager.state.value}, setting to PROCESSING")
                await state_manager._transition_state(ReplayState.PROCESSING, "run_replay_loop")
            
            event_count = 0
            while (current_replay_index[session_id] < len(replay_events[session_id]) and 
                   state_manager.state in [ReplayState.PROCESSING, ReplayState.WAITING_SPEECH]):
                
                # Log state periodically to detect pause
                if event_count % 10 == 0:
                    logger.info(f"run_replay_loop: processing event {current_replay_index[session_id]}, state={state_manager.state.value}")
                
                # Fetch event and advance pointer immediately to avoid duplicates if the loop
                # is cancelled during awaits after sending.
                idx = current_replay_index[session_id]
                current_replay_index[session_id] += 1
                event_count += 1

                event = await handle_event_image(session_id, replay_events[session_id][idx])
                if event:
                    if event.get('type') == 'speak':
                        logger.info(f"Processing speech event at index {idx}: {event.get('message', {}).get('text', 'No text')[:50]}...")
                    await websocket.send_json(event)

                    # wait for speech playback if required
                    if event.get('needs_speech_complete'):
                        logger.info(f"Starting speech wait for event at index {idx}")
                        await state_manager.start_speech_wait()
                        speech_success = await state_manager.wait_for_speech_complete()
                        logger.info(f"Speech wait completed, success: {speech_success}, state: {state_manager.state.value}")
                        if not speech_success:
                            pass

                    # store character details for later quick-look requests
                    await post_process_event(session_id, event)

                    delay = await state_manager.get_event_delay(event.get('type', 'default'))
                    await asyncio.sleep(delay)

                    # If this event represents a visual/log update, check if we should break
                    # for Step mode, but continue processing any immediately following speech events
                    if break_on_page_end and event.get('type') in ('show_update', 'world_update'):
                        logger.info("run_replay_loop: visual event processed, checking for following speech events")
                        
                        # Continue processing any immediately following speech events
                        while (current_replay_index[session_id] < len(replay_events[session_id]) and 
                               state_manager.state in [ReplayState.PROCESSING, ReplayState.WAITING_SPEECH]):
                            
                            # Peek at the next event to see if it's speech
                            next_idx = current_replay_index[session_id]
                            next_event = replay_events[session_id][next_idx]
                            
                            # If the next event is speech or character update, process it as part of this step
                            if next_event.get('type') in ('speak', 'character_update'):
                                logger.info(f"run_replay_loop: processing following {next_event.get('type')} event {next_idx}")
                                current_replay_index[session_id] += 1
                                event_count += 1
                                
                                processed_following_event = await handle_event_image(session_id, next_event)
                                if processed_following_event:
                                    await websocket.send_json(processed_following_event)
                                    
                                    # Handle speech completion if required
                                    if processed_following_event.get('needs_speech_complete'):
                                        await state_manager.start_speech_wait()
                                        speech_success = await state_manager.wait_for_speech_complete()
                                        if not speech_success:
                                            pass
                                    
                                    # Store character details
                                    await post_process_event(session_id, processed_following_event)
                                    
                                    # Add delay for the event
                                    following_delay = await state_manager.get_event_delay(processed_following_event.get('type', 'default'))
                                    await asyncio.sleep(following_delay)
                            else:
                                # Next event is not speech or character_update, break out of both loops
                                logger.info("run_replay_loop: no more following events, breaking after visual event")
                                break
                        
                        # Break out of main loop after processing visual + speech events
                        logger.info("run_replay_loop: breaking due to page end (after processing speech)")
                        break
            
            logger.info(f"run_replay_loop ending normally, final state={state_manager.state.value}")

        except asyncio.CancelledError:
            logger.info("run_replay_loop cancelled (likely due to pause)")
            logger.info(f"Final state after cancellation: {state_manager.state.value}")
            raise

    try:
        while True:
            try:
                raw_message = await websocket.receive_text()
                data = json.loads(raw_message)
                logger.info(f"Websocket message received: {data.get('type')} - {data.get('action')} - {data.get('command')}")
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
                        if not await state_manager.start_operation('load_play'):
                            continue
                            
                        selected_file = data['play']
                        replay_events[session_id] = load_replay_file(replays_dir / selected_file)
                        current_replay_index[session_id] = 0
                        # Build scene marker list for fast jumping
                        scene_markers[session_id] = [
                            i for i, ev in enumerate(replay_events[session_id])
                            if ev.get('type') == 'show_update' and (
                                ev.get('text', '').lstrip().startswith('-----scene-') or
                                ev.get('text', '').lstrip().startswith('----- Act')
                            )
                        ]
                        current_scene_idx[session_id] = 0
                        is_replay_mode[session_id] = True
                        await state_manager.complete_operation('load_play')
                    except Exception as e:
                        print(f"Error loading replay file: {e}")
                        await state_manager.enter_error_state(f"Failed to load replay file: {e}")
                        break
                    continue
                
                elif data.get('action') == 'start_replay':
                    # Load replay file and start replay mode
                    replay_events[session_id] = load_replay_file(replays_dir / 'replay.json')
                    current_replay_index[session_id] = 0
                    is_replay_mode[session_id] = True
                    await send_command_ack('start_replay')

                elif data.get('action') == 'speech_complete':
                    await state_manager.signal_speech_complete()
                    continue

                elif data.get('action') == 'step':
                    logger.info(f"Step command received, current state: {state_manager.state.value}")
                    if not await state_manager.start_operation('step'):
                        logger.warning(f"Failed to start step operation, state: {state_manager.state.value}")
                        continue
                        
                    logger.info(f"Starting step replay loop, state: {state_manager.state.value}")
                    # Start the task using safe transition - use async completion for consistent state management
                    task = await state_manager.safe_task_transition(
                        run_replay_loop(session_id, break_on_page_end=True)
                    )
                    # Add completion callback to handle when step task finishes
                    def on_step_complete(finished_task):
                        if not finished_task.cancelled():
                            logger.info("Step task completed, transitioning to idle state")
                            # Use asyncio.create_task to properly handle the async completion
                            async def complete_async():
                                await state_manager.complete_operation('step')
                            asyncio.create_task(complete_async())
                        else:
                            logger.info("Step task was cancelled")
                    task.add_done_callback(on_step_complete)                   

                elif data.get('action') == 'run':
                    logger.info(f"Run command received, current state: {state_manager.state.value}")
                    if not await state_manager.start_operation('run'):
                        logger.warning(f"Failed to start run operation, state: {state_manager.state.value}")
                        continue
                        
                    logger.info(f"Starting run replay loop, state: {state_manager.state.value}")
                    # Start the task using safe transition - DON'T await it so pause can work
                    task = await state_manager.safe_task_transition(
                        run_replay_loop(session_id, break_on_page_end=False)
                    )
                    logger.info(f"Run task created and stored: {task}, state manager task: {state_manager._replay_task}")
                    
                    # Add completion callback to handle when task finishes
                    def on_run_complete(finished_task):
                        if not finished_task.cancelled():
                            logger.info("Run task completed, transitioning to idle state")
                            # Use asyncio.create_task to properly handle the async completion
                            async def complete_async():
                                await state_manager.complete_operation('run')
                            asyncio.create_task(complete_async())
                        else:
                            logger.info("Run task was cancelled (likely due to pause)")
                    task.add_done_callback(on_run_complete)

                elif data.get('action') == 'pause':
                    logger.info(f"Pause command received, current state: {state_manager.state.value}")
                    logger.info(f"Active task exists: {state_manager._replay_task is not None and not state_manager._replay_task.done()}")
                    await state_manager.pause_operation()
                    logger.info(f"Pause operation completed, new state: {state_manager.state.value}")
                    
                elif data.get('action') == 'toggle_speech':
                    speech_enabled[session_id] = not speech_enabled[session_id]
                    await websocket.send_json({
                        'type': 'speech_toggle',
                        'enabled': speech_enabled[session_id]
                    })
                    await state_manager._send_command_ack('toggle_speech')
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
                    await state_manager.signal_speech_complete()
                    continue

                # ---------------- Scene navigation -----------------
                elif data.get('action') == 'next_scene':
                    if not await state_manager.start_operation('next_scene'):
                        continue

                    # Move to next scene if available
                    if current_scene_idx[session_id] + 1 < len(scene_markers[session_id]):
                        current_scene_idx[session_id] += 1
                        current_replay_index[session_id] = scene_markers[session_id][current_scene_idx[session_id]]

                    # Send the scene-head event to refresh UI
                    idx = current_replay_index[session_id]
                    event = await handle_event_image(session_id, replay_events[session_id][idx])
                    if event:
                        await websocket.send_json(event)
                        await post_process_event(session_id, event)
                        current_replay_index[session_id] += 1
                        
                    await state_manager.complete_operation('next_scene')

                elif data.get('action') == 'prev_scene':
                    if not await state_manager.start_operation('prev_scene'):
                        continue

                    if current_scene_idx[session_id] > 0:
                        current_scene_idx[session_id] -= 1
                        current_replay_index[session_id] = scene_markers[session_id][current_scene_idx[session_id]]

                    idx = current_replay_index[session_id]
                    event = await handle_event_image(session_id, replay_events[session_id][idx])
                    if event:
                        await websocket.send_json(event)
                        await post_process_event(session_id, event)
                        current_replay_index[session_id] += 1
                        
                    await state_manager.complete_operation('prev_scene')

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


