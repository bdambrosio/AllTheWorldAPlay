from __future__ import annotations
import importlib
from pathlib import Path
import random
from typing import Optional

import os, json, math, time, requests, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.replay import SPEECH_TIMEOUT
from src.utils.VoiceService import VoiceService

from src.sim.mapview import MapVisualizer
from src.sim.character_dataclasses import Act, Goal, Task
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking
    from src.sim.cognitive.knownActor import KnownActorManager
    from sim.context import Context

from src.sim.mapview import MapVisualizer
import utils.llm_api as llm_api
from utils.llm_api import LLM, generate_image
import base64
import asyncio
from typing import Dict, Any
import logging
import traceback  # Add at top if not already imported
from datetime import datetime, timedelta
import zmq
import zmq.asyncio
import matplotlib.pyplot as plt

home = str(Path.home())
logs_dir = os.path.join(home, '.local', 'share', 'alltheworldaplay', 'logs/')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
log_path = os.path.join(logs_dir, 'simulation.log')

# Create the logger
sim_logger = logging.getLogger('simulation_core')
sim_logger.setLevel(logging.INFO)

# Create file handler and set level
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
sim_logger.addHandler(file_handler)

# Test write using the named logger
sim_logger.info("Test message")

logger = logging.getLogger('simulation_core')

class SimulationServer:
    def __init__(self):
        self.zmq_context = zmq.asyncio.Context()
        self.command_socket = self.zmq_context.socket(zmq.PULL)
        self.result_socket = self.zmq_context.socket(zmq.PUSH)
        """Initialize with existing context from scenario"""
        self.sim_context = None
        self.server = None
        self.world_name = None
        self.initialized = False
        self.running = False
        self.paused = False
        self.watcher = None  # Add watcher storage
        self.steps_since_last_update = 0
        self.run_task = None  # Add field for run task
        self.step_task = None  # Add field for current step task
        self.image_cache = {}
        self.next_actor_index = 0
        self.known_actors_dir = None
        self.narrative_play = False # if True, run integrated narrative, otherwise run step from here
        self.speech_complete_event = asyncio.Event()  # Add speech completion event
        self.speech_enabled = False  # Add speech enabled flag
        self.voice_service = VoiceService()
        self.voice_service.set_provider('elevenlabs')

    async def start(self):
        self.command_socket.connect("tcp://127.0.0.1:5555")
        self.result_socket.connect("tcp://127.0.0.1:5556")

        await self.send_result({'type': 'show_update',
                            'message': {'name':'SimulationServer', 'text':f'----- SimulationServer started-----'}})
                
    async def send_result(self, result):
        try:
            await self.result_socket.send_json(result)
        except Exception as e:
            logger.error(f"Error sending result: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
    async def send_command_ack(self, command):
        await self.result_socket.send_json({'type': 'command_ack', 'command': command})
    
    
    async def update_character_states(self):
        """send character updates to UI for all characters"""
        for actor in self.sim_context.actors:
            await self.send_character_update(actor)
    
    async def send_character_update(self, actor, new_image=True):
        actor_data = actor.to_json()

        try:
            if actor.name not in self.image_cache or new_image:
                description = actor.generate_image_description()
                if description:
                    image_path = generate_image(self.sim_context.llm, description, filepath=actor.name+'.png')
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()
                        actor_data['image'] = image_data
                        self.image_cache[actor.name] = image_data
            elif 'image' not in actor_data:
                actor_data['image'] = self.image_cache[actor.name]
            
            await self.send_result({
                'type': 'character_update',
                'character': actor_data
            })
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error generating image for {actor.name}: {e}")
    
    async def send_character_detail(self, actor):
        actor_data = actor.get_explorer_state()
        try:
            await self.send_result({
                'type': 'character_detail',
                'character': actor_data
            })
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error generating image for {actor.name}: {e}")

    async def send_world_update(self):
        """send world update to UI"""
        update = self.sim_context.to_json()
        image_path = update['image']
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        update['image'] = image_data
        await self.send_result({
            'type': 'world_update',
            'world': update
        })

    async def get_play_list(self):
        try:
            # Get path to plays directory relative to main.py
            main_dir = Path(__file__).parent
            plays_dir = (main_dir / '../plays').resolve()
                        
            # List .py files, excluding system files
            play_files = [f.name for f in plays_dir.glob('*.py') if f.is_file() and not f.name.startswith('__')]
                        
            await self.send_result({
                'type': 'play_list',
                'plays': play_files
            })
            await self.send_command_ack('initialize')    

        except Exception as e:
            await self.send_result({
                'type': 'play_error',
                'error': f'Failed to list plays: {str(e)}'
            })

    async def load_play(self, command):
        self.image_cache = {}   
        play_name = command.get('play')
        main_dir = Path(__file__).parent
        config_path = (main_dir / '../plays/config.py').resolve()
        narrative_path = (main_dir / '../plays/narratives/').resolve()
        play_path = (main_dir / '../plays' / play_name).resolve()
        try:
            """Initialize new simulation instance"""
            if play_path is None:
                raise ValueError("Play path is required")

            import importlib.util
            if 'webworld_config' in sys.modules:
                del sys.modules['webworld_config']
            spec = importlib.util.spec_from_file_location("webworld_config", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, 'server_name'):
                raise ValueError("config.py must define a 'server_name' variable")
            if hasattr(module, 'server_name'):
                # scenario specific server name
                self.server_name = module.server_name
            if hasattr(module, 'model_name'):
                self.model_name = module.model_name
                llm_api.MODEL = self.model_name
                llm_api.set_model(self.model_name)

            if 'webworld_play' in sys.modules:
                del sys.modules['webworld_play']
            spec = importlib.util.spec_from_file_location("webworld_play", play_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)            
            if not hasattr(module, 'W'):
                raise ValueError("Play file must define a 'W' variable holding context")

            if hasattr(module, 'world_name'):
                self.world_name = module.world_name
            else:
                self.world_name = 'World'
            
            self.sim_context = module.W
            await self.send_command_ack('load_play')
            if hasattr(module, 'narrative'):
                self.sim_context.load_narrative(narrative_path / module.narrative)
            self.initialized = True

            #sim_context loaded and initialized, now set up voice service
            await self.sim_context.start()

            await self.send_world_update()
            await asyncio.sleep(0.1)
            for char in self.sim_context.actors:
                await self.send_character_update(char)
            await asyncio.sleep(0.1)
            
            self.next_actor_index = 0

            self.steps_since_last_update = 0
            home = str(Path.home())
            self.known_actors_dir = os.path.join(home, '.local', 'share', 'alltheworldaplay', 'known_actors', play_name.replace('.py', '/'))
            if not os.path.exists(self.known_actors_dir):
                os.makedirs(self.known_actors_dir)
            logger.info(f"SimulationServer: Play '{play_name}' loaded and fully initialized with {len(self.sim_context.actors)} actors")
            self.sim_context.message_queue.put({'name':self.sim_context.name, 'text':f'----- {play_name} loaded-----'})
            if hasattr(module, 'map_file_name'):
                await self.sim_context.create_character_narratives(play_name, module.map_file_name)
                self.narrative_play = True
                #await self.sim_context.run_character_narratives()
                self.narrative_task = asyncio.create_task(self.sim_context.run_integrated_narrative())
            else:
                print(f'No narrative or map file name found for {play_name}, running unplanned')
        except Exception as e:
            traceback.print_exc()
            await self.send_result({
                'type': 'play_error',
                'error': f'Failed to load play: {str(e)}'
            })

    async def step(self):
        """Perform one simulation step with optional character update callback"""
        try:
            self.processing = True  # Mark as processing
            # Process any queued injects before the step

            if self.initialized and self.narrative_play:
                self.sim_context.step = True
            elif self.initialized:
                char = self.sim_context.actors[self.next_actor_index]
                # Process character step
                print(f'{char.name} cognitive cycle')   
                await char.cognitive_cycle()
                #char.actor_models.save_to_file(os.path.join(self.known_actors_dir, f'{char.name}_known_actors.json'))
                await self.send_character_update(char)
                #await self.send_character_detail(char)
                await asyncio.sleep(0.1)
                self.next_actor_index += 1
                if self.next_actor_index >= len(self.sim_context.actors):
                    self.next_actor_index = 0
            
                #await self.update_character_states()
                #now handle context
                if self.steps_since_last_update > random.randint(4, 6):    
                    await asyncio.sleep(0.1)
                    await self.sim_context.update()
                    await self.send_world_update()
                    self.steps_since_last_update = 0
                else:
                    self.steps_since_last_update += 1
                # Yield control briefly to allow other tasks
                await asyncio.sleep(0.1)
                
            # Send command acknowledgment
            await self.send_command_ack('step')
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
        finally:
            self.processing = False  # Clear processing flag
            
    async def run_loop(self):
        """Separate task for running continuous steps"""
        try:

            while self.running and not self.paused:
                try:
                    # Complete the full step
                    await self.step()
                    
                    # Only check for cancellation after step completes
                    if not self.running or self.paused:
                        break
                        
                    # Brief delay between steps
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in run loop: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    self.running = False
                    self.paused = True
                    await self.send_result({
                        'type': 'error',
                        'error': f'Run loop error: {str(e)}'
                    })
                    break
        finally:
            # Ensure we're in a clean state when exiting
            self.processing = False
                
    async def run_cmd(self):
        """Start continuous simulation"""
        self.running = True
        self.paused = False
        if self.narrative_play:
            self.sim_context.run = True
            await self.send_command_ack('run')
        # Cancel existing run task if any
        elif self.run_task:
            self.run_task.cancel()
            try:
                await self.run_task
            except asyncio.CancelledError:
                pass
            # Start new run task
            self.run_task = asyncio.create_task(self.run_loop())
        
        # Send command acknowledgment
        await self.send_command_ack('run')
        
    async def pause_cmd(self):
        """Pause running simulation"""
        # Signal the run loop to stop at next safe point
        self.paused = True
        self.running = False
        self.sim_context.step = False
        self.sim_context.run = False
        
        # Wait for run task to complete its current step and exit
        if self.run_task:
            try:
                await self.run_task
            except asyncio.CancelledError:
                pass
            self.run_task = None
            
        await self.send_command_ack('pause')  # Acknowledge pause command
        
    async def stop_cmd(self):
        """Stop simulation completely"""
        self.running = False
        self.paused = False
        await self.send_command_ack('stop')  # Acknowledge stop command
        
    async def save_world_cmd(self, filename=None):
        """Save current world state"""
        return self.sim_context.save_world(filename)
        
    async def load_world_cmd(self, filename):
        """Load saved world state"""
        try:
            return self.sim_context.load_world(filename)
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
         
    async def inject_cmd(self, command):
        """Route inject command using watcher pattern from worldsim"""
        try:
            target_name = command.get('target')
            target,_ = self.sim_context.resolve_character(target_name.strip())
            if target is None:
                raise ValueError(f"Target {target_name} not found")
            text = command.get('text')
            Viewer = self.sim_context.get_npc_by_name('Viewer', create_if_missing=True)

            # does an npc have a task or goal? - acts say will handle this automagically
            task = Task(name='idle', description='inject', reason='inject', start_time=self.sim_context.simulation_time, duration=1, termination=f'talked to {target_name}', goal=None, actors=[Viewer, target])
            Viewer.focus_task.push(task)
            await Viewer.act_on_action(Act(mode='Say', action=text, actors=[Viewer, target], reason='inject', duration=1, source=None, target=[target]), task)
            Viewer.focus_task.pop()
            await asyncio.sleep(0.1)
            await self.send_result({
                'type': 'inject',
                'message': f'{Viewer.name} injects {target.name}: {text}'
            })

        except Exception as e:
            logger.error(f"Error in simulation inject: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
        

    def get_character_details_cmd(self, char_name):
        """Get detailed character state"""
        char = self.sim_context.get_actor_by_name(char_name)
        return char.get_explorer_state()

    def get_character_states_cmd(self):
        """Get current state of all characters including explorer state"""
        states = {}
        for char in self.sim_context.actors:
            state = char.to_json()
            state['explorer_state'] = char.get_explorer_state()  # Added here
            states[char.name] = state
        return states

    async def set_autonomy_cmd(self, command):
        """Set autonomy for all characters"""
        for char_name in command.get('autonomy'):
            char = self.sim_context.get_actor_by_name(char_name)
            if char:
                char.set_autonomy(command.get('autonomy')[char_name])

    async def process_command(self, command):
        try:
            cmd_name = command.get('action')
            if cmd_name == 'initialize':
                await self.get_play_list()
                await asyncio.sleep(0.1)
            elif cmd_name == 'load_play':
                await self.load_play(command)
                await asyncio.sleep(0.1)
            elif cmd_name == 'step':
                # Cancel any existing step task
                if self.narrative_play:
                    self.sim_context.step = True
                    await self.send_command_ack('step')
                    await asyncio.sleep(0.1)
                    return
                elif self.step_task and not self.step_task.done():
                    self.step_task.cancel()
                    try:
                        await self.step_task
                    except asyncio.CancelledError:
                        pass
                    # Start new step task
                    self.step_task = asyncio.create_task(self.step())
                    # Don't await it - let it run independently
                    await asyncio.sleep(0.1)
                    return
            elif cmd_name == 'run':
                await self.run_cmd()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'pause':
                await self.pause_cmd()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'stop':
                await self.stop_cmd()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'save_world':
                await self.save_world_cmd() 
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'load_world':
                await self.load_world_cmd()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'inject':
                await self.inject_cmd(command)
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'get_character_details':
                await self.get_character_details(command)
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'get_character_states':
                await self.get_character_states_cmd()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'choice_response':
                # Put the response in the context's choice_response queue
                if self.sim_context:
                    response = {
                        'selected_id': command.get('selected_id')
                    }
                    if command.get('custom_data'):
                        response['custom_data'] = command.get('custom_data')
                    self.sim_context.choice_response.put_nowait(response)
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'set_autonomy':
                await self.set_autonomy_cmd(command)
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'load_known_actors':
                self.load_known_actors()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'save_known_actors':
                self.save_known_actors()
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'showMap':
                try:
                    viz = MapVisualizer(self.sim_context.map)
                    viz.draw_terrain_and_infrastructure()
                    
                    # Use non-blocking mode and handle window closing
                    plt.ion()  # Turn on interactive mode
                    plt.show(block=False)  # Show without blocking
                    
                    # Wait for window to be closed
                    while plt.get_fignums():
                        plt.pause(0.1)  # Small pause to prevent CPU spinning
                    
                    plt.close('all')  # Clean up all figures
                    plt.ioff()  # Turn off interactive mode
                    
                except Exception as e:
                    logger.error(f"Error showing map: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    plt.close('all')  # Ensure cleanup even on error
                    plt.ioff()
                await asyncio.sleep(0.1)
            elif cmd_name == 'toggle_speech':
                self.speech_enabled = not self.speech_enabled
                await self.send_result({
                    'type': 'speech_toggle',
                    'enabled': self.speech_enabled
                })
                await self.send_command_ack('toggle_speech')
                await asyncio.sleep(0.1)
                return
            elif cmd_name == 'speech_complete':
                self.speech_complete_event.set()
                await self.send_command_ack('speech_complete')
                await asyncio.sleep(0.1)
                return
            else:
                await self.send_result({
                    'type': 'error',
                    'error': f'Unknown command: {cmd_name}'
                })
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            await self.send_result({
                'type': 'error',
                'error': f'Failed to process command: {str(e)}'
            })
    async def check_message_queue(self):
        """Monitor the context message queue and forward messages via ZMQ"""
        while True:
            try:
                if self.sim_context and not self.sim_context.message_queue.empty():
                    message = self.sim_context.message_queue.get_nowait()
                    if message['text'] == 'character_update':
                        # async character update messages include the actor data in the message
                        actor_data = message['data']
                        actor_name = message['name']
                        actor = self.sim_context.get_actor_by_name(actor_name)
                        if actor and 'image' in actor_data:
                            self.image_cache[actor_name] = actor_data['image']
                        if actor and 'image' not in actor_data and actor_name not in self.image_cache:
                            await self.send_character_update(actor)
                            continue
                        elif actor and 'image' not in actor_data and actor_name in self.image_cache:
                            actor_data['image'] = self.image_cache[actor_name]
                        await self.send_result({
                                'type': 'character_update',
                                'character': actor_data
                            })
                        continue
                    elif message['text'] == 'world_update':
                        # async character update messages include the actor data in the message
                        world_data = message['data']
                        image_path = world_data['image']
                        try:
                            with open(image_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode()
                                world_data['image'] = image_data
                                self.image_cache['world'] = image_data
                        except Exception as e:
                            logger.error(f"Error in world update: {e}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")
                            if self.image_cache['world']:
                                world_data['image'] = self.image_cache['world']
                
                        await self.send_result({
                            'type': 'world_update',
                            'world': world_data
                        })
                        continue
                    elif 'chat_response' in message.keys():
                        await self.send_result({
                            'type': 'chat_response',
                            'char_name': message['name'],
                            'text': message['text']
                        })
                    elif message.get('text') in ['goal_choice', 'task_choice', 'act_choice']:
                        await self.send_result(message)

                    elif message.get('text') == 'character_detail':
                        await self.send_result({
                            'type': 'character_details',
                            'name': message['name'],
                            'details': message['data']
                        })
                    else:

                        await self.send_result({
                            'type': 'show_update',
                            'message': {'name': message['name'], 'text': message['text']}  # Send the text directly instead of wrapping in message
                        })
                        if message.get('elevenlabs_params') and self.speech_enabled:  # Check self.speech_enabled
                            # generate a speak event
                            try:
                                name = message.get('name')
                                text = message.get('text')
                                if text.startswith("..."):
                                    text = text[3:-3]
                                    to_speak = f"<prosody volume='x-soft' rate='90%'>{text}</prosody>"
                                else:
                                    to_speak = text
                                speech_params = message.get('elevenlabs_params', {})
                                try:
                                    speech_paramsj = json.loads(speech_params)
                                    print(f"Synthesizing speech: {to_speak} with params: {speech_params}")
                                    audio_path = await self.sim_context.voice_service.synthesize(to_speak, speech_paramsj)
                                except Exception as e:
                                    print(f"Error synthesizing speech: {e}")
                                    print(f"Traceback:\n{traceback.format_exc()}")
                                    audio_path = None
                                
                                # Read the generated audio file and add it to the event
                                if audio_path and os.path.exists(audio_path):
                                    with open(audio_path, 'rb') as f:
                                        audio_data = base64.b64encode(f.read()).decode()
                                    speak_event = {
                                        'type': 'speak',
                                        'message': {
                                            'name': name,
                                            'text': text,
                                            'elevenlabs_params': speech_params
                                        },
                                        'audio': audio_data,
                                        'audio_format': 'mp3'
                                    }
                                    os.unlink(audio_path)  # Clean up the temporary file
                                
                                    # Clear event before sending speech
                                    self.speech_complete_event.clear()
                                    
                                    # Send the speak event
                                    await self.send_result(speak_event)
                                    print(f"Speaking: {name} {text}")
                                    
                                    # Wait for speech completion or timeout
                                    try:
                                        await asyncio.wait_for(self.speech_complete_event.wait(), 20.0)
                                    except asyncio.TimeoutError:
                                        print(f"Speech timeout for: {name} - {text[:50]}...")
                            except Exception as e:
                                print(f"Error processing speech: {e}")

            except Exception as e:
                print(f"Error in message queue: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
            await asyncio.sleep(0.1)
        
    async def run(self):
        await self.start()
        
        # Start message queue monitoring task
        message_task = asyncio.create_task(self.check_message_queue())
        
        try:
            while True:
                try:
                    command = await self.command_socket.recv_json()
                    await self.process_command(command)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Error processing command: {e}")
                    # Send error back to main process
                    await self.send_result({
                        'type': 'error',
                        'error': str(e)
                    })
        finally:
            message_task.cancel()  # Ensure message task is cleaned up

    async def get_character_details(self, command):
        """Get and send detailed character state"""
        try:
            char_name = command.get('name')  # Get name from command
            if not char_name:
                raise ValueError("Character name not provided")
                
            if not self.sim_context:
                raise ValueError("No simulation context loaded")
                
            details = self.get_character_details_cmd(char_name)
            if not details:
                raise ValueError(f"Character {char_name} not found")
            
            logger.info(f"Sending character details for {char_name}")
            response = {
                'type': 'character_details',
                'name': char_name,
                'details': details,
                'processing': False  # Indicate processing complete
            }
            # logger.debug(f"Character details response: {response}")
            await self.send_result(response)
            
        except Exception as e:
            logger.error(f"Error getting character details: {e}")
            await self.send_result({
                'type': 'error',
                'error': f'Failed to get character details: {str(e)}',
                'processing': False
            })

    def load_known_actors(self):
        """Load known actors for all characters from default location"""
        for char in self.sim_context.actors:
            try:
                filepath = os.path.join(self.known_actors_dir, f'{char.name}_known_actors.json')
                if os.path.exists(filepath):
                    char.actor_models = char.actor_models.load_from_file(filepath, char, self.sim_context)
            except Exception as e:
                logging.error(f"Error loading known actors for {char.name}: {e}")

    def save_known_actors(self):
        """Save known actors for all characters to default location"""
        for char in self.sim_context.actors:
            try:
                filepath = os.path.join(self.known_actors_dir, f'{char.name}_known_actors.json')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                char.actor_models.save_to_file(filepath)
            except Exception as e:
                logging.error(f"Error saving known actors for {char.name}: {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    server = SimulationServer()
    asyncio.run(server.run())
