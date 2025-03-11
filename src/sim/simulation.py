import importlib
from pathlib import Path
import random
from typing import Optional

import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.agh import Act, Character, Task
from sim.context import Context
from utils.llm_api import LLM, generate_image
import base64
from sim.human import Human  # Add this import
import asyncio
from collections import deque
from typing import Dict, Any
import logging
import traceback  # Add at top if not already imported
from datetime import datetime, timedelta
import zmq
import zmq.asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    async def start(self):
        self.command_socket.connect("tcp://127.0.0.1:5555")
        self.result_socket.connect("tcp://127.0.0.1:5556")
        logger.info("SimulationServer: ZMQ connections established and ready for commands")
                
    async def send_result(self, result):
        await self.result_socket.send_json(result)
        
    async def send_command_ack(self, command):
        await self.result_socket.send_json({'type': 'command_ack', 'command': command})
    
    
    async def update_character_states(self):
        """send character updates to UI for all characters"""
        for actor in self.sim_context.actors:
            await self.send_character_update(actor)
    
    async def send_character_update(self, actor, new_image=True):
        actor_data = actor.to_json()
        try:
            if new_image:
                description = actor.generate_image_description()
                if description:
                    image_path = generate_image(self.sim_context.llm, description, filepath=actor.name+'.png')
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()
                        actor_data['image'] = image_data
                        self.image_cache[actor.name] = image_data
            else:
                actor_data['image'] = self.image_cache[actor.name]
            
            await self.send_result({
                'type': 'character_update',
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
        play_name = command.get('play')
        main_dir = Path(__file__).parent
        config_path = (main_dir / '../plays/config.py').resolve()
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
            server_name = module.server_name

            if 'webworld_play' in sys.modules:
                del sys.modules['webworld_play']
            spec = importlib.util.spec_from_file_location("webworld_play", play_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'W'):
                raise ValueError("Play file must define a 'W' variable holding context")
            if hasattr(module, 'server_name'):
                # scenario specific server name
                self.server_name = module.server_name
            if hasattr(module, 'world_name'):
                self.world_name = module.world_name
            else:
                self.world_name = 'World'
            
            self.sim_context = module.W
            await self.send_command_ack('load_play')
            self.initialized = True

            await self.send_world_update()
            await asyncio.sleep(0.1)
            for char in self.sim_context.actors:
                await self.send_character_update(char)
            await asyncio.sleep(0.1)
            
            logger.info(f"SimulationServer: Play '{play_name}' loaded and fully initialized with {len(self.sim_context.actors)} actors")

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

            if self.initialized:
                task, chars = self.sim_context.next_act()
                # Process character step
                for char in chars:
                    if char:
                        print(f'{char.name} cognitive cycle')   
                        await char.cognitive_cycle()
                        await self.send_character_update(char)
                        break # only execute for first available actor
            
                #await self.update_character_states()
                #now handle context
                if self.steps_since_last_update > random.randint(4, 6):    
                    await asyncio.sleep(0.1)
                    await self.sim_context.senses('')
                    await self.send_world_update()
                    self.steps_since_last_update = 0
                else:
                    self.steps_since_last_update += 1
                # Yield control briefly to allow other tasks
                await asyncio.sleep(0.1)
                
                # Send command acknowledgment
                await self.send_command_ack('step')
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
        
        # Cancel existing run task if any
        if self.run_task:
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
            target = self.sim_context.resolve_reference(None, target_name.strip())
            if target is None:
                raise ValueError(f"Target {target_name} not found")
            text = command.get('text')
            viewer = self.sim_context.resolve_reference(target, 'viewer', create_if_missing=True)
            # does an npc have a task or goal? - acts say will handle this automagically
            task = Task(name='idle', description='inject', reason='inject', termination='', goal=None, actors=[viewer, target])
            viewer.focus_task.push(task)
            await viewer.act_on_action(Act('Say', 'Say', text, [viewer, target],'inject', None , target), task)
            await asyncio.sleep(0.1)
            await self.send_result({
                'type': 'inject',
                'message': f'{viewer.name} injects {target.name}: {text}'
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
                if self.step_task and not self.step_task.done():
                    self.step_task.cancel()
                    try:
                        await self.step_task
                    except asyncio.CancelledError:
                        pass
                # Start new step task
                self.step_task = asyncio.create_task(self.step())
                # Don't await it - let it run independently
                await asyncio.sleep(0.1)
            elif cmd_name == 'run':
                await self.run_cmd()
                await asyncio.sleep(0.1)
            elif cmd_name == 'pause':
                await self.pause_cmd()
                await asyncio.sleep(0.1)
            elif cmd_name == 'stop':
                await self.stop_cmd()
                await asyncio.sleep(0.1)
            elif cmd_name == 'save_world':
                await self.save_world_cmd() 
                await asyncio.sleep(0.1)
            elif cmd_name == 'load_world':
                await self.load_world_cmd()
                await asyncio.sleep(0.1)
            elif cmd_name == 'inject':
                await self.inject_cmd(command)
                await asyncio.sleep(0.1)
            elif cmd_name == 'get_character_details':
                await self.get_character_details(command)
                await asyncio.sleep(0.1)
            elif cmd_name == 'get_character_states':
                await self.get_character_states_cmd()
                await asyncio.sleep(0.1)
            elif cmd_name == 'choice_response':
                # Put the response in the context's choice_response queue
                if self.sim_context:
                    self.sim_context.choice_response.put_nowait({
                        'selected_id': command.get('selected_id')
                    })
                await asyncio.sleep(0.1)
            elif cmd_name == 'set_autonomy':
                await self.set_autonomy_cmd(command)
                await asyncio.sleep(0.1)
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
            if self.sim_context and not self.sim_context.message_queue.empty():
                message = self.sim_context.message_queue.get_nowait()
                if message['text'] == 'character_update':
                    await self.send_character_update(message['name'], new_image=False)
                elif 'chat_response' in message.keys():
                    await self.send_result({
                        'type': 'chat_response',
                        'char_name': message['name'],
                        'text': message['text']
                    })
                elif message.get('text') in ['goal_choice', 'task_choice', 'act_choice']:
                    await self.send_result(message)
                else:
                    await self.send_result({
                        'type': 'show_update',
                        'message': message
                    })
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

if __name__ == "__main__":
    server = SimulationServer()
    asyncio.run(server.run())
