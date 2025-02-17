import zmq
import asyncio
import json
import yaml

from sim import agh, context

    # Make Engine a singleton
    
_instance = None

class Engine:

    def __init__(self):
            print("Creating Engine singleton...")  # Debug
            # Do initialization here instead of __init__
            self.context = zmq.Context()
            self.command_socket = self.context.socket(zmq.REP)
            self.command_socket.bind("tcp://127.0.0.1:5555")
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind("tcp://127.0.0.1:5556")
            self.simulation_context = None

    def publish(self, msg):
        """Publish update to UI"""
        self.publisher.send_json(msg)

    async def handle_commands(self):
        """Process incoming commands from main.py"""
        while True:
            message = self.command_socket.recv_json()
            command = message['command']
            print(f"Received command: {command}")
            if command == 'initialize':
                # Create Engine instance before importing play
                get_engine()
                with open('/home/bruce/Downloads/AllTheWorldAPlay/src/plays/lost.yaml', 'r') as f:
                    config = yaml.safe_load(f)

                server = config['server']
                actors = []
                for c in config['characters']:
                    char = agh.Agh(c['name'], c['description'], server=server)
                    if 'drives' in c:
                        print(f"Drives for {c['name']}:")
                        print(f"Type: {type(c['drives'])}")
                        print(f"Content: {c['drives']}")
                        print(f"First drive type: {type(c['drives'][0])}")
                        char.set_drives(c['drives'])
                    if 'history' in c:
                        for h in c['history']:
                            char.add_to_history(h)
                    actors.append(char)
                self.simulation_context = context.Context(
                    actors, 
                    config['world']['description'],
                    server=server,
                    engine=self
                )
                self.command_socket.send_json({"status": "ok"})
                
            elif command == 'step':
                task, actors = self.simulation_context.next_act()
                for actor in actors:
                    if actor:
                        actor.cognitive_cycle()
                        # Publish actor state update
                        state = actor.get_explorer_state()
                        self.publish({
                            "type": "character_update",
                            "name": actor.name,
                            "data": state
                        })
                self.command_socket.send_json({"status": "ok"})
                
            elif command == 'get_state':
                state = {
                    'actors': [a.get_explorer_state() for a in self.simulation_context.actors],
                    'world': self.simulation_context.current_state
                }
                self.command_socket.send_json(state)

def get_engine():
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = Engine()
        print("Engine singleton created")  # Debug
    return _instance

async def run_engine():
    """Creates the engine if necessary, then runs handle_commands."""
    engine = get_engine()
    await engine.handle_commands()

# Only run if called as a script, not when imported
if __name__ == '__main__':
    asyncio.run(run_engine())

