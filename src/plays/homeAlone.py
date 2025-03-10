from enum import Enum
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sim.worldsim as worldsim
import sim.context as context, sim.agh as agh
import plays.config as configuration

server_name = configuration.server_name
# Create characters

jill = agh.Character("Jill", "I am a young confused woman.", server_name=server_name)
jill.set_drives([
    "self-knowledge: comfort with knowing one's place in the world.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "world knowledge: understanding the world and its workings."])
jill.add_to_history('Nothing except the awareness that I perceive nothing.')

server_name = configuration.server_name
class GardenTerrain(Enum):
    Grass = 1      # Roads and walkways
    Tree = 2    # Commercial/residential buildings
    Flower = 3       # Open spaces, squares
    Bush = 4        # Green spaces

class GardenResource(Enum):
    Bug = 1     # Public transport
    Butterfly = 2        # Retail locations
    Flower = 3        # Food/drink venues
    Leaf = 4       # Rest spots
    Seed = 5    # Waste disposal

W = context.Context([jill],
            "A soft glow of light in the room, a warm, cozy atmosphere.", terrain_types=GardenTerrain, 
resources=GardenResource, server_name=server_name)

#worldsim.main(context, server_name=server_name)
