import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from sim.map import WorldMap, Agent 
from sim.mapview import MapVisualizer
from sim.scenarios import rural
import matplotlib.pyplot as plt

def test_rural_terrain_generation():
    world = WorldMap(
        width=75,
        height=75,
        scenario_module=rural,
    
    )

    # Debug statistics
    print("\nTerrain Statistics:")
    terrain_counts = {}
    for x in range(world.width):
        for y in range(world.height):
            terrain_type = world.patches[x][y].terrain_type
            if terrain_type is not None:
                terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
    
    for terrain_type, count in terrain_counts.items():
        print(f"{terrain_type.name}: {count} patches")

    print("\nProperty Statistics:")
    property_count = 0
    allocated_patches = 0
    for x in range(world.width):
        for y in range(world.height):
            if world.patches[x][y].property_id is not None:
                allocated_patches += 1
                property_count = max(property_count, world.patches[x][y].property_id)
    
    print(f"Number of properties: {property_count}")
    print(f"Total allocated patches: {allocated_patches}")
    print(f"Average property size: {allocated_patches/property_count if property_count else 0:.1f}")

    # Create visualizer with scenario module
    viz = MapVisualizer(world)
    viz.draw_elevation()
    plt.show()
    
    viz.draw_terrain_and_infrastructure()
    plt.show()

    a1 = Agent(35, 25, world, "a1")
    a1.move('Northwest')

    a2 = Agent(35, 25, world, "a2")
    world.register_agent(a1)
    world.register_agent(a2)
    home = world.get_resource_by_id('Marquadt Farmhouse')
    print(a1.look())
    a2.move_toward(home['name'])
    a1.move_to_resource(home['name'])

    market = world.get_resource_by_id('MARKET#1')
    print(a1.look())
    a1.move_toward(market['name'])
    a1.move_to_resource(market['name'])

if __name__ == "__main__":
    test_rural_terrain_generation()
