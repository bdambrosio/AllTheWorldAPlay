import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from sim.map import WorldMap
from sim.mapview import MapVisualizer
from sim.scenarios import rural
import matplotlib.pyplot as plt

def test_rural_terrain_generation():
    terrain_rules = {
        'elevation_noise_scale': 50.0,
        'water_level': 0.2,
        'mountain_level': 0.8
    }
    
    infrastructure_rules = {
        'road_attempts': 7
    }
    
    property_rules = {
        'min_size': 50,
        'max_size': 150
    }

    resource_rules = {}  # Add empty resource rules
    
    world = WorldMap(
        width=75,
        height=75,
        scenario_module=rural,
        terrain_rules=terrain_rules,
        infrastructure_rules=infrastructure_rules,
        property_rules=property_rules,
        resource_rules=resource_rules  # Make sure this line is added
    )

    # Debug statistics
    print("\nTerrain Statistics:")
    terrain_counts = {}
    for x in range(world.width):
        for y in range(world.height):
            terrain_type = world.patches[x][y].terrain_type
            if terrain_type is not None:  # Add check for None
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
    plt.show()  # Show the elevation map
    
    viz.draw_terrain_and_infrastructure()
    plt.show()  # Show the terrain and infrastructure map

if __name__ == "__main__":
    test_rural_terrain_generation()
