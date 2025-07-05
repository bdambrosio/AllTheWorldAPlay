#!/usr/bin/env python3
"""
Test script for the new dynamic resource system.
Demonstrates how to add resource types at runtime and place them on the map.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sim.map import WorldMap, ResourceTypeRegistry
from plays.scenarios import coastal

def test_dynamic_resources():
    print("Testing Dynamic Resource System")
    print("=" * 40)
    
    # Create a small test map using the coastal scenario
    print("1. Creating map with coastal scenario...")
    map_instance = WorldMap(20, 20, coastal)
    
    # Verify the ResourceTypeRegistry is working
    print(f"2. Resource types: {type(map_instance.resource_types)}")
    print(f"   Available types: {list(map_instance.resource_types.__members__.keys())}")
    
    # Test accessing existing resource types
    print("3. Testing existing resource access...")
    gallery = map_instance.resource_types.Gallery
    print(f"   Gallery resource: {gallery} (value: {gallery.value})")
    
    # Test adding new dynamic resource types
    print("4. Adding dynamic resource types...")
    art_installation = map_instance.add_dynamic_resource_type(
        "ArtInstallation", 
        "A provocative sculpture that challenges community values"
    )
    print(f"   Added ArtInstallation: {art_installation} (value: {art_installation.value})")
    
    statue = map_instance.add_dynamic_resource_type(
        "Statue",
        "A memorial statue in the town square"
    )
    print(f"   Added Statue: {statue} (value: {statue.value})")
    
    # Verify new types are accessible
    print("5. Testing dynamic resource access...")
    accessed_installation = map_instance.resource_types.ArtInstallation
    print(f"   Accessed via registry: {accessed_installation}")
    print(f"   Equal to original: {accessed_installation == art_installation}")
    
    # Show updated members list
    print("6. Updated resource types:")
    for name in map_instance.resource_types.__members__.keys():
        resource_type = getattr(map_instance.resource_types, name)
        description = map_instance.resource_types.get_description(name)
        print(f"   {name}: {resource_type.value} - {description}")
    
    # Test placing dynamic resources
    print("7. Testing dynamic resource placement...")
    placed_installations = map_instance.place_dynamic_resource(
        resource_type_name="ArtInstallation",
        description="A controversial sculpture challenging traditional values",
        terrain_weights={"Downtown": 2.0, "Garden": 1.0},
        requires_property=True,
        count=2
    )
    
    print(f"   Placed {len(placed_installations)} art installations")
    for resource_id in placed_installations:
        resource = map_instance.resource_registry[resource_id]
        print(f"     {resource['name']}: {resource['description']} at {resource['location']}")
    
    # Test terrain-based placement
    print("8. Testing terrain-based placement...")
    placed_statues = map_instance.place_dynamic_resource(
        resource_type_name="Statue",
        description="A memorial statue commemorating local heroes",
        terrain_weights={"Garden": 3.0, "Harbor": 1.0},
        requires_property=False,
        count=1
    )
    
    print(f"   Placed {len(placed_statues)} statues")
    for resource_id in placed_statues:
        resource = map_instance.resource_registry[resource_id]
        print(f"     {resource['name']}: {resource['description']} at {resource['location']}")
    
    print("\n" + "=" * 40)
    print("Dynamic Resource System Test Complete!")
    print(f"Total resource types: {len(map_instance.resource_types.__members__)}")
    print(f"Total placed resources: {len(map_instance.resource_registry)}")

if __name__ == "__main__":
    test_dynamic_resources() 