import random
import time
from enum import Enum, auto
from collections import defaultdict
import heapq
from colorama import init, Fore, Back, Style
import noise
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom
import networkx as nx
from dataclasses import dataclass
from typing import Dict, Any, Type, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
import re

# Initialize colorama
init()

class ResourceType:
    """Represents a single resource type, compatible with Enum interface"""
    def __init__(self, name: str, value: int, description: str = ""):
        self.name = name
        self.value = value
        self.description = description
    
    def __eq__(self, other):
        if isinstance(other, ResourceType):
            return self.value == other.value
        if hasattr(other, 'value'):  # Enum compatibility
            return self.value == other.value
        return False
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"ResourceType.{self.name}"
    
    def __str__(self):
        return self.name

class ResourceTypeRegistry:
    """Dynamic registry for resource types, maintains Enum interface compatibility"""
    def __init__(self, base_enum=None):
        self._members = {}
        self._counter = 1000  # Start high to avoid conflicts with existing enums
        self._descriptions = {}
        
        # Import existing enum members if provided
        if base_enum:
            for name, member in base_enum.__members__.items():
                self._members[name] = ResourceType(name, member.value)
                self._descriptions[name] = getattr(member, 'description', '')
    
    def __getattr__(self, name):
        if name.startswith('_'):  # Don't interfere with private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if name in self._members:
            return self._members[name]
        raise AttributeError(f"No resource type: {name}")
    
    def __hasattr__(self, name):
        return name in self._members
    
    def add_type(self, name: str, description: str = "") -> 'ResourceType':
        """Add a new resource type dynamically"""
        if name not in self._members:
            new_type = ResourceType(name, self._counter, description)
            self._members[name] = new_type
            self._descriptions[name] = description
            self._counter += 1
            return new_type
        return self._members[name]
    
    @property 
    def __members__(self):
        """Maintain compatibility with Enum.__members__"""
        return dict(self._members)
    
    def get_description(self, name: str) -> str:
        """Get description for a resource type"""
        return self._descriptions.get(name, "")

def normalize_name_for_enum_lookup(name: str) -> str:
    """
    Convert any case format to PascalCase with underscores for enum lookup.
    Examples: 
    - "apple tree" -> "Apple_Tree"
    - "fallen branch" -> "Fallen_Branch" 
    - "bus stop" -> "Bus_Stop"
    - "coffee maker" -> "Coffee_Maker"
    - "AppleTree" -> "Apple_Tree" (handles camelCase)
    - "APPLE_TREE" -> "Apple_Tree" (handles UPPER_CASE)
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Clean and normalize the input
    name = name.strip()
    
    # Handle camelCase by inserting spaces before capitals
    # e.g. "AppleTree" -> "Apple Tree"
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # Replace non-alphanumeric characters with spaces
    name = re.sub(r'[^a-zA-Z0-9\s]', ' ', name)
    
    # Split into words and filter out empty strings
    words = [word for word in name.split() if word]
    
    # Capitalize each word and join with underscores
    return '_'.join(word.capitalize() for word in words)

def find_enum_member_by_name(enum_class, name: str):
    """
    Find enum member by name, handling various case formats.
    Works with both traditional Enums and ResourceTypeRegistry.
    
    Args:
        enum_class: The enum class or ResourceTypeRegistry to search in
        name: String name to find (in any case format)
        
    Returns:
        Enum member or ResourceType if found, None otherwise
    """
    if not name or not isinstance(name, str):
        return None
        
    # First normalize the input name
    normalized_name = normalize_name_for_enum_lookup(name)
    
    # Handle ResourceTypeRegistry
    if isinstance(enum_class, ResourceTypeRegistry):
        # Try direct match first with normalized name
        try:
            return getattr(enum_class, normalized_name)
        except AttributeError:
            pass
            
        # Try case-insensitive search through all members
        for member_name, member in enum_class.__members__.items():
            if member_name.lower() == normalized_name.lower():
                return member
            # Also try direct comparison with original name
            if member_name.lower() == name.lower().strip():
                return member
    else:
        # Handle traditional Enums
        # Try direct match first with normalized name
        if hasattr(enum_class, normalized_name):
            return getattr(enum_class, normalized_name)
        
        try:
            for member in enum_class:
                if member.name.lower() == normalized_name.lower():
                    return member
                # Also try direct comparison with original name
                if member.name.lower() == name.lower().strip():
                    return member
        except TypeError:
            # enum_class is not iterable, fall back to hasattr check
            pass
    
    return None

class RuralInfrastructure(Enum):
    Road = 1         # All roads/paths combined
    MarketSquare = 2 # Central gathering point

class Direction(Enum):
    Current = auto()
    North = auto()
    Northeast = auto()
    East = auto()
    Southeast = auto()
    South = auto()
    Southwest = auto()
    West = auto()
    Northwest = auto()

    @staticmethod
    def from_string(text):
        """
        Convert text to Direction enum, with enhanced robustness
        
        Args:
            text: String or Direction enum to convert
            
        Returns:
            Direction enum or None if no valid direction found
        """
        # Return as-is if already a Direction
        if isinstance(text, Direction):
            return text
            
        # Handle None/empty input
        if not text:
            return None
            
        # Convert to lowercase string for matching
        if not isinstance(text, str):
            return None
            
        text = text.lower().strip()
        
        # Handle common variations
        direction_map = {
            'n': Direction.North,
            'ne': Direction.Northeast, 
            'e': Direction.East,
            'se': Direction.Southeast,
            's': Direction.South, 
            'sw': Direction.Southwest,
            'w': Direction.West,
            'nw': Direction.Northwest,
            'north': Direction.North,
            'northeast': Direction.Northeast,
            'east': Direction.East, 
            'southeast': Direction.Southeast,
            'south': Direction.South,
            'southwest': Direction.Southwest,
            'west': Direction.West,
            'northwest': Direction.Northwest
        }
        
        # Try exact match first
        if text in direction_map:
            return direction_map[text]
            
        # Try matching parts of input against direction names
        for name, direction in direction_map.items():
            if name == text:
                return direction
                
        return None

def get_direction_offset(direction):
    """
    Get x,y offset for a direction
    
    Args:
        direction: Direction enum or string
        
    Returns:
        Tuple of (dx, dy) offsets or (0,0) if invalid
    """
    # Convert string to enum if needed
    if isinstance(direction, str):
        direction = Direction.from_string(direction)
        
    # Return no movement if invalid direction
    if direction is None:
        return (0, 0)
        
    offsets = {
        Direction.North: (0, -1),
        Direction.Northeast: (1, -1),
        Direction.East: (1, 0),
        Direction.Southeast: (1, 1),
        Direction.South: (0, 1),
        Direction.Southwest: (-1, 1),
        Direction.West: (-1, 0),
        Direction.Northwest: (-1, -1),
        Direction.Current: (0, 0)
    }
    
    return offsets[direction]



def get_water_flow_direction(world, x, y):
    if not world.patches[x][y].has_water:
        return None

    neighbors = world.get_neighbors(x, y)
    upstream = [n for n in neighbors if
                world.patches[n[0]][n[1]].has_water and world.patches[n[0]][n[1]].elevation > world.patches[x][
                    y].elevation]
    downstream = [n for n in neighbors if
                  world.patches[n[0]][n[1]].has_water and world.patches[n[0]][n[1]].elevation < world.patches[x][
                      y].elevation]

    if upstream and downstream:
        return "through"
    elif upstream:
        return "downstream"
    elif downstream:
        return "upstream"
    else:
        return "stagnant"


def get_direction_name(dx, dy):
    if dx == 0 and dy < 0: return Direction.North
    if dx > 0 and dy < 0: return Direction.Northeast
    if dx > 0 and dy == 0: return Direction.East
    if dx > 0 and dy > 0: return Direction.Southeast
    if dx == 0 and dy > 0: return Direction.South
    if dx < 0 and dy > 0: return Direction.Southwest
    if dx < 0 and dy == 0: return Direction.West
    if dx < 0 and dy < 0: return Direction.Northwest
    return Direction.Current

class Resource(Enum):
    Berries = 1
    Mushrooms = 2
    FallenLog = 3

class OwnershipType(Enum):
    Unallocated = 0
    Infrastructure = 1
    Allocated = 2

class ResourceAllocation:
    def __init__(self, resource_type, count, requires_property=True, terrain_weights=None):
        self.resource_type = resource_type
        self.count = count
        self.requires_property = requires_property
        self.terrain_weights = terrain_weights or {}
        self.placed = 0

    @classmethod
    def from_dict(cls, data):
        return cls(
            resource_type=data['resource_type'],
            count=data['count'],
            requires_property=data['requires_property'],
            terrain_weights=data['terrain_weights']
        )



class Patch:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.elevation = 0.0
        self.has_water = False
        self.has_path = False
        self.height = 0
        self.ownership_type = OwnershipType.Unallocated
        self.road_type = None  # Future: different road types
        self.property_id = None
        self.resource_type = None
        self.terrain_type = None
        self.infrastructure_type = None  # Added this
        self.property_type = None  # Might as well add this for future use
        self.resources = {}

    def get_slope(self):
        """Calculate the average slope (elevation difference) from this patch to its neighbors."""
        if not hasattr(self, '_slope'): # Cache the slope value
            total_diff = 0.0
            count = 0
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nx, ny = self.x + dx, self.y + dy
                if (0 <= nx < self.map.width and 0 <= ny < self.map.height):
                    total_diff += abs(self.elevation - self.map.patches[nx][ny].elevation)
                    count += 1
            self._slope = total_diff / count if count > 0 else 0
        return self._slope

class WorldMap:
    def __init__(self, width, height, scenario_module):
        self.width = width
        self.height = height
        self.scenario_module = scenario_module
        
        # Get types from scenario module
        self.terrain_types = scenario_module.terrain_types
        self.infrastructure_types = scenario_module.infrastructure_types
        self.property_types = scenario_module.property_types
        self.resource_types = scenario_module.resource_types
        
        # Get rules from scenario module
        self._terrain_rules = scenario_module.terrain_rules
        self._infrastructure_rules = scenario_module.infrastructure_rules
        self._property_rules = scenario_module.property_rules
        self._resource_rules = scenario_module.resource_rules
        
        self.resource_registry = {}
        self._resource_counters = {}  # Track counters for each resource type
        self.property_registry = {}  # Store property data

        self.patches = [[Patch(x, y) for y in range(height)] for x in range(width)]
        self.road_graph = nx.Graph()
        self.agents = []
        
        # Generate world using scenario rules
        self.generate_terrain()
        self.generate_properties()
        self.generate_resources()
        self.generate_infrastructure()

        if 'Water' not in [t.name for t in self.terrain_types]:
            raise ValueError("Scenario must define a Water terrain type")

    def get_owned_resources(self):
        """Get all resources that are NPC-owned"""
        return [res for res in self.resource_registry.values() 
                if res['properties'].get('owner')]

    def generate_terrain(self):
        """Generate terrain must ensure all patches get a terrain_type"""
        if not self._terrain_rules:
            print("WARNING: No terrain rules provided!")
            return
            
        print("Generating terrain...")
        
        # Generate elevation noise
        scale = self._terrain_rules.get('elevation_noise_scale', 50.0)
        elevation = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                elevation[x][y] = np.random.normal(0, 1)
        
        # Smooth elevation
        elevation = gaussian_filter(elevation, sigma=scale/10)
        
        # Normalize elevation to 0-1
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        # Assign elevation and terrain types
        water_level = self._terrain_rules.get('water_level', 0.2)
        mountain_level = self._terrain_rules.get('mountain_level', 0.8)
        
        print(f"DEBUG: Starting terrain assignment...")
        none_count_before = sum(1 for x in range(self.width) 
                              for y in range(self.height) 
                              if self.patches[x][y].terrain_type is None)
        print(f"DEBUG: Patches with None terrain before assignment: {none_count_before}")
        
        terrain_by_elevation = self._terrain_rules.get('terrain_by_elevation', {
            'water': {'max': 0.2, 'type': 'Water'},
            'mountain': {'min': 0.8, 'type': 'Mountain'}
        })
        
        lowland_distribution = self._terrain_rules.get('lowland_distribution', {
            'Forest': 0.4,
            'Grassland': 0.3,
            'Field': 0.3
        })

        for x in range(self.width):
            for y in range(self.height):
                self.patches[x][y].elevation = elevation[x][y]
                elev = elevation[x][y]
                
                # Check elevation-based terrains first
                terrain_set = False
                for terrain_def in terrain_by_elevation.values():
                    min_elev = terrain_def.get('min', -float('inf'))
                    max_elev = terrain_def.get('max', float('inf'))
                    if min_elev <= elev <= max_elev:
                        terrain_type = find_enum_member_by_name(self.terrain_types, terrain_def['type'])
                        if terrain_type:
                            self.patches[x][y].terrain_type = terrain_type
                            terrain_set = True
                            break
                
                # If no elevation-based terrain set, use distribution
                if not terrain_set:
                    r = random.random()
                    cumulative = 0
                    for terrain_name, chance in lowland_distribution.items():
                        cumulative += chance
                        if r < cumulative:
                            terrain_type = find_enum_member_by_name(self.terrain_types, terrain_name)
                            if terrain_type:
                                self.patches[x][y].terrain_type = terrain_type
                                break

        # Modify the smoothing code to remove HILL references
        for _ in range(3):  # 3 smoothing passes
            changes = []
            for x in range(self.width):
                for y in range(self.height):
                    patch = self.patches[x][y]
                    # Skip only elevation-determined terrains (WATER, MOUNTAIN)
                    if (patch.elevation < water_level or 
                        patch.elevation > mountain_level):
                        continue
                    
                    # Count neighbor terrains
                    neighbor_counts = {}
                    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), 
                                 (0,1), (1,-1), (1,0), (1,1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.width and 
                            0 <= ny < self.height):
                            ntype = self.patches[nx][ny].terrain_type
                            # Skip elevation-determined neighbors (WATER, MOUNTAIN)
                            elevation_terrain = False
                            for terrain_def in terrain_by_elevation.values():
                                if ntype == getattr(self.terrain_types, terrain_def['type']):
                                    elevation_terrain = True
                                    break
                            if not elevation_terrain:
                                neighbor_counts[ntype] = neighbor_counts.get(ntype, 0) + 1
                    
                    # Change to most common neighbor type if significantly more common
                    if neighbor_counts:
                        most_common = max(neighbor_counts.items(), key=lambda x: x[1])
                        if most_common[1] >= 5 and most_common[0] != patch.terrain_type:
                            changes.append((x, y, most_common[0]))
            
            # Apply all changes at once
            for x, y, new_type in changes:
                self.patches[x][y].terrain_type = new_type

        none_count_after = sum(1 for x in range(self.width) 
                             for y in range(self.height) 
                             if self.patches[x][y].terrain_type is None)
        print(f"DEBUG: Patches with None terrain after assignment: {none_count_after}")
        if none_count_after > 0:
            print("WARNING: Some patches still have None terrain!")
            # Print first few None locations for debugging
            for x in range(self.width):
                for y in range(self.height):
                    if self.patches[x][y].terrain_type is None:
                        print(f"None terrain at ({x}, {y}), elevation: {self.patches[x][y].elevation}")
                        break

    def get_resource_type(self, resource_name):
        """Check if a string matches a resource type and return the matching resource type.
        
        Args:
            resource_name: String to check against resource types
            
        Returns:
            The matching resource type if found, None otherwise
        """
        return find_enum_member_by_name(self.resource_types, resource_name)
    
    def add_dynamic_resource_type(self, name: str, description: str = "") -> ResourceType:
        """Add a new resource type dynamically to the registry.
        
        Args:
            name: Name of the new resource type
            description: Optional description
            
        Returns:
            The created ResourceType object
        """
        if isinstance(self.resource_types, ResourceTypeRegistry):
            return self.resource_types.add_type(name, description)
        else:
            raise ValueError("Dynamic resource types require ResourceTypeRegistry, not static Enum")
    
    def place_dynamic_resource(self, resource_type_name: str, description: str = "", 
                             terrain_weights: dict = None, requires_property: bool = False,
                             count: int = 1) -> list:
        """Place a dynamic resource on the map.
        
        Args:
            resource_type_name: Name of the resource type to place
            description: Description of the resource instance
            terrain_weights: Dict mapping terrain types to placement weights
            requires_property: Whether the resource requires a property
            count: Number of instances to place
            
        Returns:
            List of placed resource IDs
        """
        resource_type = self.get_resource_type(resource_type_name)
        if not resource_type:
            resource_type = self.add_dynamic_resource_type(resource_type_name, description)
        
        # Convert string terrain names to actual terrain types
        converted_weights = {}
        if terrain_weights:
            for terrain_name, weight in terrain_weights.items():
                terrain_type = self.get_terrain_type(terrain_name)
                if terrain_type:
                    converted_weights[terrain_type] = weight
        
        # Find valid locations based on terrain weights and property requirements
        candidates = []
        for x in range(self.width):
            for y in range(self.height):
                patch = self.patches[x][y]
                if patch.resources:  # Skip if patch already has resources
                    continue
                    
                if requires_property and patch.property_id is None:
                    continue
                    
                weight = converted_weights.get(patch.terrain_type, 0)
                if weight > 0:
                    candidates.append((x, y, weight))
        
        # Place resources using weighted random selection
        placed_resources = []
        for _ in range(min(count, len(candidates))):
            if not candidates:
                break
                
            # Weighted random selection
            total_weight = sum(c[2] for c in candidates)
            if total_weight <= 0:
                break
                
            rand_val = random.random() * total_weight
            cumulative = 0
            selected_idx = 0
            
            for i, (x, y, weight) in enumerate(candidates):
                cumulative += weight
                if rand_val <= cumulative:
                    selected_idx = i
                    break
            
            x, y, _ = candidates.pop(selected_idx)
            
            # Create and place the resource
            resource_id = self._generate_resource_id(resource_type)
            resource_data = {
                'id': resource_id,
                'type': resource_type,
                'name': f"{resource_type_name}{self._resource_counters.get(resource_type, 1)}",
                'description': description or f"A {resource_type_name.lower()}",
                'location': (x, y),
                'properties': {}
            }
            
            self.resource_registry[resource_id] = resource_data
            self.patches[x][y].resources[resource_id] = resource_data
            placed_resources.append(resource_id)
            
            # Update counter
            self._resource_counters[resource_type] = self._resource_counters.get(resource_type, 0) + 1
            
        return placed_resources
    
    def random_location_by_resource(self, resource_name):
        """Get a random location of a given resource type"""
        resource_type = self.get_resource_type(resource_name)
        if resource_type is None:
            return random.randint(0, self.width-1), random.randint(0, self.height-1)
        candidates = []
        for candidate in self.resource_registry:
            if self.resource_registry[candidate]['type'] == resource_type:
                candidates.append(self.resource_registry[candidate]['location'])
        if len(candidates) == 0:
            return random.randint(0, self.width-1), random.randint(0, self.height-1)
        return random.choice(candidates)
    
    def random_location_by_terrain(self, terrain_name):
        """Get a random location of a given terrain type"""
        if terrain_name is None:
            return random.randint(0, self.width-1), random.randint(0, self.height-1)
        terrain_type = self.get_terrain_type(terrain_name)
        if terrain_type is None:
            return self.random_location_by_resource(terrain_name)
        candidates = []
        for x in range(self.width):
            for y in range(self.height):
                if self.patches[x][y].terrain_type == terrain_type:
                    candidates.append((x, y))
        if len(candidates) == 0:
            return random.randint(0, self.width-1), random.randint(0, self.height-1)
        return random.choice(candidates)
    
    def generate_properties(self):
        valid_terrain = [find_enum_member_by_name(self.terrain_types, t) 
                        for t in self._property_rules['valid_terrain']]
        valid_terrain = [t for t in valid_terrain if t is not None]  # Filter out None values
        
        for x in range(self.width):
            for y in range(self.height):
                if (not self.patches[x][y].has_water and 
                    self.patches[x][y].property_id is None and
                    self.patches[x][y].terrain_type in valid_terrain):
                    self.patches[x][y].property_id = None
        
        property_id = 0
        min_size = self._property_rules.get('min_size', 50)
        max_size = self._property_rules.get('max_size', 150)
        
        # Start from suitable terrain patches
        candidates = []
        for x in range(self.width):
            for y in range(self.height):
                if (not self.patches[x][y].has_water and 
                    self.patches[x][y].property_id is None and
                    self.patches[x][y].terrain_type in valid_terrain):
                    candidates.append((x, y))
        
        # Shuffle to randomize property placement
        random.shuffle(candidates)
        
        # Try to create properties from each potential starting point
        for start_x, start_y in candidates:
            if self.patches[start_x][start_y].property_id is not None:
                continue
            
            # Generate random target size for this property
            target_size = random.randint(min_size, max_size)
            
            # Try to create property of target size
            property_patches = []
            size = self.flood_fill_property(start_x, start_y, property_id, target_size, property_patches)
            if size >= min_size:
                self.register_property(property_id, property_patches)
                property_id += 1
                print(f"Created property {property_id} with {size} patches")

    def is_near_path(self, x: int, y: int, max_distance: int) -> bool:
        # Get path type from infrastructure rules
        path_type_name = self._infrastructure_rules.get('path_type', next(iter(self.infrastructure_types.__members__)))
        path_type = find_enum_member_by_name(self.infrastructure_types, path_type_name)
        
        for dx in range(-max_distance, max_distance + 1):
            for dy in range(-max_distance, max_distance + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.patches[nx][ny].infrastructure_type == path_type):
                    return True
        return False

    def flood_fill_property(self, start_x: int, start_y: int, property_id: int, target_size: int, property_patches: list) -> int:
        """
        Flood fill to create a property, collecting patches for registry
        
        Args:
            start_x, start_y: Starting coordinates
            property_id: ID to assign to property
            target_size: Target size in patches
            property_patches: List to collect (x,y) coordinates of property patches
            
        Returns:
            Size of created property
        """
        if (self.patches[start_x][start_y].property_id is not None or
            self.patches[start_x][start_y].terrain_type == self.terrain_types.Water):
            return 0
            
        size = 0
        stack = [(start_x, start_y)]
        visited = set()
        
        while stack and size < target_size:  # Stop at target size
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            if (0 <= x < self.width and 0 <= y < self.height and
                self.patches[x][y].property_id is None and
                self.patches[x][y].terrain_type != self.terrain_types.Water):
                
                self.patches[x][y].property_id = property_id
                property_patches.append((x, y))  # Collect patch coordinates
                size += 1
                
                # Randomize neighbor order for more natural shapes
                neighbors = self.get_neighbors(x, y)
                random.shuffle(neighbors)
                for nx, ny in neighbors:
                    if (nx, ny) not in visited:
                        stack.append((nx, ny))
                    
        return size

    def _generate_resource_id(self, resource_type):
        """Generate unique ID for a resource"""
        # Check for names in rules
        if (self._resource_rules.get('names') and 
            resource_type.name in self._resource_rules['names'] and 
            len(self._resource_rules['names'][resource_type.name]) > 
            self._resource_counters.get(resource_type, 0)):
            # Use next name from list
            name = self._resource_rules['names'][resource_type.name][self._resource_counters.get(resource_type, 0)]
            self._resource_counters[resource_type] = self._resource_counters.get(resource_type, 0) + 1
            return name
        
        # Default to TYPE#N
        if resource_type not in self._resource_counters:
            self._resource_counters[resource_type] = 1
        resource_id = f"{resource_type.name}{self._resource_counters[resource_type]}"
        self._resource_counters[resource_type] += 1
        return resource_id

    def generate_resources(self):
        """Generate resources based on scenario-specific types and rules"""
        if not self._resource_rules or not self._resource_rules.get('allocations'):
            return
            
        print("Generating resources...")
        for allocation in self._resource_rules['allocations']:
            resource_type = allocation['resource_type']
            count = allocation['count']
            requires_property = allocation['requires_property']
            terrain_weights = {}
            for terrain_type, weight in allocation['terrain_weights'].items():
                if isinstance(terrain_type, str):
                    terrain_type = find_enum_member_by_name(self.terrain_types, terrain_type)
                if terrain_type is not None:
                    terrain_weights[terrain_type] = weight
            
            # Find valid locations
            candidates = []
            for x in range(self.width):
                for y in range(self.height):
                    patch = self.patches[x][y]
                    if patch.resources:  # Skip if patch already has resources
                        continue
                        
                    if requires_property and patch.property_id is None:
                        continue
                        
                    weight = terrain_weights.get(patch.terrain_type, 0)
                    if weight > 0:
                        candidates.append((x, y, weight))
            
            # Place resources
            placed = 0
            while placed < count and candidates:
                total_weight = sum(w for _, _, w in candidates)
                if total_weight <= 0:
                    break
                    
                r = random.uniform(0, total_weight)
                cumulative = 0
                for i, (x, y, weight) in enumerate(candidates):
                    cumulative += weight
                    if r <= cumulative:
                        # Generate unique ID and register resource
                        resource_id = self._generate_resource_id(resource_type)
                        
                        # Get property owner if resource is on property
                        property_id = self.patches[x][y].property_id
                        owner = None
                        if property_id is not None:
                            owner = self.get_property_owner(property_id)
                        
                        # Register resource with location, name, and owner
                        resource_record = { 
                            'type': resource_type,
                            'name': resource_id,  # Store the ID (which might be a name) with the resource
                            'description': allocation['description'],
                            'location': (x, y),
                            'properties': {'owner': owner} if owner else {}
                        }
                        self.resource_registry[resource_id] = resource_record
                        
                        # Add to patch
                        self.patches[x][y].resources[resource_id] = 1
                        
                        candidates.pop(i)
                        placed += 1
                        print(f"DEBUG: Placed {resource_id} at ({x}, {y})" + 
                              (f" owned by {owner.name}" if owner else ""))
                        break

    def get_resource_by_name(self, name):
        for resource_id in self.resource_registry:
            if self.resource_registry[resource_id]['name'] == name:
                return self.resource_registry[resource_id]
        return None

    def get_resource_list(self):
        resource_list = []
        for resource_id in self.resource_registry:
            resource_list.append(f'{self.resource_registry[resource_id]["name"]} {self.resource_registry[resource_id]["description"]} located at {self.resource_registry[resource_id]["location"]}')
        return resource_list
        
        
    def get_resource_property(self, resource_id, property_name):
        """Get a property value for a resource"""
        if resource_id not in self.resource_registry:
            print(f"ERROR: Resource {resource_id} not found")
            return None
        return self.resource_registry[resource_id]['properties'].get(property_name)

    def set_resource_property(self, resource_id, property_name, value):
        """Set a property value for a resource"""
        if resource_id not in self.resource_registry:
            print(f"ERROR: Resource {resource_id} not found")
            return False
        self.resource_registry[resource_id]['properties'][property_name] = value
        return True

    def delete_resource_property(self, resource_id, property_name):
        """Delete a property from a resource"""
        if resource_id not in self.resource_registry:
            print(f"ERROR: Resource {resource_id} not found")
            return False
        if property_name in self.resource_registry[resource_id]['properties']:
            del self.resource_registry[resource_id]['properties'][property_name]
            return True
        return False

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors

    def generate_paths(self):
        # Create paths connecting points of interest
        pass


            
    def get_visibility_cached(self, x, y, observer_height):
        """Optimized visibility calculation with caching"""
        cache_key = (x, y, observer_height)
        
        # Check if we have a recent cache entry
        current_time = time.time()
        if hasattr(self, '_visibility_cache'):
            cache_entry = self._visibility_cache.get(cache_key)
            if cache_entry and (current_time - cache_entry['time'] < 1.0):  # 1 second cache
                return cache_entry['visible_patches']
        else:
            self._visibility_cache = {}

        # Calculate visibility using quadrants for better performance
        visible_patches = []
        
        # Split into quadrants for better performance
        quadrants = [
            (range(x, min(x + 21, self.width)), range(y, min(y + 21, self.height))),    # NE
            (range(x, max(x - 21, 0), -1), range(y, min(y + 21, self.height))),         # NW
            (range(x, min(x + 21, self.width)), range(y, max(y - 21, 0), -1)),          # SE
            (range(x, max(x - 21, 0), -1), range(y, max(y - 21, 0), -1))                # SW
        ]
        
        for x_range, y_range in quadrants:
            for curr_x in x_range:
                for curr_y in y_range:
                    if self.is_visible(x, y, curr_x, curr_y, observer_height):
                        visible_patches.append(self.patches[curr_x][curr_y])
        
        # Cache the result
        self._visibility_cache[cache_key] = {
            'visible_patches': visible_patches,
            'time': current_time
        }
        
        return visible_patches

    def get_visible_agents(self, x, y, observer_height):
        """Get list of agents visible from given position"""
        visible_patches = self.get_visibility_cached(x, y, observer_height)
        visible_agents = []
        
        for agent in self.agents:
            agent_patch = self.patches[agent.x][agent.y]
            if agent_patch in visible_patches:
                visible_agents.append(agent)
                
        return visible_agents
    
    def get_visibility(self, x, y, observer_height):
        """Get list of patches visible from given position"""
        visible_patches = []
        for dx in range(-20, 21):  # Observe up to 20 patches in each direction
            for dy in range(-20, 21):
                target_x, target_y = x + dx, y + dy
                if 0 <= target_x < self.width and 0 <= target_y < self.height:
                    if self.is_visible(x, y, target_x, target_y, observer_height):
                        visible_patches.append(self.patches[target_x][target_y])
        return visible_patches

    def is_visible(self, x1, y1, x2, y2, observer_height):
        """Optimized line of sight calculation using Bresenham's algorithm"""
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance == 0:
            return True
            
        # Always see adjacent cells (including diagonals)
        if distance <= math.sqrt(2):
            return True

        # Rest of existing code for non-adjacent cells...
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        observer_elevation = self.patches[x1][y1].elevation * 100 + observer_height
        target_elevation = self.patches[x2][y2].elevation * 100 + self.patches[x2][y2].height

        while True:
            if x == x2 and y == y2:
                break

            check_patch = self.patches[x][y]
            check_elevation = check_patch.elevation * 100 + check_patch.height

            # Calculate elevation angle efficiently
            dist_so_far = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
            if dist_so_far > 0:  # Avoid division by zero
                current_angle = math.atan2(check_elevation - observer_elevation, dist_so_far)
                target_angle = math.atan2(target_elevation - observer_elevation, distance)
            
                if current_angle > target_angle:
                    return False

            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x = x + sx
            if e2 < dx:
                err = err + dx
                y = y + sy

            if not (0 <= x < self.width and 0 <= y < self.height):
                return False

        return True

    def print_visibility_map(self, x, y, observer_height):
        visible_patches = self.get_visibility(x, y, observer_height)
        for py in range(self.height):
            for px in range(self.width):
                if self.patches[px][py] in visible_patches:
                    if px == x and py == y:
                        print(Fore.RED + 'O' + Style.RESET_ALL, end='')  # Observer position
                    elif self.patches[px][py].terrain_type == self.terrain_types.Water:
                        print(Fore.BLUE + '~' + Style.RESET_ALL, end='')
                    elif self.patches[px][py].resource_type:
                        print(Fore.GREEN + '*' + Style.RESET_ALL, end='')
                    else:
                        print('·', end='')
                else:
                    print(Fore.BLACK + '█' + Style.RESET_ALL, end='')  # Non-visible areas
            print()

    def generate_initial_roads(self):
        """Generate initial road network from market to map edges"""
        if not hasattr(self, 'market_location'):
            return
        
        # Define edge points to connect to
        edges = []
        
        # North and South edges
        for x in range(0, self.width, self.width // 4):
            edges.append((x, 0))  # North edge
            edges.append((x, self.height-1))  # South edge
        
        # East and West edges
        for y in range(0, self.height, self.height // 4):
            edges.append((0, y))  # West edge
            edges.append((self.width-1, y))  # East edge
        
        # Try to connect market to each edge point
        for edge_point in edges:
            try:
                self.add_road(self.market_location, edge_point)
            except nx.NetworkXNoPath:
                print(f"Could not connect market to edge point {edge_point}")
                continue

    def register_agent(self, agent):
        self.agents.append(agent)

    def unregister_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def get_agent(self, name):
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def register_property(self, property_id, patches):
        """Register a property and its patches"""
        self.property_registry[property_id] = {
            'patches': patches,  # List of (x,y) tuples
            'owner': None
        }

    def set_property_owner(self, property_id, owner):
        """Set the owner of a property"""
        if property_id not in self.property_registry:
            print(f"ERROR: Property {property_id} not found")
            return False
        self.property_registry[property_id]['owner'] = owner
        return True

    def get_property_owner(self, property_id):
        """Get the owner of a property"""
        if property_id not in self.property_registry:
            print(f"ERROR: Property {property_id} not found")
            return None
        return self.property_registry[property_id]['owner']

    def get_resource_by_id(self, resource_id):
        """Get resource data by ID"""
        path_name = self._infrastructure_rules.get('path_type', 'road')
        resource_id = resource_id.strip().capitalize()
        if resource_id == path_name:
            return "path"
        if resource_id not in self.resource_registry:
            resource_id_no_hash = resource_id.replace('#', '')
            if resource_id_no_hash in self.resource_registry:
                return self.resource_registry[resource_id_no_hash]
            else:
                print(f"ERROR: Resource {resource_id} not found")
                return None
        return self.resource_registry[resource_id]

    def generate_market_resource(self):
        """Generate central resource (market/hut/etc) before other resources and infrastructure"""
        market_x = random.randint(self.width // 4, 3 * self.width // 4)
        market_y = random.randint(3 * self.height // 4, self.height - 1)
        
        resource_id = self._generate_resource_id(self.scenario_module.required_resource)
        self.resource_registry[resource_id] = {
            'type': self.scenario_module.required_resource,
            'name': resource_id,
            'description': "map nexus",
            'location': (market_x, market_y),
            'properties': {}
        }
        self.patches[market_x][market_y].resources[resource_id] = 1
        print(f"DEBUG: Placed {self.scenario_module.required_resource_name} resource {resource_id} at ({market_x}, {market_y})")

    def generate_infrastructure(self):
        """Generate roads connecting central resource to other resources"""
        if not self._infrastructure_rules:
            return
            
        print("Generating infrastructure...")
        
        # Generate central resource first
        self.generate_market_resource()
        
        # Get central resource location from registry
        market_resource = next((res for res in self.resource_registry.values() 
                              if res['type'] == self.scenario_module.required_resource), None)
        if not market_resource:
            print(f"ERROR: No {self.scenario_module.required_resource_name} placed")
            return
            
        market_x, market_y = market_resource['location']
        print(f"DEBUG: Starting road network from {self.scenario_module.required_resource_name} at ({market_x}, {market_y})")
        
        # Convert terrain cost rules to enum values
        self.terrain_costs = {}
        for terrain_name, cost in self._infrastructure_rules['terrain_costs'].items():
            terrain_type = find_enum_member_by_name(self.terrain_types, terrain_name)
            if terrain_type:
                self.terrain_costs[terrain_type] = cost
        
        # Get the path type from scenario (default to the first infrastructure type if not specified)
        path_type_name = self._infrastructure_rules.get('path_type', 
            next(iter(self.infrastructure_types.__members__)))
        path_type = find_enum_member_by_name(self.infrastructure_types, path_type_name)
        
        # Initialize road graph
        self.road_graph = nx.Graph()
        
        # Generate number of paths based on density
        paths_to_generate = int(self.width * self.height * self._infrastructure_rules['road_density'])
        print(f"Generating {paths_to_generate} {path_type_name.lower()}s...")
        
        # Track connected resources
        connected = {(market_x, market_y)}
        
        # Connect each resource to nearest part of existing network
        while True:
            best_cost = float('inf')
            best_path = None
            best_start = None
            best_end = None
            
            # Find nearest unconnected resource
            for resource_id, resource in self.resource_registry.items():
                if resource['type'] == self.scenario_module.required_resource:
                    continue
                    
                res_x, res_y = resource['location']
                if (res_x, res_y) in connected:
                    continue
                    
                # Try connecting to each point in existing network
                for start_x, start_y in connected:
                    cost = self.find_path_cost(start_x, start_y, res_x, res_y)
                    if cost < best_cost:
                        best_cost = cost
                        best_start = (start_x, start_y)
                        best_end = (res_x, res_y)
            
            # No more resources to connect
            if best_start is None:
                break
                
            # Build the road
            print(f"DEBUG: Adding road from {best_start} to {best_end}")
            self.build_road(best_start[0], best_start[1], best_end[0], best_end[1])
            connected.add(best_end)

        # When adding paths to the map, store the path type
        for (x1, y1), (x2, y2) in self.road_graph.edges():
            # Add path type to the edge data
            self.road_graph[(x1, y1)][(x2, y2)]['type'] = path_type

    def find_path_cost(self, start_x, start_y, end_x, end_y):
        """Estimate cost of path between points"""
        if start_x == end_x and start_y == end_y:
            return 0
            
        # Simple manhattan distance weighted by average terrain cost
        distance = abs(end_x - start_x) + abs(end_y - start_y)
        
        # Check if path is blocked by impossible terrain
        x, y = start_x, start_y
        while x != end_x or y != end_y:
            if x < end_x: x += 1
            elif x > end_x: x -= 1
            if y < end_y: y += 1
            elif y > end_y: y -= 1
            
            if self.terrain_costs[self.patches[x][y].terrain_type] == float('inf'):
                return float('inf')
                
        return distance

    def build_road(self, start_x, start_y, end_x, end_y):
        """Build road between points"""
        # Get path type from infrastructure rules
        path_type_name = self._infrastructure_rules.get('path_type', next(iter(self.infrastructure_types.__members__)))
        path_type = find_enum_member_by_name(self.infrastructure_types, path_type_name)
        
        current = (start_x, start_y)
        self.patches[start_x][start_y].has_path = True
        while current != (end_x, end_y):
            next_x = current[0]
            next_y = current[1]
            
            if current[0] < end_x: next_x += 1
            elif current[0] > end_x: next_x -= 1
            if current[1] < end_y: next_y += 1
            elif current[1] > end_y: next_y -= 1
            
            next_pos = (next_x, next_y)
            self.road_graph.add_edge(current, next_pos)
            self.patches[next_x][next_y].infrastructure_type = path_type
            self.patches[next_x][next_y].has_path = True
            current = next_pos

    def get_movement_cost(self, x, y):
        """Calculate movement cost incorporating terrain type and slope."""
        terrain_type = self.patches[x][y].terrain_type
        base_cost = self.terrain_costs.get(terrain_type, 1.0)
        if base_cost == float('inf'):
            return float('inf')
        
        # Add slope factor - steeper slopes are harder to traverse
        slope_factor = self._infrastructure_rules.get('slope_factor', 2.0)
        slope_cost = self.patches[x][y].get_slope() * slope_factor
        
        return base_cost + slope_cost

    def get_terrain_type(self, terrain_name: str):
        """Check if a string matches a terrain type and return the matching terrain type.
        
        Args:
            terrain_name: String to check against terrain types
            
        Returns:
            The matching terrain type if found, None otherwise
        """
        return find_enum_member_by_name(self.terrain_types, terrain_name)

    def get_map_summary(self) -> str:
        """
        Extract enum classes, resources, and ownership info from the parsed map data.
        
        Returns:
            YAML-style formatted string with extracted info
        """
        lines = []
        
        # Extract terrain types
        if hasattr(self.terrain_types, '__members__'):
            lines.append("TerrainTypes:")
            for terrain_name in self.terrain_types.__members__:
                lines.append(f"  - {terrain_name}")
        
        # Extract infrastructure types  
        if hasattr(self.infrastructure_types, '__members__'):
            lines.append("\nInfrastructureTypes:")
            for infra_name in self.infrastructure_types.__members__:
                lines.append(f"  - {infra_name}")
        
        # Extract property types
        if hasattr(self.property_types, '__members__'):
            lines.append("\nPropertyTypes:")
            for prop_name in self.property_types.__members__:
                lines.append(f"  - {prop_name}")
        
        # Extract resource types
        if hasattr(self.resource_types, '__members__'):
            lines.append("\nResourceTypes:")
            for resource_name in self.resource_types.__members__:
                lines.append(f"  - {resource_name}")
        
        # Extract resource allocation rules and actual instances
        if self._resource_rules and self._resource_rules.get('allocations'):
            lines.append("\nResourceAllocations:")
            for allocation in self._resource_rules['allocations']:
                resource_type = allocation['resource_type']
                lines.append(f"  {resource_type.name}:")
                lines.append(f"    description: {allocation['description']}")
                lines.append(f"    count: {allocation['count']}")
                lines.append(f"    requires_property: {allocation['requires_property']}")
                
                # Check for ownership
                if allocation.get('has_npc', False):
                    lines.append(f"    has_owner: true")
                else:
                    lines.append(f"    has_owner: false")
        
        # Extract actual resource instances created
        if self.resource_registry:
            lines.append("\nResourceInstances:")
            for resource_id, resource_data in self.resource_registry.items():
                text = f"  {resource_id}"
                # Check if resource has an owner
                owner = resource_data['properties'].get('owner')
                if owner:
                    text += f", owner: {owner.name}"
                lines.append(text)
        
        return '\n'.join(lines)

class Agent:
    def __init__(self, x, y, world, name):
        self.x = x
        self.y = y
        self.world = world
        self.name = name
        self.world.register_agent(self)


    def look(self):
        obs = get_detailed_visibility_description(self.world, self.x, self.y, self, 5)
        return obs

    def local_map(self):
        obs = get_detailed_visibility_description(self.world, self.x, self.y, self, 5)
        return obs

    def move(self, direction):
        direction = Direction.from_string(direction)
        if not direction:
            return False
        dx, dy = self.get_direction_offset(direction)
        new_x, new_y = self.x + dx, self.y + dy

        # Option 1: Wrapping edges
        # new_x = new_x % self.world.width
        # new_y = new_y % self.world.height

        # Option 2: Hard boundaries
        if 0 <= new_x < self.world.width and 0 <= new_y < self.world.height:
            self.x, self.y = new_x, new_y
            return True
        else:
            return False

    def __del__(self):
        self.world.unregister_agent(self)


    @staticmethod
    def get_direction_offset(direction_text):
        """
        Get x,y offset for a direction, handling natural language descriptions
        
        Args:
            direction_text: String that may contain a direction within it
            
        Returns:
            Tuple of (dx, dy) offsets or (0,0) if no valid direction found
        """
        # First try direct conversion
        direction = Direction.from_string(direction_text)
        if direction:
            offsets = {
                Direction.North: (0, -1),
                Direction.Northeast: (1, -1),
                Direction.East: (1, 0),
                Direction.Southeast: (1, 1),
                Direction.South: (0, 1),
                Direction.Southwest: (-1, 1),
                Direction.West: (-1, 0),
                Direction.Northwest: (-1, -1)
            }
            return offsets[direction]

        # If that fails, look for direction words in the text
        direction_words = {
            'north': (0, -1),
            'northeast': (1, -1), 
            'east': (1, 0),
            'southeast': (1, 1),
            'south': (0, 1),
            'southwest': (-1, 1),
            'west': (-1, 0),
            'northwest': (-1, -1),
            # Add common abbreviations
            'n': (0, -1),
            'ne': (1, -1),
            'e': (1, 0),
            'se': (1, 1),
            's': (0, 1),
            'sw': (-1, 1),
            'w': (-1, 0),
            'nw': (-1, -1)
        }

        # Convert to lowercase and split into words
        words = direction_text.lower().split()
        
        # Look for any direction word in the text
        for word in words:
            if word in direction_words:
                return direction_words[word]
                
        # If no direction found, return no movement
        return (0, 0)

    def get_detailed_visibility_description(self, height=5):
        return get_detailed_visibility_description(self.world, self.x, self.y, self, height)

    def move_to_resource(self, resource_id):
        """Move agent to resource location"""
        if resource_id not in self.world.resource_registry:
            print(f"ERROR: Resource {resource_id} not found")
            return False
            
        x, y = self.world.resource_registry[resource_id]['location']
        self.x = x
        self.y = y
        return True

    def direction_toward(self, resource_id):
        """Move one step toward resource"""
        if resource_id in self.world.resource_registry:
            target_x, target_y = self.world.resource_registry[resource_id]['location']
        else:
            print(f"ERROR: Resource {resource_id} not found")
            return False
        target_x, target_y = self.world.resource_registry[resource_id]['location']
        dx = target_x - self.x
        dy = target_y - self.y
        direction = get_direction_name(dx, dy)
        return direction

    def move_toward(self, resource_id):
        """Move one step toward resource"""
        if resource_id in self.world.resource_registry:
            target_x, target_y = self.world.resource_registry[resource_id]['location']
        else:
            print(f"ERROR: Resource {resource_id} not found")
            return False
        target_x, target_y = self.world.resource_registry[resource_id]['location']
        return self.move_toward_location(target_x, target_y)
                    
    def move_toward_location(self, target_x, target_y):
        # Already there
        if (self.x, self.y) == (target_x, target_y):
            return True
            
        # Get direction to target
        dx = target_x - self.x
        dy = target_y - self.y
        direction = get_direction_name(dx, dy)
        
        # Use existing move method
        return self.move(direction)

def manhattan_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)


def get_slope_description(elevation_change):
    if abs(elevation_change) < 0.01:  # You can adjust this threshold as needed
        return "Level"
    elif elevation_change > 0:
        return "Uphill"
    else:
        return "Downhill"


def get_detailed_visibility_description(world, camera_x, camera_y, observer, observer_height):
    visible_patches = world.get_visibility(camera_x, camera_y, observer_height)
    visible_patches.append(world.patches[camera_x][camera_y])  # Add current patch

    root = ET.Element("visibility_report")
    pos = ET.SubElement(root, "position")
    pos.set("x", str(camera_x))
    pos.set("y", str(camera_y))
    pos.text = ""

    for direction in Direction:
        direction_element = ET.SubElement(root, direction.name)

        if direction == Direction.Current:
            direction_patches = [world.patches[camera_x][camera_y]]
        else:
            direction_patches = [p for p in visible_patches if get_direction_name(p.x - camera_x, p.y - camera_y) == direction]

        if not direction_patches:
            ET.SubElement(direction_element, "visibility").text = "No clear visibility"
            continue

        max_distance = max(manhattan_distance(camera_x, camera_y, p.x, p.y) for p in direction_patches)
        adjacent_patch = min(direction_patches, key=lambda p: manhattan_distance(camera_x, camera_y, p.x, p.y))
        elevation_change = adjacent_patch.elevation - world.patches[camera_x][camera_y].elevation

        if direction != Direction.Current:
            ET.SubElement(direction_element, "visibility").text = str(max_distance)

        ET.SubElement(direction_element, "terrain").text = adjacent_patch.terrain_type.name

        if direction != Direction.Current:
            slope_element = ET.SubElement(direction_element, "slope")
            slope_element.text = get_slope_description(elevation_change)

        resources_element = ET.SubElement(direction_element, "resources")
        for patch in direction_patches:
            distance = manhattan_distance(camera_x, camera_y, patch.x, patch.y)
            for resource_id in patch.resources:
                resource_element = ET.SubElement(resources_element, "resource")
                resource_element.set("id", resource_id)
                resource_element.set("distance", str(distance))
                resource_element.text = ""

        water_patches = [p for p in direction_patches if p.has_water]
        if water_patches:
            water_element = ET.SubElement(direction_element, "water")
            if direction == Direction.Current:
                flow_direction = get_water_flow_direction(world, camera_x, camera_y)
                water_element.set("flow", flow_direction)
            else:
                water_distances = [manhattan_distance(camera_x, camera_y, p.x, p.y) for p in water_patches]
                water_element.set("distances", ",".join(map(str, sorted(water_distances))))

        path_patches = [p for p in direction_patches if p.has_path]
        path_name = world._infrastructure_rules.get('path_type', 'road')
        if path_patches:
            path_element = ET.SubElement(direction_element, path_name)
            path_distances = [manhattan_distance(camera_x, camera_y, p.x, p.y) for p in path_patches]
            path_element.set("distances", ",".join(map(str, sorted(path_distances))))

        visible_agents = [agent for agent in world.agents if
                            world.patches[agent.x][agent.y] in direction_patches and agent != observer]
        if visible_agents:
            agents_element = ET.SubElement(direction_element, "characters")
            for agent in visible_agents:
                agent_element = ET.SubElement(agents_element, "character")
                agent_element.set("name", agent.name)
                agent_element.set("distance", str(manhattan_distance(camera_x, camera_y, agent.x, agent.y)))
                agent_element.text = ""

    return ET.tostring(root, encoding="unicode")


def print_formatted_xml(xml_string, indent=2):
    """
    Print a formatted version of the given XML string with specified indentation.

    :param xml_string: The XML string to format
    :param indent: Number of spaces for each indentation level (default: 2)
    """
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent=' ' * indent)

    # Remove empty lines
    lines = pretty_xml.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    formatted_xml = '\n'.join(non_empty_lines)

    print(formatted_xml)


def extract_direction_info(world, xml_string, direction_name):
    """
    Extract information about a specific direction from the visibility description XML.

    :param xml_string: The XML string returned by get_detailed_visibility_description
    :param direction_name: The name of the direction to extract (e.g., "North", "Southeast")
    :return: A dictionary containing the extracted information
    """
    root = ET.fromstring(xml_string)

    # Find the direction element
    direction_elem = root.find(f".//{direction_name}")

    if direction_elem is None:
        return f"No information found for direction: {direction_name}"

    info = {}

    # Extract basic information
 
    visibility = direction_elem.findtext('visibility')
    if visibility is not None:
        try:
            info['visibility'] = int(visibility) 
        except:
            info['visibility'] = visibility
    else:
        info['visibility'] = 0

    info['terrain'] = direction_elem.findtext('terrain')

    # Extract slope information if available
    slope_elem = direction_elem.find('slope')
    if slope_elem is not None:
        info['slope'] = {
            'description': slope_elem.text,
         }

    # Extract resource information
    resources = direction_elem.find('resources')
    if resources is not None:
        info['resources'] = [
            {'id': resource.get('id'), 'distance': int(resource.get('distance'))}
            for resource in resources.findall('resource')
        ]

    # Extract water information if available
    water_elem = direction_elem.find('water')
    if water_elem is not None:
        info['water'] = {
            'flow': water_elem.get('flow')
        } 
        try:
            info['water']['distances'] = water_elem.get('distances')
        except:
            pass

    # Extract path information if available
    path_name = world._infrastructure_rules.get('path_type', 'road')
    path_elem = direction_elem.find(path_name)
    if path_elem is not None:
        info[path_name] = {
            'distances': path_elem.get('distances')
        } 

    # Extract agent information if available
    agents_elem = direction_elem.find('characters')
    if agents_elem is not None:
        try:
            info['characters'] = [
                {'name': agent.get('name'), 'distance': int(agent.get('distance')  )}
                for agent in agents_elem.findall('character')
            ]
        except:
            info['characters'] = [
                {'name': agent.get('name'), 'distance': agent.get('distance') }
                for agent in agents_elem.findall('character')
            ]

    return info


def hash_direction_info(direction_info, distance_threshold=10, world=None):
    text_view = ""
    percept = ""
    percept_summary = ""
    resources = []
    characters = []
    paths = []
    for dir in direction_info.keys():
        if dir == 'Current':
            percept_summary = f'You are in {direction_info[dir]['terrain']} terrain. '
        percept += f"#view {dir}:"
        if 'visibility' in direction_info[dir]:
            percept += f" visibility {direction_info[dir]['visibility']}"
        if 'terrain' in direction_info[dir]:
            percept += f", terrain {direction_info[dir]['terrain']}"
        if 'slope' in direction_info[dir]:
            percept += f", slope {direction_info[dir]['slope']['description']} "
        percept += "; "
        if 'resources' in direction_info[dir] and len(direction_info[dir]['resources']) > 0:
            resource_added = False
            for resource in direction_info[dir]['resources']:
                if resource['distance'] <= distance_threshold:
                    if not resource_added:
                        percept += f"resources: "
                        resource_added = True
                    percept += f"{resource['id']} distance {resource['distance']}, "
                    resources.append(resource['id'])
            percept = percept[:-2] + '; '
        path_name = world._infrastructure_rules.get('path_type', 'road')
        if path_name in direction_info[dir] and len(direction_info[dir][path_name]) > 0:
            path_distances = direction_info[dir][path_name]['distances']
            percept += f"{path_name}: distances {path_distances}"
            paths.append(path_name)

        if 'characters' in direction_info[dir] and len(direction_info[dir]['characters']) > 0:  
            character_added = False
            for character in direction_info[dir]['characters']:
                if character['distance'] <= distance_threshold:
                    if not character_added:
                        percept += f"characters: "
                        character_added = True
                    percept += f"{character['name']} distance {character['distance']}, "
                    characters.append(character['name'])
            percept = percept[:-2]
        percept += "\n"
    percept_summary += f"You see {', '.join(characters)} and {', '.join(resources)} resources{f' and {path_name}(s)' if len(paths) > 0 else ''}"
    print(percept_summary)
    return percept, resources, characters, paths, percept_summary
                
