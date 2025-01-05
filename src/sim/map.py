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

# Initialize colorama
init()


class Direction(Enum):
    CURRENT = auto()
    NORTH = auto()
    NORTHEAST = auto()
    EAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()

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
            'n': Direction.NORTH,
            'ne': Direction.NORTHEAST, 
            'e': Direction.EAST,
            'se': Direction.SOUTHEAST,
            's': Direction.SOUTH, 
            'sw': Direction.SOUTHWEST,
            'w': Direction.WEST,
            'nw': Direction.NORTHWEST,
            'north': Direction.NORTH,
            'northeast': Direction.NORTHEAST,
            'east': Direction.EAST, 
            'southeast': Direction.SOUTHEAST,
            'south': Direction.SOUTH,
            'southwest': Direction.SOUTHWEST,
            'west': Direction.WEST,
            'northwest': Direction.NORTHWEST
        }
        
        # Try exact match first
        if text in direction_map:
            return direction_map[text]
            
        # Try matching parts of input against direction names
        for name, direction in direction_map.items():
            if name in text:
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
        Direction.NORTH: (0, -1),
        Direction.NORTHEAST: (1, -1),
        Direction.EAST: (1, 0),
        Direction.SOUTHEAST: (1, 1),
        Direction.SOUTH: (0, 1),
        Direction.SOUTHWEST: (-1, 1),
        Direction.WEST: (-1, 0),
        Direction.NORTHWEST: (-1, -1),
        Direction.CURRENT: (0, 0)
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
    if dx == 0 and dy < 0: return Direction.NORTH
    if dx > 0 and dy < 0: return Direction.NORTHEAST
    if dx > 0 and dy == 0: return Direction.EAST
    if dx > 0 and dy > 0: return Direction.SOUTHEAST
    if dx == 0 and dy > 0: return Direction.SOUTH
    if dx < 0 and dy > 0: return Direction.SOUTHWEST
    if dx < 0 and dy == 0: return Direction.WEST
    if dx < 0 and dy < 0: return Direction.NORTHWEST
    return Direction.CURRENT

class TerrainType(Enum):
    DENSE_FOREST = 1
    LIGHT_FOREST = 2
    CLEARING = 3

class Resource(Enum):
    BERRIES = 1
    MUSHROOMS = 2
    FALLEN_LOG = 3

class Patch:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.terrain = None  # We'll set this in WorldMap.__init__
        self.elevation = 0
        self.resources = []
        self.has_water = False
        self.has_path = False
        self.height = 0

    def __lt__(self, other):
        return self.elevation < other.elevation

class WorldMap:
    def __init__(self, width, height, iterations=5):
        self.width = width
        self.height = height
        self.patches = [[Patch(x, y) for y in range(height)] for x in range(width)]
        self.generate_elevation()
        self.generate_terrain(iterations)
        self.generate_resources()
        self.set_terrain_heights()
        self.agents = []  # New list to store registered agents
        self.generate_water_features(num_sources=10, min_length=20)
        self.agents = []  # List exists but isn't fully utilized


    def register_agent(self, agent):
        self.agents.append(agent)

    def unregister_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)


    def generate_terrain(self, iterations):
        # Initialize with random terrain
        for row in self.patches:
            for patch in row:
                patch.terrain = random.choice(list(TerrainType))

        # Apply cellular automaton rules
        for _ in range(iterations):
            new_terrain = [[patch.terrain for patch in row] for row in self.patches]

            for x in range(self.width):
                for y in range(self.height):
                    neighbors = self.get_neighbors(x, y)
                    terrain_counts = {t: 0 for t in TerrainType}

                    for nx, ny in neighbors:
                        terrain_counts[self.patches[nx][ny].terrain] += 1

                    # Set terrain to the most common neighbor terrain
                    new_terrain[x][y] = max(terrain_counts, key=terrain_counts.get)

            # Update patches with new terrain
            for x in range(self.width):
                for y in range(self.height):
                    self.patches[x][y].terrain = new_terrain[x][y]


    def generate_elevation(self):
        scale = 50.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        seed = random.randint(0, 1000)

        def generate_noise_layer():
            return [[noise.pnoise2((x+random.uniform(-1,1))/scale,
                                   (y+random.uniform(-1,1))/scale,
                                   octaves=octaves,
                                   persistence=persistence,
                                   lacunarity=lacunarity,
                                   repeatx=self.width,
                                   repeaty=self.height,
                                   base=seed+random.randint(0,1000))
                     for y in range(self.height)]
                    for x in range(self.width)]

        # Generate multiple noise layers
        layers = [generate_noise_layer() for _ in range(3)]

        # Combine layers
        combined = [[sum(layer[x][y] for layer in layers) / len(layers)
                     for y in range(self.height)]
                    for x in range(self.width)]

        # Normalize elevation values
        min_elevation = min(min(row) for row in combined)
        max_elevation = max(max(row) for row in combined)

        for x in range(self.width):
            for y in range(self.height):
                normalized = (combined[x][y] - min_elevation) / (max_elevation - min_elevation)
                # Add some random variation
                self.patches[x][y].elevation = normalized + random.uniform(-0.1, 0.1)
                # Ensure elevation stays within [0, 1]
                self.patches[x][y].elevation = max(0, min(1, self.patches[x][y].elevation))

    def generate_water_features(self, num_sources=5, min_length=8):
        all_patches = [(patch.elevation, random.random(), patch) for row in self.patches for patch in row]
        water_sources = heapq.nlargest(num_sources, all_patches)

        for _, _, source in water_sources:
            self._generate_river(source, min_length)

    def _generate_river(self, source, min_length):
        current = source
        path = [current]

        while len(path) < min_length or (current.elevation > -0.3 and len(path) < self.width):
            current.has_water = True
            neighbors = self.get_neighbors(current.x, current.y)
            valid_neighbors = [
                self.patches[nx][ny] for nx, ny in neighbors
                if not self.patches[nx][ny].has_water and self.patches[nx][ny].elevation <= current.elevation
            ]

            if not valid_neighbors:
                # If stuck, try to continue in the same direction
                last_direction = (current.x - path[-2].x, current.y - path[-2].y) if len(path) > 1 else (0, 1)
                next_x, next_y = current.x + last_direction[0], current.y + last_direction[1]
                if 0 <= next_x < self.width and 0 <= next_y < self.height:
                    next_patch = self.patches[next_x][next_y]
                    if not next_patch.has_water:
                        path.append(next_patch)
                        current = next_patch
                        continue
                break

            next_patch = min(valid_neighbors, key=lambda p: p.elevation)
            path.append(next_patch)
            current = next_patch

        # Ensure minimum length
        while len(path) < min_length:
            neighbors = self.get_neighbors(current.x, current.y)
            valid_neighbors = [self.patches[nx][ny] for nx, ny in neighbors if not self.patches[nx][ny].has_water]
            if not valid_neighbors:
                break
            next_patch = random.choice(valid_neighbors)
            path.append(next_patch)
            current = next_patch

        for patch in path:
            patch.has_water = True

    def _smooth_river(self, path):
        for i in range(1, len(path) - 1):
            prev, current, next = path[i - 1], path[i], path[i + 1]
            dx = (prev.x + next.x) // 2 - current.x
            dy = (prev.y + next.y) // 2 - current.y

            if abs(dx) <= 1 and abs(dy) <= 1:
                smoothed = self.patches[current.x + dx][current.y + dy]
                smoothed.has_water = True
                path[i] = smoothed

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def generate_resources(self):
        for row in self.patches:
            for patch in row:
                if patch.has_water:
                    continue  # Skip water patches

                # Berries: More common in clearings and light forests, less common at high elevations
                if random.random() < 0.1 and patch.elevation < 0.7:
                    if patch.terrain == TerrainType.CLEARING:
                        patch.resources.append(Resource.BERRIES)
                    elif patch.terrain == TerrainType.LIGHT_FOREST and random.random() < 0.7:
                        patch.resources.append(Resource.BERRIES)

                # Mushrooms: More common in dense and light forests, prefer middle elevations
                if 0.3 < patch.elevation < 0.8:
                    if patch.terrain in [TerrainType.DENSE_FOREST, TerrainType.LIGHT_FOREST] and random.random() < 0.15:
                        patch.resources.append(Resource.MUSHROOMS)

                # Fallen logs: More common in dense forests, slightly more common at higher elevations
                if random.random() < 0.05 + (0.05 * patch.elevation):
                    if patch.terrain == TerrainType.DENSE_FOREST:
                        patch.resources.append(Resource.FALLEN_LOG)
                    elif patch.terrain == TerrainType.LIGHT_FOREST and random.random() < 0.3:
                        patch.resources.append(Resource.FALLEN_LOG)

    def print_map(self):
        terrain_chars = {
            TerrainType.DENSE_FOREST: Fore.GREEN + '█' + Style.RESET_ALL,
            TerrainType.LIGHT_FOREST: Fore.LIGHTGREEN_EX + '▓' + Style.RESET_ALL,
            TerrainType.CLEARING: Fore.YELLOW + '░' + Style.RESET_ALL
        }

        resource_chars = {
            Resource.BERRIES: Fore.RED + '•' + Style.RESET_ALL,
            Resource.MUSHROOMS: Fore.MAGENTA + '∩' + Style.RESET_ALL,
            Resource.FALLEN_LOG: Fore.WHITE + '=' + Style.RESET_ALL
        }

        for y in range(self.height):
            for x in range(self.width):
                patch = self.patches[x][y]
                if patch.has_water:
                    print(Fore.BLUE + '~' + Style.RESET_ALL, end='')
                elif patch.resources:
                    print(resource_chars[patch.resources[0]], end='')
                else:
                    print(terrain_chars[patch.terrain], end='')
            print()  # New line after each row

    def print_color_elevation_map(self):
        # Define elevation ranges and corresponding colors
        elevation_colors = [
            (0.0, Fore.BLUE), (0.2, Fore.CYAN), (0.4, Fore.GREEN),
            (0.6, Fore.YELLOW), (0.8, Fore.RED)
        ]

        for y in range(self.height):
            for x in range(self.width):
                elevation = self.patches[x][y].elevation
                for threshold, color in elevation_colors:
                    if elevation <= threshold:
                        print(color + '█' + Style.RESET_ALL, end='')
                        break
                else:
                    print(Fore.WHITE + '█' + Style.RESET_ALL, end='')
            print()  # New line after each row

    def generate_paths(self):
        # Create paths connecting points of interest
        pass

    def set_terrain_heights(self):
        for row in self.patches:
            for patch in row:
                if patch.terrain == TerrainType.DENSE_FOREST:
                    patch.height = 20  # 20 feet
                elif patch.terrain == TerrainType.LIGHT_FOREST:
                    patch.height = 15  # 15 feet
                elif patch.terrain == TerrainType.CLEARING:
                    patch.height = 1  # 1 foot (grass)
                if patch.has_water:
                    patch.height = 0  # Water level

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
        """Check if target position is visible from observer position
        Uses Bresenham's line algorithm for more efficient line traversal"""
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance == 0:
            return True

        # Use Bresenham's line algorithm for more efficient line traversal
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

    def get_visibility(self, x, y, observer_height):
            visible_patches = []
            for dx in range(-20, 21):  # Observe up to 20 patches in each direction
                for dy in range(-20, 21):
                    target_x, target_y = x + dx, y + dy
                    if 0 <= target_x < self.width and 0 <= target_y < self.height:
                        if self.is_visible(x, y, target_x, target_y, observer_height):
                            visible_patches.append(self.patches[target_x][target_y])
            return visible_patches

    def is_visible_old(self, x1, y1, x2, y2, observer_height):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance == 0:
            return True  # The observer's own patch is always visible

        # Calculate the angle to the target patch
        angle = math.atan2(y2 - y1, x2 - x1)

        # Check visibility along the line of sight
        for i in range(1, int(distance) + 1):
            check_x = int(x1 + i * math.cos(angle))
            check_y = int(y1 + i * math.sin(angle))

            if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                return False

            check_patch = self.patches[check_x][check_y]
            observer_elevation = self.patches[x1][y1].elevation * 100 + observer_height
            target_elevation = check_patch.elevation * 100 + check_patch.height

            # Calculate the elevation angle to the current check point
            elevation_angle = math.atan2(target_elevation - observer_elevation, i)

            # If this angle is greater than the angle to the target, the target is not visible
            if elevation_angle > math.atan2(
                    self.patches[x2][y2].elevation * 100 + self.patches[x2][y2].height - observer_elevation, distance):
                return False

        return True

    # 4. Optimized Line of Sight
    def is_visible(self, x1, y1, x2, y2, observer_height):
        """Optimized line of sight calculation using Bresenham's algorithm"""
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance == 0:
            return True

        # Use Bresenham's line algorithm for more efficient line traversal
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
                    elif self.patches[px][py].has_water:
                        print(Fore.BLUE + '~' + Style.RESET_ALL, end='')
                    elif self.patches[px][py].resources:
                        resource_chars = {
                            Resource.BERRIES: Fore.RED + '•' + Style.RESET_ALL,
                            Resource.MUSHROOMS: Fore.MAGENTA + '∩' + Style.RESET_ALL,
                            Resource.FALLEN_LOG: Fore.WHITE + '=' + Style.RESET_ALL
                        }
                        print(resource_chars[self.patches[px][py].resources[0]], end='')
                    else:
                        terrain_chars = {
                            TerrainType.DENSE_FOREST: Fore.GREEN + '█' + Style.RESET_ALL,
                            TerrainType.LIGHT_FOREST: Fore.LIGHTGREEN_EX + '▓' + Style.RESET_ALL,
                            TerrainType.CLEARING: Fore.YELLOW + '░' + Style.RESET_ALL
                        }
                        print(terrain_chars[self.patches[px][py].terrain], end='')
                else:
                    print(Fore.BLACK + '█' + Style.RESET_ALL, end='')  # Non-visible areas
            print()



class Agent:
    def __init__(self, x, y, world, name):
        self.x = x
        self.y = y
        self.world = world
        self.name = name
        self.world.register_agent(self)


    def look(self):
        obs = get_detailed_visibility_description(self.world, self.x, self.y, 5)
        return obs

    def local_map(self):
        obs = get_detailed_visibility_description(self.world, self.x, self.y, 5)
        return obs

    def move(self, direction):
        direction = Direction.from_string(direction)
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
    def get_direction_offset(direction):
        direction = Direction.from_string(direction)
        offsets = {
            Direction.NORTH: (0, -1),
            Direction.NORTHEAST: (1, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTHEAST: (1, 1),
            Direction.SOUTH: (0, 1),
            Direction.SOUTHWEST: (-1, 1),
            Direction.WEST: (-1, 0),
            Direction.NORTHWEST: (-1, -1)
        }
        return offsets[direction]

    def get_detailed_visibility_description(self, height=5):
        return get_detailed_visibility_description(self.world, self.x, self.y, height)

def manhattan_distance(x1, y1, x2, y2):
    return abs(x2 - x1) + abs(y2 - y1)


def get_slope_description(elevation_change):
    if abs(elevation_change) < 0.01:  # You can adjust this threshold as needed
        return "Level"
    elif elevation_change > 0:
        return "Uphill"
    else:
        return "Downhill"


def get_detailed_visibility_description(world, x, y, observer_height):
    visible_patches = world.get_visibility(x, y, observer_height)
    visible_patches.append(world.patches[x][y])  # Add current patch

    root = ET.Element("visibility_report")
    ET.SubElement(root, "position", x=str(x), y=str(y))

    for direction in Direction:
        direction_element = ET.SubElement(root, "direction", name=direction.name)

        if direction == Direction.CURRENT:
            direction_patches = [world.patches[x][y]]
        else:
            direction_patches = [p for p in visible_patches if get_direction_name(p.x - x, p.y - y) == direction]

        if not direction_patches:
            ET.SubElement(direction_element, "visibility").text = "No clear visibility"
            continue

        max_distance = max(manhattan_distance(x, y, p.x, p.y) for p in direction_patches)
        adjacent_patch = min(direction_patches, key=lambda p: manhattan_distance(x, y, p.x, p.y))
        elevation_change = adjacent_patch.elevation - world.patches[x][y].elevation

        if direction != Direction.CURRENT:
            ET.SubElement(direction_element, "visibility").text = str(max_distance)

        ET.SubElement(direction_element, "terrain").text = adjacent_patch.terrain.name.replace('_', ' ').title()

        if direction != Direction.CURRENT:
            slope_element = ET.SubElement(direction_element, "slope")
            slope_element.text = get_slope_description(elevation_change)
            slope_element.set("elevation_change", f"{elevation_change:.2f}")

        resources_element = ET.SubElement(direction_element, "resources")
        for patch in direction_patches:
            distance = manhattan_distance(x, y, patch.x, patch.y)
            for resource in patch.resources:
                resource_element = ET.SubElement(resources_element, "resource")
                resource_element.set("name", resource.name.replace('_', ' ').title())
                resource_element.set("distance", str(distance))

        water_patches = [p for p in direction_patches if p.has_water]
        if water_patches:
            water_element = ET.SubElement(direction_element, "water")
            if direction == Direction.CURRENT:
                flow_direction = get_water_flow_direction(world, x, y)
                water_element.set("flow", flow_direction)
            else:
                water_distances = [manhattan_distance(x, y, p.x, p.y) for p in water_patches]
                water_element.set("distances", ",".join(map(str, sorted(water_distances))))

        visible_agents = [agent for agent in world.agents if
                          world.patches[agent.x][agent.y] in direction_patches and (agent.x != x or agent.y != y)]
        if visible_agents:
            agents_element = ET.SubElement(direction_element, "agents")
            for agent in visible_agents:
                agent_element = ET.SubElement(agents_element, "agent")
                agent_element.set("name", agent.name)
                agent_element.set("distance", str(manhattan_distance(x, y, agent.x, agent.y)))

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


def extract_direction_info(xml_string, direction_name):
    """
    Extract information about a specific direction from the visibility description XML.

    :param xml_string: The XML string returned by get_detailed_visibility_description
    :param direction_name: The name of the direction to extract (e.g., "NORTH", "SOUTHEAST")
    :return: A dictionary containing the extracted information
    """
    root = ET.fromstring(xml_string)

    # Find the direction element
    direction_elem = root.find(f".//direction[@name='{direction_name}']")

    if direction_elem is None:
        return f"No information found for direction: {direction_name}"

    info = {}

    # Extract basic information
    info['visibility'] = direction_elem.findtext('visibility')
    info['terrain'] = direction_elem.findtext('terrain')

    # Extract slope information if available
    slope_elem = direction_elem.find('slope')
    if slope_elem is not None:
        info['slope'] = {
            'description': slope_elem.text,
            'elevation_change': slope_elem.get('elevation_change')
        }

    # Extract resource information
    resources = direction_elem.find('resources')
    if resources is not None:
        info['resources'] = [
            {'name': resource.get('name'), 'distance': resource.get('distance')}
            for resource in resources.findall('resource')
        ]

    # Extract water information if available
    water_elem = direction_elem.find('water')
    if water_elem is not None:
        info['water'] = {
            'flow': water_elem.get('flow'),
            'distances': water_elem.get('distances')
        }

    # Extract agent information if available
    agents_elem = direction_elem.find('agents')
    if agents_elem is not None:
        info['agents'] = [
            {'name': agent.get('name'), 'distance': agent.get('distance')}
            for agent in agents_elem.findall('agent')
        ]

    return info



if __name__ == '__main__':
    # Example usage
    world = WorldMap(80, 40)
    world.generate_water_features(num_sources=5, min_length=10)

    me = Agent(40, 20, world, "Player")
    agent1 = Agent(40, 19, world, "Agent 1")
    agent2 = Agent(38, 18, world, "Agent 2")

    print(me.look())
    detailed_visibility_description = get_detailed_visibility_description(world, me.x, me.y, 5)
    print_formatted_xml(detailed_visibility_description)

    me.move(Direction.NORTHEAST)
    detailed_visibility_description = get_detailed_visibility_description(world, me.x, me.y, 5)
    print_formatted_xml(detailed_visibility_description)

    me.move(Direction.NORTHEAST)
    detailed_visibility_description = get_detailed_visibility_description(world, me.x, me.y, 5)
    print_formatted_xml(detailed_visibility_description)
    print()
    print(extract_direction_info(detailed_visibility_description, 'NORTH'))

