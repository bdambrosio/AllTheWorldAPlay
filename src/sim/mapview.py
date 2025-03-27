import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

class MapVisualizer:
    def __init__(self, world):
        self.world = world
        
        # Get terrain types from world's scenario
        terrain_types = self.world.terrain_types
        
        # Define base colors for common terrain types
        base_colors = {
            'Water': 'blue',
            'Mountain': 'gray',
            'Forest': 'darkgreen',
            'Grassland': 'lightgreen',
            'Field': 'yellow',
            'Clearing': 'lightgreen',
            'Meadow': 'yellow'
        }
        
        # Create terrain colors mapping using world's terrain types
        self.terrain_colors = {}
        
        # Keep track of used colors
        used_colors = set()
        
        # First pass: assign base colors to matching terrain types
        for terrain_type in terrain_types:
            if terrain_type.name in base_colors:
                self.terrain_colors[terrain_type] = base_colors[terrain_type.name]
                used_colors.add(base_colors[terrain_type.name])
        
        # Second pass: assign unused base colors to remaining terrain types
        unused_colors = [color for color in base_colors.values() if color not in used_colors]
        color_index = 0
        
        for terrain_type in terrain_types:
            if terrain_type not in self.terrain_colors:
                if color_index < len(unused_colors):
                    self.terrain_colors[terrain_type] = unused_colors[color_index]
                    color_index += 1
                else:
                    # If we run out of unused colors, generate a new distinct color
                    hue = color_index / len(terrain_types)
                    self.terrain_colors[terrain_type] = plt.cm.hsv(hue)
        

    def draw_elevation(self, ax=None):
        """Draw elevation map using a continuous colormap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Create elevation data array
        elevation_data = np.zeros((self.world.width, self.world.height))
        for x in range(self.world.width):
            for y in range(self.world.height):
                elevation_data[x, y] = self.world.patches[x][y].elevation

        # Draw elevation using terrain colormap
        im = ax.imshow(elevation_data.T, cmap='terrain', origin='upper')
        plt.colorbar(im, ax=ax, label='Elevation')
        ax.set_title('Elevation Map')
        plt.tight_layout()

    def draw_terrain_and_infrastructure(self):
        plt.figure(figsize=(10, 10))
        plt.gca().set_facecolor('darkgrey')  # Add dark background
        legend_elements = []

        # Draw base terrain
        for x in range(self.world.width):
            for y in range(self.world.height):
                patch = self.world.patches[x][y]
                color = self.terrain_colors.get(patch.terrain_type, 'white')
                plt.plot(x, y, 's', color=color, markersize=5)
                # Add resource indicator if patch has any resources
                if patch.resources and self.world.scenario_module.required_resource not in patch.resources:
                    plt.plot(x, y, 'o', color='purple', markersize=3)
                    print(f"DEBUG: Drawing resource indicator at ({x}, {y})")

        # Draw property boundaries
        for x in range(self.world.width):
            for y in range(self.world.height):
                patch = self.world.patches[x][y]
                if patch.property_id is not None:
                    # Check neighboring patches for property boundaries
                    for nx, ny in [(x+1,y), (x,y+1)]:
                        if (nx < self.world.width and ny < self.world.height and 
                            self.world.patches[nx][ny].property_id != patch.property_id):
                            plt.plot([x, nx], [y, ny], 'k-', linewidth=0.5)

        # Draw roads
        road_edges = list(self.world.road_graph.edges(data=True))
        if road_edges:
            # Create a color map for different path types
            path_colors = {}
            
            # Define base colors for common infrastructure types
            base_path_colors = {
                'Road': 'yellow',
                'Trail': 'brown'
            }
            
            # Assign colors to infrastructure types that exist in the scenario
            for infra_type in self.world.infrastructure_types:
                if infra_type.name in base_path_colors:
                    path_colors[infra_type] = base_path_colors[infra_type.name]
                    # Add legend entry once per type
                    legend_elements.append(plt.Line2D([0], [0], color=base_path_colors[infra_type.name], 
                                                   linewidth=2, label=infra_type.name))
            
            # Default color for unknown types
            path_colors[None] = 'gray'
            
            # Draw the paths (without legend entries)
            for (x1, y1), (x2, y2), data in road_edges:
                path_type = data.get('type', None)
                color = path_colors.get(path_type, 'gray')
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)

        # Draw markets
        for x in range(self.world.width):
            for y in range(self.world.height):
                patch = self.world.patches[x][y]
                if self.world.scenario_module.required_resource in patch.resources:
                    plt.plot(x, y, 'r*', markersize=15)
                    print(f"DEBUG: Drawing {self.world.scenario_module.required_resource_name} at {x}, {y}")

        # Add legend elements for terrain types
        for terrain_type, color in self.terrain_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                           markerfacecolor=color, markersize=10,
                                           label=terrain_type.name))

        # Add resource indicator to legend
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='purple', markersize=10,
                                        label='Resources Present'))

        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(False)
        plt.axis('equal')

def visualize_map(world_map):
    """Convenience function to quickly view a map"""
    viz = MapVisualizer(world_map)
    viz.draw_terrain_and_infrastructure()  # Show terrain and infrastructure together 