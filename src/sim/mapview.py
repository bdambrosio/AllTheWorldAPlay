import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

class MapVisualizer:
    def __init__(self, world):
        self.world = world
        # Should get colors from world's scenario data, not hardcoded
        self.terrain_colors = {
            self.world.terrain_types.WATER: 'blue',
            self.world.terrain_types.MOUNTAIN: 'gray',
            self.world.terrain_types.HILL: 'brown',
            self.world.terrain_types.FOREST: 'darkgreen',
            self.world.terrain_types.GRASSLAND: 'lightgreen',
            self.world.terrain_types.FIELD: 'yellow'
        }
        
        # Create colormap for terrain
        terrain_colors = ['blue', 'green', 'darkgreen', 'yellow', 'gray', 'brown']
        self.terrain_cmap = ListedColormap(terrain_colors)

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
                if patch.resources and self.world.resource_types.MARKET not in patch.resources:
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
        road_edges = list(self.world.road_graph.edges())
        if road_edges:
            print(f"DEBUG: Drawing {len(road_edges)} road edges")
            for (x1, y1), (x2, y2) in road_edges:
                plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=2)

        # Draw markets
        for x in range(self.world.width):
            for y in range(self.world.height):
                patch = self.world.patches[x][y]
                if self.world.resource_types.MARKET in patch.resources:
                    plt.plot(x, y, 'r*', markersize=15)
                    print(f"DEBUG: Drawing market at {x}, {y}")

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