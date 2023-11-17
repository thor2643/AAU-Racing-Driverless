import pygame
import sys
import numpy as np
import os
from scipy.ndimage import zoom

def depth_map_to_point_cloud(depth_map):
    rows, cols = depth_map.shape
    y, x = np.mgrid[0:rows, 0:cols]

    # Downsampling for performance
    downsample_factor = 40
    depth_map_downsampled = zoom(depth_map, 1/downsample_factor)
    x_downsampled = zoom(x, 1/downsample_factor)
    y_downsampled = zoom(y, 1/downsample_factor)

    # Identify NaN values and replace them with a default depth (or remove them)
    invalid_depth = np.isnan(depth_map_downsampled)
    depth_map_downsampled[invalid_depth] = 0  # Replace NaN with 0 (or use another default depth)

    x_world = (x_downsampled - cols / (2 * downsample_factor)) * depth_map_downsampled
    y_world = (y_downsampled - rows / (2 * downsample_factor)) * depth_map_downsampled
    z_world = depth_map_downsampled

    # Optionally, remove points with NaN depth from the point cloud
    point_cloud = np.dstack((x_world, y_world, z_world))
    point_cloud = point_cloud[~invalid_depth]

    return point_cloud

# Directory containing depth map files
depth_maps_directory = "Data_AccelerationTrack\\1\DepthData\\"

# List all .npy files in the directory
depth_map_files = [f for f in os.listdir(depth_maps_directory) if f.endswith(".npy")]

depth_point_cloud = []

pygame.init()

# Set up the Pygame window
window_size = (800, 600)
screen = pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.HWACCEL)
pygame.display.set_caption('Real-Time Depth Camera Point Cloud')

clock = pygame.time.Clock()

# Function to update the depth camera point cloud in real-time
def update_depth_point_cloud():
    global depth_point_cloud

    # Iterate over each depth map file
    for depth_map_file in depth_map_files:
        current_depth_map = np.load(os.path.join(depth_maps_directory, depth_map_file))
        current_point_cloud = depth_map_to_point_cloud(current_depth_map)
        depth_point_cloud.extend(current_point_cloud)

    # Draw the accumulated depth camera point cloud
    for point in depth_point_cloud:
        x, y, _ = point  # Extract x and y coordinates
        pygame.draw.circle(screen, (0, 255, 0), (int(x) + 400, int(y) + 300), 1)

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))
    update_depth_point_cloud()
    clock.tick(60)

pygame.quit()
sys.exit()
