import pygame
from pygame.locals import *
import math

def generate_lidar_data():
    data = []
    for angle_degrees in range(0, 360, 5):
        angle_radians = math.radians(angle_degrees)
        distance = 100 + 50 * math.sin(angle_radians)  # Simulated distance
        data.append((angle_radians, distance))
    return data

def polar_to_cartesian(polar_data):
    cartesian_data = []
    for angle, distance in polar_data:
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        cartesian_data.append((x, y))
    return cartesian_data

def display_point_cloud(screen, point_cloud):
    screen.fill((255, 255, 255))  # Clear the screen
    for x, y in point_cloud:
        pygame.draw.circle(screen, (0, 0, 0), (int(x) + 400, int(y) + 300), 1)  # Draw a black point

def display_robot_position(screen, x, y):
    pygame.draw.circle(screen, (255, 0, 0), (int(x) + 400, int(y) + 300), 5)  # Draw a red circle for the robot

def display_obstacles(screen, obstacles):
    for x, y in obstacles:
        pygame.draw.circle(screen, (255, 0, 0), (int(x) + 400, int(y) + 300), 10)  # Draw red obstacles

def display_trajectory(screen, trajectory, color):
    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i - 1]
        x2, y2 = trajectory[i]
        pygame.draw.line(screen, color, (int(x1) + 400, int(y1) + 300), (int(x2) + 400, int(y2) + 300), 2)

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("LiDAR Point Cloud with Obstacle Course")
clock = pygame.time.Clock()

robot_x = 0  # Starting at the origin
robot_y = 0

planned_trajectory = []
obstacles = [(-50, 0), (-30, 0), (-10, 0), (10, 0), (30, 0), (50, 0)]  # Obstacle positions forming a corridor

running = True
angle = 0  # Initialize the angle
radius = 100  # Radius of the circular trajectory

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    lidar_data = generate_lidar_data()
    point_cloud = polar_to_cartesian(lidar_data)

    # Calculate the next position on the circular trajectory
    angle += math.radians(1)  # Increase the angle (1 degree)
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    robot_x = x
    robot_y = y

    planned_trajectory.append((x, y))

    display_point_cloud(screen, point_cloud)
    display_robot_position(screen, robot_x, robot_y)
    display_obstacles(screen, obstacles)

    display_trajectory(screen, planned_trajectory, (0, 255, 0))

    # Check for collisions with obstacles
    for obstacle_x, obstacle_y in obstacles:
        distance = math.sqrt((robot_x - obstacle_x)**2 + (robot_y - obstacle_y)**2)
        if distance < 15:  # Adjust this threshold for collision detection
            print("Collision detected!")

    pygame.display.update()
    clock.tick(60)

pygame.quit()
