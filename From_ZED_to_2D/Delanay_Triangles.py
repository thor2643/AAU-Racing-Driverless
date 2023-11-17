import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

def generate_racetrack(num_points, track_length, max_variation, point_spacing):
    direction = 'yellow' 
    opposite_direction = 'blue'

    t = np.linspace(0, 2 * np.pi, num_points)
    x = track_length * np.sin(t) + np.random.uniform(-max_variation, max_variation, num_points)
    y = track_length * np.cos(t) + np.random.uniform(-max_variation, max_variation, num_points)

    # Create pairs of points spaced with the specified distance perpendicular to the line between points
    points = []

    for i in range(len(x)):
        if i == len(x) - 1:
            break
        else:
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            dx = dx / dist
            dy = dy / dist

            num_intermediate_points = int(dist / point_spacing)
            if num_intermediate_points < 1:
                num_intermediate_points = 1

            for j in range(num_intermediate_points):
                intermediate_x = x[i] + j * (dx * point_spacing)
                intermediate_y = y[i] + j * (dy * point_spacing)

                # Calculate midpoint of the line segment
                midpoint_x = (x[i] + x[i + 1]) / 2
                midpoint_y = (y[i] + y[i + 1]) / 2

                # Offset the points perpendicular to the line
                offset_x = intermediate_x - midpoint_x
                offset_y = intermediate_y - midpoint_y

                points.extend([[intermediate_x, intermediate_y, direction], [midpoint_x + offset_y, midpoint_y - offset_x, opposite_direction]])

    return points

def delaunay_triangles_filtered(point_array, point_array_without_color):
    
    # Generate the Delaunay triangulation
    tri = Delaunay(point_array_without_color)
    
    # Check if the triangles are made by two points of the same color and add them to the removal list
    triangles_to_remove = []
    for triangle in tri.simplices:
        if (point_array[triangle[0]][2] == point_array[triangle[1]][2] and
            point_array[triangle[0]][2] == point_array[triangle[2]][2]):
            triangles_to_remove.append(triangle)

    # Remove the triangles that are made by two points of the same color
    filtered_triangles = [triangle for triangle in tri.simplices if not any(np.array_equal(triangle, t) for t in triangles_to_remove)]

    return filtered_triangles

def find_midpoints(triangles, point_array):
    midpoints = []

    # Find midpoints of all edges of the triangles
    for triangle in triangles:
        # Find the midpoint of the first edge
        midpoint_x1 = (point_array[triangle[0]][0] + point_array[triangle[1]][0]) / 2
        midpoint_y1 = (point_array[triangle[0]][1] + point_array[triangle[1]][1]) / 2
       
        # Find the midpoint of the second edge
        midpoint_x2 = (point_array[triangle[1]][0] + point_array[triangle[2]][0]) / 2
        midpoint_y2 = (point_array[triangle[1]][1] + point_array[triangle[2]][1]) / 2
        
        # Find the midpoint of the third edge
        midpoint_x3 = (point_array[triangle[2]][0] + point_array[triangle[0]][0]) / 2
        midpoint_y3 = (point_array[triangle[2]][1] + point_array[triangle[0]][1]) / 2

        # Check if the midpoints are between the points of different color, using the third entry of the point array
        if (point_array[triangle[0]][2] != point_array[triangle[1]][2]):
            midpoints.append([midpoint_x1, midpoint_y1])
        if (point_array[triangle[1]][2] != point_array[triangle[2]][2]):
            midpoints.append([midpoint_x2, midpoint_y2])
        if (point_array[triangle[2]][2] != point_array[triangle[0]][2]):
            midpoints.append([midpoint_x3, midpoint_y3])

    return midpoints

def remove_Colors(point_array):
    point_array_without_color = []
    for point in point_array:
        point_array_without_color.append([point[0], point[1]])

    # convert the point array to a NumPy array
    point_array_without_color = np.array(point_array_without_color)

    return point_array_without_color

def plot_points(point_array, point_array_without_color, tri, midpoints):

    # Use matplotlib to plot the points and the triangles found using the Delaunay triangulation
    plt.triplot(point_array_without_color[:, 0], point_array_without_color[:, 1], np.array(tri))
    plt.plot(np.array(midpoints)[:, 0], np.array(midpoints)[:, 1], 'o', color='red')

    for i in range(0, len(point_array), 2):
        x1, y1, color1 = point_array[i]
        x2, y2, color2 = point_array[i + 1]
        plt.plot(x1, y1, 'o', color=color1)
        plt.plot(x2, y2, 'o', color=color2)  # Add a line in the opposite direction
    plt.show()

def main():
    # Generate the racetrack points
    point_array = generate_racetrack(20, 10, 0.1, 5.0)

    # convert the point array to a NumPy array, without color information
    point_array_without_color = remove_Colors(point_array)

    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = delaunay_triangles_filtered(point_array, point_array_without_color)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = find_midpoints(tri, point_array)

    # Plot the points
    plot_points(point_array, point_array_without_color, tri, midpoints)

#main()

