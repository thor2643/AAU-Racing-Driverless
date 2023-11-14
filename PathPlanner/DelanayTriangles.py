import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from BW_Alg import Point, bowyer_watson
from scipy import interpolate

generation_time = 2

def generate_racetrack(num_points, track_length, max_variation, point_spacing):
    direction = 'yellow' 
    opposite_direction = 'blue'

    t = np.linspace(0, 2 * np.pi, num_points)
    x = track_length * np.sin(t) + np.random.uniform(-max_variation, max_variation, num_points)
    y = track_length * np.cos(t) + np.random.uniform(-max_variation, max_variation, num_points)

    # All the the points should be positive in the coordinates. If not, they will be shifted to the positive quadrant
    if min(x) < 0:
        x = x - min(x)
    if min(y) < 0:
        y = y - min(y)

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

def delaunay_triangles_filtered(point_array, point_array_without_color, use_scipy=True):
    if use_scipy:
        # Generate the Delaunay triangulation using scipy.spatial.Delaunay
        tri = Delaunay(point_array_without_color)
        
        # Check if the triangles are made by two points of the same color and add them to the removal list
        triangles_to_remove = []
        for triangle in tri.simplices:
            if (point_array[triangle[0]][2] == point_array[triangle[1]][2] and
                point_array[triangle[0]][2] == point_array[triangle[2]][2]):
                triangles_to_remove.append(triangle)

        # Remove the triangles that are made by two points of the same color
        filtered_triangles = [triangle for triangle in tri.simplices if not any(np.array_equal(triangle, t) for t in triangles_to_remove)]
    else:
         # Generate the Delaunay triangulation using the Bowyer-Watson algorithm
        point_objects = [Point(point[0], point[1]) for point in point_array_without_color]
        tri = bowyer_watson(point_objects)
        
        # Check if the triangles are made by two points of the same color and add them to the removal list
        triangles_to_remove = []
        for triangle in tri:
            p1, p2, p3 = triangle.vertices
            if (point_array[p1][2] == point_array[p2][2] and
                point_array[p1][2] == point_array[p3][2]):
                triangles_to_remove.append(triangle)

        # Remove the triangles that are made by two points of the same color
        filtered_triangles = [triangle for triangle in tri if triangle not in triangles_to_remove]

    return filtered_triangles

def find_midpoints(triangles, point_array):
    midpoints = []

    # Find midpoints of all edges of the triangles
    for triangle in triangles:
        for i in range(3):
            j = (i + 1) % 3  # Calculate the next vertex index in a circular manner
            if point_array[triangle[i]][2] != point_array[triangle[j]][2]:
                midpoint_x = (point_array[triangle[i]][0] + point_array[triangle[j]][0]) / 2
                midpoint_y = (point_array[triangle[i]][1] + point_array[triangle[j]][1]) / 2
                midpoints.append([midpoint_x, midpoint_y])

    #remove duplicates
    midpoints = list(set(map(tuple, midpoints)))
    midpoints = [list(elem) for elem in midpoints]

    # Find the point closest to the origin
    distances = [np.sqrt(point[0] ** 2 + point[1] ** 2) for point in midpoints]
    leftmost_index = np.argmin(distances)
    leftmost_point = midpoints[leftmost_index]
    

    # Create a list to store the sorted midpoints and start with the leftmost point, and remove it from midpoints
    sorted_midpoints = [leftmost_point]
    midpoints.remove(leftmost_point)

    while midpoints:
        last_point = sorted_midpoints[-1]
        distances = [np.sqrt((point[0] - last_point[0]) ** 2 + (point[1] - last_point[1]) ** 2) for point in midpoints]
        
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        
        # Add the closest point to the sorted list and remove it from midpoints
        sorted_midpoints.append(midpoints.pop(closest_index))

    # Replace midpoints with the sorted list
    midpoints = sorted_midpoints

    # Check if the last midpoint is two distances away from the last point and remove it if it is
    if len(midpoints) > 2:
        last_point = midpoints[-1]
        second_last_point = midpoints[-2]
        Third_last_point = midpoints[-3]
        distance_Last_to_second = np.sqrt((last_point[0] - second_last_point[0]) ** 2 + (last_point[1] - second_last_point[1]) ** 2)
        distance_second_to_third = np.sqrt((second_last_point[0] - Third_last_point[0]) ** 2 + (second_last_point[1] - Third_last_point[1]) ** 2)
        if (distance_Last_to_second < 2 * distance_second_to_third)!= True:
            midpoints.pop(-1)

    return midpoints

def remove_Colors(point_array):
    point_array_without_color = []
    for point in point_array:
        point_array_without_color.append([point[0], point[1]])

    # convert the point array to a NumPy array
    point_array_without_color = np.array(point_array_without_color)

    return point_array_without_color

def plot_points(point_array, point_array_without_color, tri, midpoints):
    # Check if tri is empty
    if not tri:
        print("No triangles found")
    else:
        # Use matplotlib to plot the points and the triangles found using the Delaunay triangulation
        plt.triplot(point_array_without_color[:, 0], point_array_without_color[:, 1], np.array(tri))
        
        # Plot the trajectory
        trajectory_planning(midpoints)

        # Plot the midpoints in red, but with the number of which the point comes in the list
        for i in range(len(midpoints)):
            plt.plot(midpoints[i][0], midpoints[i][1], 'o', color='red')
            plt.text(midpoints[i][0], midpoints[i][1], str(i))

        for i in range(0, len(point_array), 2):
            # Check if the last point is reached
            x1, y1, color1 = point_array[i]
            plt.plot(x1, y1, 'o', color=color1)
            if i + 1 < len(point_array):
                x2, y2, color2 = point_array[i + 1]
                plt.plot(x2, y2, 'o', color=color2)  # Add a line in the opposite direction
            
            
        plt.show(block=False)

        # Close the plot windows afte 1 second
        plt.pause(generation_time)
        plt.close("all")
        
def trajectory_planning(point_array):
    # Separate x and y coordinates
    x_coords = [point[0] for point in point_array]
    y_coords = [point[1] for point in point_array]

    # Add the first point to the end of the list to make a closed loop
    # x_coords.append(x_coords[0])
    # y_coords.append(y_coords[0])

    # Use cubic spline interpolation to find the trajectory between the points
    tck, u = interpolate.splprep([x_coords, y_coords], s=0, k=1)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, len(point_array)), tck)

    print("x_i: ", x_i)
    print("y_i: ", y_i)

    # Plot the trajectory
    plt.plot(x_i, y_i, color='DarkGreen')

def progresive_triangulation():
    # Generate the cones
    cones = generate_racetrack(20, 10, 0.1, 5.0)

    # Sorting the points. First the point closest to the origin is found
    distances = [np.sqrt(point[0] ** 2 + point[1] ** 2) for point in cones]
    leftmost_index = np.argmin(distances)

    # Sort the cones based on the leftmost index
    sorted_cones = [] 
    for cone_index in range(len(cones) - leftmost_index):
        sorted_cones.append(cones.pop(leftmost_index))

    # Add the rest of the cones to the sorted list. Go from the end of the list to the beginning
    for cone_index in range(len(cones)):
        sorted_cones.append(cones.pop(0))

    # Convert the point array to a NumPy array, without color information
    point_array_without_color = remove_Colors(sorted_cones)

    # Take the first four cones initially
    initial_cones = sorted_cones[:4]
    sorted_cones = sorted_cones[4:]

    cones_discovered = []
    cones_discovered_without_color = []

    # Add the initial cones to the discovered cones and without color list
    cones_discovered.extend(initial_cones)
    cones_discovered_without_color.extend(point_array_without_color[:4])
    point_array_without_color = point_array_without_color[4:]

    while sorted_cones:
            for i in range(np.random.randint(2, 5)):
                # Check if the list is empty
                if sorted_cones == []:
                    break
                else:   

                    if (type(cones_discovered_without_color)) == np.ndarray:
                        # Convert the point array to a list
                        cones_discovered_without_color = cones_discovered_without_color.tolist()

                    cones_discovered.append(sorted_cones.pop(0))
                    cones_discovered_without_color.append(point_array_without_color[0])
                    point_array_without_color = point_array_without_color[1:]

            # convert the point array to a NumPy array, without color information
            cones_discovered_without_color = np.array(cones_discovered_without_color)

            # Use the filter function to remove the triangles that are made by three points of the same color
            tri = delaunay_triangles_filtered(cones_discovered, cones_discovered_without_color, use_scipy=True)

            # Find the midpoints of the triangles that are made by two points of different color
            midpoints = find_midpoints(tri, cones_discovered)

            # Plot the points
            plot_points(cones_discovered, cones_discovered_without_color, tri, midpoints)

def batch_triangulation():
    # Generate the racetrack points
    point_array = generate_racetrack(20, 10, 1, 5.0)

    # convert the point array to a NumPy array, without color information
    point_array_without_color = remove_Colors(point_array)

    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = delaunay_triangles_filtered(point_array, point_array_without_color, use_scipy=True)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = find_midpoints(tri, point_array)

    # Plot the points
    plot_points(point_array, point_array_without_color, tri, midpoints)

def main(triangulation_method = "progresive"):
    if triangulation_method == "batch":
        batch_triangulation()
    elif triangulation_method == "progresive":
        progresive_triangulation()
    else:
        print("Triangulation method not recognized")

main("progresive")

