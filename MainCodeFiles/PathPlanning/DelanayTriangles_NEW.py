import numpy as np
from scipy.spatial import Delaunay
from PathPlanning.BW_Alg_copy import Point, bowyer_watson

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



