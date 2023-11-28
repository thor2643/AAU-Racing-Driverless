

def plot_points_with_delanay(point_array):
    # convert the point array to a NumPy array, without color information
    point_array_without_color = DT.remove_Colors(point_array)
    #print(point_array_without_color)


    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = DT.delaunay_triangles_filtered(point_array, point_array_without_color)
    #print(tri)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, point_array)

    # Plot the points
    #DT.plot_points(point_array, point_array_without_color, tri, midpoints)

    return midpoints