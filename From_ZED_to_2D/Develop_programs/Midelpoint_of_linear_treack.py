#import Delanay_Triangles_copy as DT
import DelanayTriangles_copy_2 as DT
import numpy as np

#generate points for linear track with 8 cones

#perameters:
number_of_cones=8
y_start=0
x_start=3
dist_between_cones_y=3
dist_between_cones_x=3
diff_y=0.5
diff_x=0.2


y_cor_1=[]
y_cor_2=[]
for i in range(number_of_cones):
    y_cor_1.append(i*dist_between_cones_y+np.random.uniform(-diff_y, diff_y))
    y_cor_2.append(i*dist_between_cones_y+np.random.uniform(-diff_y, diff_y))

print(f"y_cor_1={y_cor_1}, y_cor_2={y_cor_2}")




yellow_points=np.array([np.random.uniform(x_start-diff_x, x_start+diff_x, number_of_cones),y_cor_1]).T
blue_points=np.array([np.random.uniform(x_start+dist_between_cones_x-diff_x, x_start+dist_between_cones_x+diff_x, number_of_cones),y_cor_2]).T
#points_=np.array([yellow_points,blue_points]).T
#print(points_)

print(f"yellow_points=\n{yellow_points}")
print(f"blue_points=\n{blue_points}")

#make a list of all points
points=[]
for i in range(number_of_cones):
    points.append([yellow_points[i,0],yellow_points[i,1],'yellow'])
    points.append([blue_points[i,0],blue_points[i,1],'blue'])

print(f"points=\n{points}")

points

def plot_points_with_delanay(point_array):
    # convert the point array to a NumPy array, without color information
    point_array_without_color = DT.remove_Colors(point_array)
    print(point_array_without_color)


    # Use the filter function to remove the triangles that are made by three points of the same color
    tri = DT.delaunay_triangles_filtered(point_array, point_array_without_color)
    print(tri)

    # Find the midpoints of the triangles that are made by two points of different color
    midpoints = DT.find_midpoints(tri, point_array)

    # Plot the points
    DT.plot_points(point_array, point_array_without_color, tri, midpoints)




plot_points_with_delanay(points)


#DT.main()


