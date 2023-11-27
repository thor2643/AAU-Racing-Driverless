import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct

run_number = 1
frame_number = 150

data_path="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\ZED_camera\\Recordings_folder"
depth_data_path = os.path.join(data_path,"Run{}".format(run_number))

depth_arr=np.load(os.path.join(depth_data_path,"depth_{}.npy".format(frame_number)))


#depth_arr_crop=depth_arr[420:,200:-200]
#depth_arr_crop=depth_arr[420:-50,200:400]
depth_arr_crop=depth_arr[-200:-150,50:-50]

arr_2=depth_arr_crop.copy()

img_point_cloud_float=depth_arr_crop[:,:,3]
img_point_cloud=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
img_point_cloud_BGRA=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
for y, row in enumerate(img_point_cloud_float):
    for x, value in enumerate(row):
        img_point_cloud_BGRA[y,x]=struct.unpack('4B',struct.pack('f', img_point_cloud_float[y,x]))
        img_point_cloud[y,x]=[img_point_cloud_BGRA[y,x,2],img_point_cloud_BGRA[y,x,1],img_point_cloud_BGRA[y,x,0],img_point_cloud_BGRA[y,x,3]]

cv2.imshow("Image_point_cloud", cv2.cvtColor(img_point_cloud_BGRA, cv2.COLOR_BGRA2RGBA))
cv2.waitKey(0)
cv2.destroyAllWindows()

#indexes of all non nan values
arr_1_indexes=np.argwhere(np.isnan(arr_2[:,:,2])==False)



#emty numpy array with 3 columns

#arr_1_values_list=np.empty((arr_1_indexes.shape[0],3))
x_cor=np.array([])
y_cor=np.array([])
z_cor=np.array([])

for i,index in enumerate(arr_1_indexes):
    x_cor=np.append(x_cor,arr_2[index[0],index[1],0])
    y_cor=np.append(y_cor,arr_2[index[0],index[1],1])
    z_cor=np.append(z_cor,arr_2[index[0],index[1],2])

    #arr_1_values_list=[np.append(arr_1_values_list[0],[arr_2[index[0],index[1],0]]),np.append(arr_1_values_list[1],[arr_2[index[0],index[1],1]]),np.append(arr_1_values_list[2],[arr_2[index[0],index[1],2]])]


print(arr_2[arr_1_indexes[0][0],arr_1_indexes[0][1],:3])
print([x_cor[0],y_cor[0],z_cor[0]])



print("hej")
#arr_1_indexes=np.argwhere(np.isnan(arr_2[:,:,2]))

print(arr_1_indexes.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_cor, y_cor, z_cor, c='r', marker='o')
ax.set_xlabel('X_cor')
ax.set_ylabel('Y_cor')
ax.set_zlabel('Z_cor')


import pyransac3d as pyrsc
import pandas as pd
points=np.array([x_cor,y_cor,z_cor]).T
print(points.shape)

plane1 = pyrsc.Plane()
best_eq, best_inliers = plane1.fit(points,thresh=0.01, minPoints=10000, maxIteration=10000)#thresh=0.01, minPoints=100, maxIteration=1000

A,B,C,D=best_eq[0],best_eq[1],best_eq[2],best_eq[3]


#print(f"A: {A}, B: {B}, C: {C}, D: {D}")
print(f"best_eq: {best_eq}, best_inliers: {best_inliers}")

#plot plane
y_surf,x_surf = np.meshgrid(np.linspace(y_cor.min(), y_cor.max(), 100),np.linspace(x_cor.min(), x_cor.max(), 100))
Z = (D - A*x_surf - B*y_surf) / C
ax.plot_surface(x_surf, y_surf, Z, color="blue", alpha=0.5, linewidth=0)
plt.show()
"""
# fit plane using RANSAC algorithm
plane, inliers = pyrsc.plane_fitting.ransac(
    x_cor,
    y_cor,
    z_cor,
    threshold=0.01,
    n_iterations=1000,
    n_points=3,
    return_all=True)

print(plane)"""