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
depth_arr_crop=depth_arr[450:,275:-275]

arr_2=depth_arr_crop.copy()

###############
#Make image from point cloud:
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
#############

#indexes of all non nan values
arr_1_indexes=np.argwhere(np.isnan(arr_2[:,:,2])==False)

