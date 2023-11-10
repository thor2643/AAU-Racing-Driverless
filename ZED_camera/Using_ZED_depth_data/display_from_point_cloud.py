import pyzed.sl as sl
import os
import numpy as np
import cv2
import struct

run_number = 1
frame_number = 150

data_path="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\ZED_camera\\Recordings_folder"
depth_data_path = os.path.join(data_path,"Run{}".format(run_number))

depth_array=np.load(os.path.join(depth_data_path,"depth_{}.npy".format(frame_number)))
depth_arr=depth_array.copy()


#if depth_arr[:,:,2] is nan or inf then set it to 0 and map everything to 255-0
depth_arr[np.isnan(depth_arr[:,:,2])]=0
depth_arr[np.isinf(depth_arr[:,:,2])]=0
depth_arr[:,:,2]=depth_arr[:,:,2]/np.max(depth_arr[:,:,2])*255



print(depth_arr[:,:,2])

depth_img = np.array(depth_arr[:,:,2], dtype=np.uint8)
cv2.imshow("Depth grey", depth_img)

depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
cv2.imshow("Depth", depth_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img_point_cloud_float=depth_arr[:,:,3]
img_point_cloud=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
img_point_cloud_BGRA=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
for y, row in enumerate(img_point_cloud_float):
    for x, value in enumerate(row):
        img_point_cloud_BGRA[y,x]=struct.unpack('4B',struct.pack('f', img_point_cloud_float[y,x]))
        img_point_cloud[y,x]=[img_point_cloud_BGRA[y,x,2],img_point_cloud_BGRA[y,x,1],img_point_cloud_BGRA[y,x,0],img_point_cloud_BGRA[y,x,3]]

cv2.imshow("Image_point_cloud", cv2.cvtColor(img_point_cloud_BGRA, cv2.COLOR_BGRA2RGBA))
cv2.waitKey(0)
cv2.destroyAllWindows()