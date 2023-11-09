import pyzed.sl as sl
import cv2
import numpy as np
import math
import struct

# Create a ZED camera object
zed = sl.Camera()
# Set configuration parameters
init_params = sl.InitParameters()
#init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_resolution = sl.RESOLUTION.HD720
#init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 30

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) : #Ensure the camera has opened succesfully
    exit(-1)

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()


image = sl.Mat()
depth = sl.Mat()
depth_for_display = sl.Mat()
point_cloud = sl.Mat()

x_cor=200
y_cor=200

# A new image is available if grab() returns SUCCESS
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # Retrieve left image
    zed.retrieve_image(image, sl.VIEW.LEFT)
    #Retrieve depth image. Depth is aligned on the left image
    zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)
    # Retrieve depth map. Depth is aligned on the left image
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    # Retrieve colored point cloud. Point cloud is aligned on the left image.
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)


    point_cloud_np = point_cloud.get_data()
    depth_np = depth.get_data()
    erro, point_cloud_value = point_cloud.get_value(y_cor,x_cor)
    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])

    print("point_cloud_np shape: ", point_cloud_np.shape)
    print("depth_np shape: ", depth_np.shape)
    print("Image shape: ", image.get_data().shape, " Image type: ", image.get_data().dtype)

    print("point_cloud_np: ", point_cloud_np[y_cor,x_cor])
    print("depth_np: ", depth_np[y_cor,x_cor])
    print(f"Cordinats - X: {point_cloud_value[0]}, Y: {point_cloud_value[1]}, Z: {point_cloud_value[2]} \nDistance: {distance}")
    rgba_float = point_cloud_value[3]
    # Pack rgba_float into four bytes using the format 'f', which means a single-precision float
    bytes = struct.pack('f', rgba_float)
    # Convert the bytes to four 8-bit integers using the format '4B', which means four unsigned chars
    rgba_ints = struct.unpack('4B', bytes)
    # Print the RGBA values:
    print("BGR from image: ", image.get_data()[y_cor,x_cor])
    print(f"Pixel colors - B:{rgba_ints[0]}, G:{rgba_ints[1]}, R:{rgba_ints[2]}, A:{rgba_ints[3]}")

    #get all pixels point_cloud_np
    img_point_cloud_float=point_cloud_np[:,:,3]
    print(img_point_cloud_float.shape)
    img_point_cloud=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
    img_point_cloud_BGRA=np.zeros((img_point_cloud_float.shape[0],img_point_cloud_float.shape[1],4), dtype=np.uint8)
    for y, row in enumerate(img_point_cloud_float):
        for x, value in enumerate(row):
            img_point_cloud_BGRA[y,x]=struct.unpack('4B',struct.pack('f', img_point_cloud_float[y,x]))
            img_point_cloud[y,x]=[img_point_cloud_BGRA[y,x,2],img_point_cloud_BGRA[y,x,1],img_point_cloud_BGRA[y,x,0],img_point_cloud_BGRA[y,x,3]]
            
    print(img_point_cloud.shape)
    cv2.imshow("depth", depth_for_display.get_data())
    cv2.imshow("Image", image.get_data())
    #show point_cloud_np as RGBA image
    cv2.imshow("Image_point_cloud", cv2.cvtColor(img_point_cloud_BGRA, cv2.COLOR_BGRA2RGBA))
    #cv2.imshow("Image_point_cloud", img_point_cloud)
    cv2.waitKey(0)
    