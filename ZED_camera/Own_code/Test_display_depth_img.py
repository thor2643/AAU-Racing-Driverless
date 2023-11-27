import pyzed.sl as sl
import cv2
import numpy as np

# Create a ZED camera object
zed = sl.Camera()
# Set configuration parameters
init_params = sl.InitParameters()
#init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_resolution = sl.RESOLUTION.HD720
#init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 30 

####New code:

init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.MILLIMETER
#image_size = zed.get_camera_information().camera_resolution
image_size= sl.Resolution(1280, 720)


#image_size.width = image_size.width /2
#image_size.height = image_size.height /2
image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
###

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) :
    exit(-1)

# Get camera information (serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: ", zed_serial)

# Capture x frames and stop
frames_to_capture = 1 #x frames
wait_time = 0 #ms
i = 0
image = sl.Mat()
while (i < frames_to_capture) :
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
        
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp at the time the image was captured
        print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),timestamp.get_milliseconds()))
        i = i+1
        #show image
        cv2.imshow("Image", image.get_data())
        cv2.waitKey(wait_time)

         # Retrieve the left image, depth image in the half-resolution
        zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
        # Retrieve the RGBA point cloud in half resolution
        #zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

        # To recover data from sl.Mat to use it with opencv, use the get_data() method
        # It returns a numpy array that can be used as a matrix with opencv
        image_ocv = image_zed.get_data()
        depth_image_ocv = depth_image_zed.get_data()

        cv2.imshow("Image", image_ocv)
        cv2.imshow("Depth", depth_image_ocv)
        cv2.waitKey(wait_time)


# Close the camera
zed.close()
#return 0