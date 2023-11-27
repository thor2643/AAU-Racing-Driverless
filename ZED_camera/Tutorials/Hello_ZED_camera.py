import pyzed.sl as sl
import cv2
import numpy as np

# Create a ZED camera object
zed = sl.Camera()
# Set configuration parameters
init_params = sl.InitParameters()
#init_params.camera_resolution = sl.RESOLUTION.HD1080 
init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
init_params.camera_fps = 30 

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) :
    exit(-1)

# Get camera information (serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: ", zed_serial)

# Capture 50 frames and stop
i = 0
image = sl.Mat()
while (i < 50) :
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp at the time the image was captured
        print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),timestamp.get_milliseconds()))
        i = i+1
        #show image
        cv2.imshow("Image", image.get_data())
        cv2.waitKey(33)



# Close the camera
zed.close()
#return 0