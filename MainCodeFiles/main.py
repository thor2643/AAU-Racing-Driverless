import imutils
import pyzed.sl as sl
import numpy as np
import cv2

#Our own modules
from Yolo.yoloDet import YoloTRT


#--------------------Initialize YOLO--------------------#

model = YoloTRT(library="libmyplugins.so", engine="yolov5n.engine", conf=0.5, yolo_ver="v5")


#--------------------Initialize camera--------------------#

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
frame_rate = 30
init_params.camera_fps = frame_rate

#Initialize empty matrices where data can be written to
image = sl.Mat()
point_cloud = sl.Mat()

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) : #Ensure the camera has opened succesfully
    exit(-1)

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()


#---------------------Main loop-----------------------#

#while True:
for i in range(120):
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Get the numpy array from the sl.Mat object
        image_np = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        point_cloud_np = point_cloud.get_data()

        detections, t = model.Inference(image_np)
        #for obj in detections:
            #print(obj['class'], obj['conf'], obj['box'])
        print("FPS: {} sec".format(1/t))

    cv2.imshow("Output", image_np)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    else:
        continue



#Close cv2 windows
cv2.destroyAllWindows()

# Close the camera
zed.close()










