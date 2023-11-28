import imutils
import pyzed.sl as sl
import numpy as np
import cv2
import time
import cuda_context


#Our own modules
from Yolo.yoloDet import YoloTRT


#--------------------Initialize camera--------------------#

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA #.HD720
#init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
#init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.depth_stabilization = 0
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 1
init_params.camera_fps = 30



#Initialize empty matrices where data can be written to
image = sl.Mat()
point_cloud = sl.Mat()

print("cam initialises")

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) : #Ensure the camera has opened succesfully
    exit(-1)

zed_cuda_ctx = cuda_context.PyCudaContext()

print("cam open")
# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()



#--------------------Initialize YOLO--------------------#

print("yolo initialises")
zed_cuda_ctx.pop_ctx()

model = YoloTRT(library="MainCodeFiles/Yolo/libmyplugins.so", engine="MainCodeFiles/Yolo/yolov5n.engine", conf=0.5, yolo_ver="v5")

zed_cuda_ctx.push_ctx()


#--------------------Initialize miscellaneous--------------------#

categories_idxs = {"yellow_cone": 0, "blue_cone": 1, "orange_cone": 2, "large_orange_cone": 3, "unknown_cone": 4}


visualise_output = False

print("ready to loop")



#---------------------Main loop-----------------------#

#while True:
for i in range(50):
    cones = [] 
    t1 = time.time()
    # A new image is available if grab() returns SUCCESS
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)

        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) #, sl.MEM.GPU, 640, 360)
        #zed.retrieve_measure(point_cloud, sl.MEASURE.DEPTH)

        # Get the numpy array from the sl.Mat object
        image_np = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

        point_cloud_np = point_cloud.get_data()

        zed_cuda_ctx.pop_ctx()
        detections, t = model.Inference(image_np, plot_boxes=visualise_output)
        for obj in detections:
            cones.append([categories_idxs[obj['class']], obj['box']])
            #print(obj['class'], obj['conf'], obj['box'])
        zed_cuda_ctx.push_ctx()


        #----------Emil kode placeres her------------#
        
        print(cones)





    if visualise_output:
        cv2.imshow("Output", image_np)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        else:
            pass

    t2 = time.time()
    
    print(f"FPS: {1/(t2-t1)}")
    

#Close cv2 windows
cv2.destroyAllWindows()

# Close the camera
zed.close()










