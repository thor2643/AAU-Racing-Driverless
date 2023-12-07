import imutils
import pyzed.sl as sl
import numpy as np
import cv2
import time
import cuda_context
import serial


#Our own modules
from Yolo.yoloDet import YoloTRT
from PathPlanning.Box_to_angle import boxes_to_steering_angle, boxes_to_cone_pos
from PathPlanning.check_for_orange import check_for_orange_cones
from Control.manual_steering import manual_steering



#-------------------- Initialize camera --------------------#

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

print("Zed camera initialises...")

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) : #Ensure the camera has opened succesfully
    exit(-1)

zed_cuda_ctx = cuda_context.PyCudaContext()

print("Initialisation complete!")
print()
# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()



#-------------------- Initialize YOLO --------------------#

print("YOLO model initialises...")
zed_cuda_ctx.pop_ctx()

model = YoloTRT(library="MainCodeFiles/Yolo/libmyplugins.so", engine="MainCodeFiles/Yolo/200_SGD_YOLOv5n.engine", conf=0.5, yolo_ver="v5")

zed_cuda_ctx.push_ctx()

print("YOLO initialisation complete!")
#-------------------- Initialize miscellaneous --------------------#

categories_idxs = {"yellow_cone": 0, "blue_cone": 1, "orange_cone": 2, "large_orange_cone": 3, "unknown_cone": 4}

try:
    ser = serial.Serial("/dev/ttyUSB1", 115200, dsrdtr=None)
except:
    try: 
        ser = serial.Serial("/dev/ttyUSB0", 115200, dsrdtr=None)
    except:
        ser = None

ser.setRTS(False)
ser.setDTR(False)

visualise_output = False

#controls what track are run. True=acceleration. False= Trackdrive
Track = True
#keeps track of number of orange cones
number_of_orange_cones = 0
#number of times two orange cones has to be detected to be considered seeing two orange cones:
orange_seen_in_row = 0
#making it wait after an orange cone is detected
Wait_cone = False
STOP_for_orange = False
firstcone = True

error_cnt = 0
max_error_cnt = 15

servo_angle = 95
speed = 180

#--------------------- Info prints -----------------------#

print()
firstKey = input("Do you want to start driving?[a/t/n], (a=acceleration, t=Trackdrive)")
if firstKey.lower() == "a":
    Track = True
    print(Track)
elif firstKey.lower() == "t":
    Track = False
else:
    print("Quitting")
    quit()

print()
print("To interrupt the program and go to menu: CTRL+C")
time.sleep(3)
print()
print()

print()
print()
print("!!!!!!!!Starts driving!!!!!!!!!!!")
time.sleep(1)



#--------------------- Main loop -----------------------#

while True:
    try:
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


            #Run the yolo prediction
            zed_cuda_ctx.pop_ctx()
            detections, t = model.Inference(image_np, plot_boxes=visualise_output)
            for obj in detections:
                cones.append([[obj['box'][0], obj['box'][1], obj['box'][2], obj['box'][3]], categories_idxs[obj['class']]])
                #print(obj['class'], obj['conf'], obj['box'])
            zed_cuda_ctx.push_ctx()


        #print(cones) #cones er en liste af lister, hvor hver liste indeholder [[x1,y1,x2,y2], type]
        try:
            servo_angle, stering_angle = boxes_to_steering_angle(cones, point_cloud_np, 0.58)
            error_cnt = 0
        except Exception as error:
            error_cnt += 1

            if error_cnt > max_error_cnt:
                raise RuntimeError("Too many faulty path calculations")
            else:
                print(f"\nAn exception was raised\n {error}\n")
                print("No path path was calculated.\nUsing previous path")


        if Wait_cone == True and Track == False and STOP_for_orange == False:
            t3 = time.time()
            if t3 - time_for_last_cone >= 10:
                Wait_cone = False
                firstcone = True

        if Wait_cone == True and STOP_for_orange == True:
            t3 = time.time()
            if t3 - time_for_last_cone >= 3:
                Wait_cone = False
               
        ################## Signe kommer her ##############
        if Wait_cone == False:
            time_for_last_cone = time.time()
            #recieving cone positions and types
            cones_pos_type = boxes_to_cone_pos(cones,point_cloud_np)
            #Checking to see if any orange cones are detected and stops if needed
            number_of_orange_cones, orange_seen_in_row, Wait_cone, STOP_for_orange, firstcone = check_for_orange_cones(cones_pos_type,Track, number_of_orange_cones, ser, servo_angle, orange_seen_in_row, Wait_cone,STOP_for_orange,firstcone)
            
        ###################
        


        if servo_angle != -1:
            ser.write(f"A{int(servo_angle)}V{speed}".encode("utf-8"))
        
        if visualise_output:
            cv2.imshow("Output", image_np)

            key_cv = cv2.waitKey(1)

        t2 = time.time()
        
        print(f"FPS: {1/(t2-t1)}")

        if ser.in_waiting:
            #Få mode fra esp [manual/auto]
            mode = ser.readline().decode('utf-8').rstrip()

            if mode == "manual":
                while True:
                    if ser.in_waiting:
                        msg = ser.readline().decode('utf-8').rstrip()

                        if msg == "auto":
                            break
            elif mode == "stop":
                print("!!! E-STOP !!!")
                break
            elif mode == "auto":
                pass

    except KeyboardInterrupt:
        ser.write("A95V90".encode("utf-8"))
        print()
        print()
        print("Program stopped")
        key = input("Continue[y] or drive manually[m] or quit[q] or emergency stop[s]: ")
        if key == "y":
            #Must push cuda context otherwise an error will occur 
            zed_cuda_ctx.push_ctx()
        elif key == "m":
            ser.write("manual".encode("utf-8"))
            manual_steering(ser)
            ser.write("auto".encode("utf-8"))
        elif key == "s":
            print("!!! E-STOP !!!")
            ser.write("stop".encode("utf-8"))
        elif key == "q":
            break
            
        else:
            break

    except RuntimeError as error:
        ser.write("A95V90".encode("utf-8"))
        print()
        print(error)
        print("\nProgram terminates")
        break
    

ser.write("A95V90".encode("utf-8"))
ser.close()

#Close cv2 windows
cv2.destroyAllWindows()

# Close the camera
zed.close()

