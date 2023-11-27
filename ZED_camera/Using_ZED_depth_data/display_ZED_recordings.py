import pyzed.sl as sl
import os
import numpy as np
import cv2
import time

run_number = 1
frame_number = 0

data_path="C:\\Users\\3103e\\Documents\\GitHub\\AAU-Racing-Driverless\\ZED_camera\\Recordings_folder"
depth_data_path = os.path.join(data_path,"Run{}".format(run_number))
video_path = os.path.join(data_path,"ZED_color_video_Run_{}.avi".format(run_number))



depth_arr=np.load(os.path.join(depth_data_path,"depth_{}.npy".format(frame_number)))
print(depth_arr.shape)


#create a function that displays the .avi video form video_path with 30 fps
def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            #time diffrence in ms between frames 33.3333_ ms = 1/30 s
            time.sleep(1/30-0.001)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

#creat a function that displays the one frame from the .avi video form video_path
def display_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(1,frame_number)
    ret, frame = cap.read()
    cv2.imshow('Frame',frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

display_frame(video_path, frame_number)
display_video(video_path)