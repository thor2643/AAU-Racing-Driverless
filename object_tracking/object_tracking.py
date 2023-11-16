import cv2
import numpy as np
import os

def load():
    #load video from folder:
    video_folder = "Data_AccelerationTrack//1//Color.avi"
    cap = cv2.VideoCapture(video_folder)
    L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std = finds_LAB_reference_from_folder("Images//Color_transfer")
    time1 = 0
    
    while True:
        # Read the frames of the video
        _ , frame = cap.read()     
        #process the frames:
        #frame = color_transfer(frame, L_s_mean, L_s_std, A_s_mean, A_s_std, B_s_mean, B_s_std)
        frame_yellow = color_enhancement(frame)
        y = remove_all_but_concrete(frame)
        frame_blue = frame_yellow.copy()
        frame_yellow = find_yellow(frame_yellow)
        frame_blue = find_blue(frame_blue)
        frame = cv2.add(frame_yellow, frame_blue)
        frame, cone_cordinates = template_matching(frame, y)
        time1 =time.time()
        
        #show the frames:
        cv2.imshow("Video", frame)
    
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()