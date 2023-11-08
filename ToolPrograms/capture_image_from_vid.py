import cv2

#Open the video file
video_path = 'Videos\\ZED\\ZED_color_video_Run_2.avi'  # Replace with your video file path
output_path = "Images\OwnData"

cap = cv2.VideoCapture(video_path)
i = 12
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    if i % 30 == 0:
        frame_filename = output_path + "\\" + f'frame{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg'
        cv2.imwrite(frame_filename, frame)

    i += 1

    

#Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()