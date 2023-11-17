from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("YOLO\\best.pt")

# Open video capture
video_path = "Data_AccelerationTrack\\1\\Color.avi"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Perform object detection
    results = model.predict(source=frame)

    for result in results:
        # Extract bounding box coordinates
        boxes = result.boxes.xyxy

        # Draw bounding boxes on the frame
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


"""
model = YOLO("YOLO\\best.pt")

results = model.predict(source="Data_AccelerationTrack\\1\\Color.avi", classes=[0,1], show=True) # Display preds. Accepts all YOLO predict arguments

for result in results:
    # Detection
    print(result.boxes.cls)
    result.boxes.xyxy   # box with xyxy format, (N, 4)
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf   # confidence score, (N, 1)
    result.boxes.cls    # cls, (N, 1)

    # Segmentation
    result.masks.data      # masks, (N, H, W)
    result.masks.xy        # x,y segments (pixels), List[segment] * N
    result.masks.xyn       # x,y segments (normalized), List[segment] * N

    # Classification
    result.probs     # cls prob, (num_class, )
"""


"""
import numpy as np
import cv2

path = "Data_AccelerationTrack\\1\\Color.avi"

def read_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    print("video_path: "+video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return None

    # Read and discard frames until you reach the desired frame
    for _ in range(frame_number - 1):
        ret, _ = cap.read()
        if not ret:
            # End of the video or an error occurred
            print("Error: Could not read frame", frame_number)
            cap.release()
            return None

    # Read the desired frame
    ret, frame = cap.read()

    if not ret:
        # End of the video or an error occurred
        print("Error: Could not read frame", frame_number)
        cap.release()
        return None

    cap.release()
    return frame

hej = read_frame(path, 7)
cv2.imshow("Hej", hej)
cv2.waitKey()
cv2.destroyAllWindows()
"""