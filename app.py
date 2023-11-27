import sys
import cv2 
import imutils
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="libmyplugins.so", engine="yolov5n.engine", conf=0.5, yolo_ver="v5")

cap = cv2.VideoCapture("Data_AccelerationTrack/1/Color.avi")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t = model.Inference(frame)
    #for obj in detections:
        #print(obj['class'], obj['conf'], obj['box'])
    print("FPS: {} sec".format(1/t))
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
