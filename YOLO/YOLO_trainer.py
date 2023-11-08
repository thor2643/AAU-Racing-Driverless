from ultralytics import YOLO

#Very important that the code is inside __main__
#Causes errors otherwise (something to do with enabling gpu)
if __name__ == '__main__':
    model = YOLO("YOLO\yolov8n.pt")
    results = model.train(data="YOLO\config.yaml", epochs=2)