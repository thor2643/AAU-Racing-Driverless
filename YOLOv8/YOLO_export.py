from ultralytics import YOLO

model = YOLO("YOLO\\best.pt")

model.export(format="onnx")