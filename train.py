from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(data='data.yaml', epochs=20, imgsz=640, batch=25)