from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='Dataset\SplitData\data.yaml', epochs = 3)