from ultralytics import YOLO, YOLOFT, YOLOWorld, NAS, SAM, FastSAM, RTDETR

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLOFT("runs/detect/train194/weights/epoch16.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", imgsz=869)
