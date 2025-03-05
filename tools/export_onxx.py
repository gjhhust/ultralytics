from ultralytics import YOLO, YOLOFT, YOLOWorld, NAS, SAM, FastSAM, RTDETR

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLOFT("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train28/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", imgsz=869)
