from ultralytics import YOLO, YOLOFT, YOLOWorld, NAS, SAM, FastSAM, RTDETR

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLOFT("/data/shuzhengwang/project/ultralytics/runs/detect/train82/weights/last.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", imgsz=869)
