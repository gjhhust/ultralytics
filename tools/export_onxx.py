from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train15/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", dynamic=True, simplify=True)
