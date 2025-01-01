from ultralytics import YOLO
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("/data/shuzhengwang/project/ultralytics/runs/detect/train32/weights/last.pt")
# model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train25/weights/last.pt")

results = model.train(data="config/dataset/Train_6_Test_gaode6.yaml",
                      cfg="config/train/default.yaml", 
                      batch=16*2, epochs=7, imgsz=896, device=[2,3],workers = 8)
