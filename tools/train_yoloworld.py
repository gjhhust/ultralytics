from ultralytics import YOLO,  YOLOWorld
import argparse
parser = argparse.ArgumentParser()
# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8l-worldv2.pt")

# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_M.yaml").load("yolov8m.pt") #train yolov8-l
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train8/weights/last.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default_XS-VID.yaml",
                      resume=False,
                      batch=10, epochs=7, imgsz=1024, device=[1],workers = 4)
