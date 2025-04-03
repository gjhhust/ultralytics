from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()

# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLO("yolo11x.pt") #train yolov8-l
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train8/weights/last.pt")

results = model.train(data="config/dataset/XS-VIDv2-all.yaml",
                      cfg="config/train/default.yaml",
                      batch=8*4, epochs=50, imgsz=1024, device=[0,1,2,3],workers = 8)
