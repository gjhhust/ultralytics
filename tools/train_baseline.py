from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()

# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLO("config/yolo/yolov8_M.yaml").load("yolov8m.pt") #train yolov8-l
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train8/weights/last.pt")

results = model.train(data="config/dataset/Train_6_Test_14569_single.yaml",
                      cfg="config/train/gaode_train_single.yaml",
                      batch=10*2, epochs=12, imgsz=896, device=[0,1],workers = 4)
