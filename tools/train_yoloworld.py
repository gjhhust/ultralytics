from ultralytics import YOLO,  YOLOWorld
import argparse
parser = argparse.ArgumentParser()
# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8s-worldv2.pt")

# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_M.yaml").load("yolov8m.pt") #train yolov8-l
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train8/weights/last.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default_XS-VID.yaml",
                      resume=False,
                      batch=10*2, epochs=7, imgsz=1024, device=[2,3],workers = 8)
# model.set_classes([ "bicycle-static","bicycle-person", "car", "person", "ignore", "truck", "boat"])
# metrics = model.val(data="config/dataset/XS-VIDv2.yaml",
#                     cfg="config/train/default_XS-VID.yaml", 
#                     batch=1,device=[0],imgsz=1024, 
#                     workers=4,
#                     save_json = True)  # no arguments needed, dataset and settings remembered
# print(path)
