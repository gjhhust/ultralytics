from ultralytics import YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_L.yaml").load("yolov8l.pt") #train yolov8-l
model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train8/weights/last.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default.yaml", 
                      batch=16, epochs=20, imgsz=1024, device=[3],workers = 2)
