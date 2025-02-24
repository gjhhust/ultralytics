from ultralytics import YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLO("config/yolo/yolov8_L.yaml").load("yolov8l.pt") #train yolov8-l
# model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train15/weights/epoch2.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default.yaml", 
                      batch=6*2, epochs=7, imgsz=1920, device=[2,3],workers = 6)
