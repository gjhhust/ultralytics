from ultralytics import YOLO


# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLO("config/yolo/yolov8_L.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train15/weights/epoch2.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default.yaml", 
                      batch=12*3, epochs=20, imgsz=1024, device=[0,1,2,3],workers = 10)
