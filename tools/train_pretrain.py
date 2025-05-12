from ultralytics import YOLO, YOLOFT


# Load a COCO-pretrained RT-DETR-l model
# model = YOLO("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train70/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
model = YOLO("config/yolo/hyper-yoloS_DCN_dy.yaml").load("hyper-yolos.pt")
# model = YOLO("/root/test/ultralytics/runs/detect/train7/weights/last.pt")

results = model.train(data="config/dataset/coco.yaml",
                      cfg="config/train/default.yaml", 
                      batch=32*4, epochs=200, imgsz=640, device=[0,1,2,3],workers = 6)
