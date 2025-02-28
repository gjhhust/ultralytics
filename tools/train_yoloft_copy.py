from ultralytics import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
model = YOLOFT("runs/detect/train196/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train25/weights/last.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLOFT("config/yoloft_onxx/yoloftS_v2-ALL-384.yaml").load("yolov8s.pt")

results = model.train(data="config/dataset/Train_6_Test_gaode6.yaml",
                      cfg="config/train/default.yaml", 
                      # batch=3*2, epochs=25, imgsz=896, device=[2,3],workers = 4)
                    batch=3*2, epochs=25, imgsz=896, device=[2,3],workers = 8)
# print("yoloftsv2-C-384_cbam_Focus \nload(./yolov8s.pt)  \nTrain_6_Test_gaode6")
