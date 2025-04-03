from ultralytics import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train196/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
model = YOLOFT("config/yoloft_onxx/yoloftS_dy.yaml").load("pretrain/yolos/yoloftS_coco200e41.3.pt")
# model = YOLOFT("config/yoloft_onxx/yoloftM_dy.yaml").load("pretrain/yolosft_dcns1_dy/best.pt")
# model = YOLOFT("runs/save/train62_yolofts_dy_dcns1/weights/best.pt")

results = model.train(data="config/dataset/Train_6_Test_gaode6.yaml",
                      cfg="config/train/gaode_train_single.yaml",
                      batch=10, epochs=30, imgsz=896, device=[0,1],workers = 6)
print("yoloftS_dy")
