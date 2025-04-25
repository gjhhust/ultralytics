from ultralytics import YOLOFT, YOLO

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train196/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
model = YOLOFT("config/yolo_time/yolov8s_ftv1_dim384_3d.yaml").load("pretrain/yolov8s_ftv1_dim384/yolov8s_ftv1_dim384.pt")
# model = YOLOFT("runs/save/train62_yolofts_dy_dcns1/weights/best.pt")

results = model.train(data="config/dataset/Train_6_Test_14569_single.yaml",
                      cfg="config/train/gaode_train_single.yaml",
                      batch=16*2, epochs=25, imgsz=896, device=[0,1],workers = 4)

# results = model.train(data="config/dataset/Train_5&6_Test_14569_single.yaml",
#                       cfg="config/train/gaode_train_single.yaml",
#                       batch=34*2, epochs=28, imgsz=896, device=[0,1],workers = 6)
print("yoloftS_dcn_dy_s1")
  