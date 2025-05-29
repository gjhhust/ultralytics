from ultralytics import YOLOFT, YOLO

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train196/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
model = YOLOFT("config/yoloft_xs_ivd/yolov8s_ftv1_dcn_dy.yaml").load("runs/xs-vid/pretrain_e200best_yolov8s_ftv1_dcn_dy.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLOFT("config/yolo_time/yolov8s_ftv1_dim384_3d.yaml").load("pretrain/yolov8s_ftv1_dim384/yolov8s_ftv1_dim384.pt")
# model = YOLOFT("runs/save/train62_yolofts_dy_dcns1/weights/best.pt")

results = model.train(data="config/dataset/XS-VIDv2.yaml",
                      cfg="config/train/default.yaml",
                      batch=12*2, epochs=25, imgsz=1024, device=[2,3],workers=4)

# results = model.train(data="config/dataset/Train_5&6_Test_14569_single.yaml",
#                       cfg="config/train/gaode_train_single.yaml",
#                       batch=34*2, epochs=28, imgsz=896, device=[0,1],workers = 6)
print("yolov8s_ftv1_dcn_dy  maskloss 1.0(lpixl + larea + 20*ldist)  nc=1  prev_value=1.0(Y), crop")
  