from ultralytics import YOLOFT, YOLO

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("/data/shuzhengwang/project/ultralytics/runs/detect/train87/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLOFT("/data/shuzhengwang/project/ultralytics/runs/detect/train201/weights/last.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
model = YOLOFT("config/yoloft_onxx/yoloftS_v2-ALL-MDC_dysample.yaml").load("runs/save/train227_yoloft_dydcn_newdata/weights/best.pt")

results = model.train(data="config/dataset/Train_6_Test_14569_single.yaml",
                      cfg="config/train/gaode_train_single.yaml",
                      batch=10*2, epochs=25, imgsz=896, device=[0,1],workers = 4)
                    # batch=6, epochs=25, imgsz=896, device=[2],workers = 6)
# print("yoloftsv2-C-384_cbam_Focus \nload(./yolov8s.pt)  \nTrain_6_Test_gaode6")
