from ultralytics import YOLOFT

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("runs/detect/train196/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLOFT("runs/detect/train19/weights/last.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
model = YOLOFT("config/yoloft_onxx/yoloftS_seg.yaml").load("pretrain/yolos/yoloftS_coco200e41.3.pt")

results = model.train(data="config/dataset/coco8-seg.yaml",
                      cfg="config/train/gaode_train.yaml",  #修改了数据变换，iou阈值0.5，关闭masic为20epoch
                      # batch=6*2, epochs=35, imgsz=896, device=[2,3],workers = 4)
                    batch=4, epochs=35, imgsz=896, device=[3],workers = 6)
# print("yoloftsv2-C-384_cbam_Focus \nload(./yolov8s.pt)  \nTrain_6_Test_gaode6")
