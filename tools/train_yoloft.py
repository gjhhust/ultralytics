from ultralytics import YOLOFT, YOLO

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("/data/shuzhengwang/project/ultralytics/runs/detect/train87/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("runs/detect/train353/weights/last.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLOFT("/data/shuzhengwang/project/ultralytics/runs/detect/train201/weights/last.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
<<<<<<< HEAD
model = YOLO("config/yolo/yolov8_S1_DCN_dy.yaml").load("pretrain/yolov8_S1_DCN_dy/best.pt")
=======
model = YOLO("")
>>>>>>> b0eddede6c78ba9d7143f398c3d138779af23d42

results = model.train(data="config/dataset/XS-VIDv2_stream.yaml",
                      cfg="config/train/default.yaml",
                      # constrained = True,
                      batch=30*2, epochs=25, imgsz=1024, device=[2,3],workers = 4)
                    # batch=6, epochs=25, imgsz=896, device=[2],workers = 6)
# print("yoloftsv2-C-384_cbam_Focus \nload(./yolov8s.pt)  \nTrain_6_Test_gaode6")
