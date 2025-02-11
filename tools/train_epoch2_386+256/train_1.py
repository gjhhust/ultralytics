from ultralytics import YOLO
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load a COCO-pretrained RT-DETR-l model
# model = YOLOFT("yoloft/train113/weights/last.pt") #resume
# model = YOLOFT("config/yoloft_dev/yoloftv2-C_DCN-L.yaml").load("yolov8l.pt") #train yoloft-l
# model = YOLO("config/yolo/yolov8_S.yaml").load("yolov8s.pt") #train yolov8-l
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
# model = YOLO("/data/shuzhengwang/project/ultralytics/runs/detect/train25/weights/last.pt")
# model = YOLO("config/yoloft/yoloftsv2-C-384.yaml").load("./yoloft-C-384_obj365_e7.pt")
model = YOLO("config/yoloft/yoloftsv2-ALL-384+256_cbam_Focus.yaml").load("yolov8s.pt")

results = model.train(data="config/dataset/Train_6_Test_gaode6.yaml",
                      cfg="config/train/default.yaml", 
                      batch=18*2, epochs=25, imgsz=896, device=[0,1],workers = 6)
print("yoloftsv2-ALL-384_cbam_Focus 768 \nload(./yolov8s.pt)  \nTrain_6_Test_gaode6")
