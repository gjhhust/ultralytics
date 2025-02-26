from ultralytics import YOLO, YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train61/weights/last.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data="config/dataset/Train_6_Test_gaode6.yaml",
                    cfg="config/train/default.yaml", 
                    batch=1,device=[3],imgsz=896, 
                    workers=4,
                    save_json = True)  # no arguments needed, dataset and settings remembered
print(path)