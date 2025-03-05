from ultralytics import YOLO, YOLOFT

# Load a COCO-pretrained RT-DETR-l model
path = "/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/detect/train22/weights/last.pt"
model = YOLOFT(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data="config/dataset/XS-VIDv2.yaml",
                    cfg="config/train/default.yaml", 
                    batch=36,device=[2],imgsz=1024, 
                    workers=4,
                    save_json = True)  # no arguments needed, dataset and settings remembered
print(path)