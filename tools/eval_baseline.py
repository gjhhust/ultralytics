from ultralytics import YOLO

# Load a COCO-pretrained RT-DETR-l model
path = "/data/shuzhengwang/project/ultralytics/runs/detect/train14/weights/last.pt"
model = YOLO(path)  # load a custom model

model.info()
# Validate the model
metrics = model.val(data="config/dataset/Train_6_Test_gaode6.yaml",
                    cfg="config/train/default.yaml", 
                    batch=1,device=[3],imgsz=896, 
                    workers=4,
                    save_json = True)  # no arguments needed, dataset and settings remembered