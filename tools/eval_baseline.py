from ultralytics import YOLO, YOLOFT
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='YOLOFT')
# parser.add_argument("--model_wight", type=str, default='')
parser.add_argument("--dataset", type=str, default='config/dataset/Train_6_Test_task1_2videos.yaml')
args = parser.parse_args()
# Load a COCO-pretrained RT-DETR-l model
# path = "/data/shuzhengwang/project/ultralytics/runs/save/train201_DCN_32.9/weights/best.pt"
if args.model_type == 'YOLOFT':
    path = "runs/save/train227_yoloft_dydcn_newdata/weights/best.pt"
    model = YOLOFT(path)  # load a custom model"
    divice_id = 0
else:
    path = "/data/shuzhengwang/project/ultralytics/runs/save/train230_yolo_dydcn_notall_newdata/weights/best.pt"
    model = YOLO(path)  # load a custom model"
    divice_id = 1

model.info()
# Validate the model
metrics = model.val(data=args.dataset,
                    cfg="config/train/gaode_train.yaml", 
                    batch=1,device=[divice_id],imgsz=896, 
                    workers=4,
                    save_json = True)  # no arguments needed, dataset and settings remembered
print(path)