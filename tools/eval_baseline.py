from ultralytics import YOLO, YOLOFT
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default='YOLO')
parser.add_argument("--model_wight", type=str, default='runs/save/train59_YOLOv8S_Dcn_dy_gaode/weights/best.pt')
parser.add_argument("--dataset", type=str, default='config/dataset/Train_6_Test_task1.yaml')
args = parser.parse_args()
# Load a COCO-pretrained RT-DETR-l model
# path = "/data/shuzhengwang/project/ultralytics/runs/save/train201_DCN_32.9/weights/best.pt"
if args.model_type == 'YOLOFT':
    path = args.model_wight
    model = YOLOFT(path)  # load a custom model"
    # model = YOLOFT("config/yoloft_onxx/yoloftS_dcn_dy_s1.yaml").load()
    divice_id = 3
else:
    path = "runs/save/train59_YOLOv8S_Dcn_dy_gaode/weights/best.pt"
    model = YOLO(path)  # load a custom model"
    divice_id = 3

model.info()
# Validate the model
metrics = model.val(data=args.dataset,
                    cfg="config/train/default.yaml", 
                    batch=1,device=[divice_id],imgsz=896, 
                    workers=4,
                    half=True,
                    save_json = True)  # no arguments needed, dataset and settings remembered
print(path)