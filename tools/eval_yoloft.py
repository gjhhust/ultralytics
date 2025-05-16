from ultralytics import YOLO, YOLOFT

model = YOLOFT("config/yoloft_xs_ivd/yolov8s_dcn_dy.yaml").load("runs/xs-vid/train74_yolov8+tinyloss_dcn_dy26.2/weights/best.pt")
# model = YOLO("config/yolo/yolov8_S1_DCN_dy.yaml").load("runs/xs-vid/train74_yolov8+tinyloss_dcn_dy26.2/weights/best.pt")

model.info()
# Validate the model
metrics = model.val(data="config/dataset/XS-VIDv2.yaml",
                    cfg="config/train/default.yaml", 
                    batch=1,device=[2],imgsz=1024, 
                    workers=4,
                    half=True,
                    save_json = True)  # no arguments needed, dataset and settings remembered
# print(path)