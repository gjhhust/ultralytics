export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

conda activate ultr_new

pip install yolox

1. base YOLOV8
8.1 13.0 8.4 0.4 10.2 22.8 3.0 31.6 25.5


2. base yloft 
29.4 80.2 11.2 12.4 21.5 42.8 11.8 -100.0 -100.0
/data/shuzhengwang/project/ultralytics/runs/detect/train25

3. 
24.2 70.9 6.7 10.2 17.3 10.7 6.8 -100.0 -100.0


# 1
119 211 4
['/data/jiahaoguo/dataset/gaode_6/images/73/0001184.jpg', '/data/jiahaoguo/dataset/gaode_6/images/211/0000680.jpg', '/data/jiahaoguo/dataset/gaode_6/images/4/0000500.jpg']   
['/data/jiahaoguo/dataset/gaode_6/images/13/0000104.jpg', '/data/jiahaoguo/dataset/gaode_6/images/238/0001148.jpg', '/data/jiahaoguo/dataset/gaode_6/images/233/0000608.jpg'] 
## special case 
 video:198 frame_number:354

# 2
video_names: ['26', '261', '60', '194', '253', '268', '167', '232', '185', '110', '65', '198', '197', '207', '147', '20', '89', '221', '40', '4']
frame_numbers: [868, 96, 128, 1076, 1166, 684, 1096, 320, 946, 908, 1054, 354, 464, 1324, 932, 850, 778, 164, 916, 1010]
gt_labels shape: torch.Size([20, 3031, 1])
gt_bboxes shape: torch.Size([20, 3031, 4])


# 3
video_names: ['233', '152', '24', '154', '17', '69', '22', '87', '145', '112', '141', '42', '198', '5', '159', '223', '55', '181', '167', '129']
frame_numbers: [1196, 298, 892, 1242, 904, 368, 144, 982, 622, 828, 1044, 208, 354, 88, 646, 1330, 940, 830, 78, 970]
gt_labels shape: torch.Size([20, 3031, 1])
gt_bboxes shape: torch.Size([20, 3031, 4])





python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train230_yolo_dydcn_notall_newdata/weights/best.onnx \
                              --model_type yolo \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task1_2videos.json  \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train230_yolo_dydcn_notall_newdata/task1_2videos_pred.json 


python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train230_yoloft_dydcn_notall_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task1_2videos.json  \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task1_2videos_pred.json 
