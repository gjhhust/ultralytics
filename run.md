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
video_names: ['63', '205', '9', '113', '198', '20', '65', '256', '93', '262', '61', '237', '232', '23', '57', '263', '74', '235', '31', '13']
frame_numbers: [1220, 992, 686, 344, 354, 164, 374, 170, 318, 1122, 1210, 408, 198, 284, 496, 64, 1276, 656, 114, 134]
gt_labels shape: torch.Size([20, 3031, 1])
gt_bboxes shape: torch.Size([20, 3031, 4])

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