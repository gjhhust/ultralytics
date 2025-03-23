# python tools/train_epoch2_386+256/train_baseline.py
python tools/eval_baseline.py --dataset /data/shuzhengwang/project/ultralytics/config/dataset/Train_6_Test_task1.yaml \
                            --model_wight runs/save/train317_YOLOftS_dcn_dy_s1/weights/best.pt --model_type YOLOFT

python tools/eval_baseline.py --dataset /data/shuzhengwang/project/ultralytics/config/dataset/Train_6_Test_task4.yaml \
                            --model_wight runs/save/train317_YOLOftS_dcn_dy_s1/weights/best.pt --model_type YOLOFT

python tools/eval_baseline.py --dataset /data/shuzhengwang/project/ultralytics/config/dataset/Train_6_Test_task5.yaml \
                            --model_wight runs/save/train317_YOLOftS_dcn_dy_s1/weights/best.pt --model_type YOLOFT

python tools/eval_baseline.py --dataset /data/shuzhengwang/project/ultralytics/config/dataset/Train_6_Test_task6.yaml \
                            --model_wight runs/save/train317_YOLOftS_dcn_dy_s1/weights/best.pt --model_type YOLOFT

                            
python tools/eval_baseline.py --dataset /data/shuzhengwang/project/ultralytics/config/dataset/Train_6_Test_task9.yaml \
                            --model_wight runs/save/train317_YOLOftS_dcn_dy_s1/weights/best.pt --model_type YOLOFT

#task1: 68.0 90.6 77.7 54.5 95.3 99.0 70.4 99.0 98.0
#task4: 54.1 85.4 56.1 31.7 88.2 99.0 49.7 98.9 94.7 
#task5: 60.2 91.3 63.3 34.3 83.2 97.1 52.9 98.5 96.8
#task6: 64.3 93.5 70.8 52.4 97.9 99.0 66.4 99.0 94.7
#task9: 64.1 89.3 72.5 50.5 94.1 97.9 67.4 93.9 95.2
# python tools/eval_baseline.py --dataset config/dataset/Train_6_Test_task1_2videos.yaml --model_type YOLOFT 