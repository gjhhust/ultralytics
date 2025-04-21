# python tools/train_epoch2_386+256/train_baseline.py
python tools/eval_baseline.py --dataset  config/dataset/Train_6_Test_task1.yaml --model_type YOLO
python tools/eval_baseline.py --dataset  config/dataset/Train_6_Test_task4.yaml --model_type YOLO
python tools/eval_baseline.py --dataset  config/dataset/Train_6_Test_task5.yaml --model_type YOLO
python tools/eval_baseline.py --dataset  config/dataset/Train_6_Test_task6.yaml --model_type YOLO
python tools/eval_baseline.py --dataset  config/dataset/Train_6_Test_task9.yaml --model_type YOLO
#task1: 56.1 82.6 60.5 18.1 85.4 98.5 46.6 99.0 96.8
#task4: 34.2 58.9 33.1 4.3 53.2 96.3 22.9 98.7 93.8
#task5: 41.1 67.3 40.5 4.7 52.1 91.3 25.4 93.7 91.8  
#task6: 46.2 76.9 46.8 16.0 85.3 97.7 39.1 99.0 92.9
#task9: 46.6 73.0 48.2 13.0 75.0 85.1 37.4 84.6 90.3
# python tools/eval_baseline.py --dataset config/dataset/Train_6_Test_task1_2videos.yaml --model_type YOLO 