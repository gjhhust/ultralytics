# python tools/train_epoch2_386+256/train_baseline.py
python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task1/task1_test.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task1_pred.json 

python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task4/task4_test.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task4_pred.json 

python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task5/task5_test.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task5_pred.json 

python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task6/task6_test.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task6_pred.json 

python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task9/task9_test.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task9_pred.json 

python tools/run_yoloft_openVINO.py --model_path runs/save/train330_YOLOftS_all2_dy_s1_t_60/weights/best_int8_openvino_model \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/dataset/gaode_6/annotations/task1_2videos.json \
                              --pred_json runs/save/train330_YOLOftS_all2_dy_s1_t_60/int8_openvino/task1_2videos_pred.json 