# python tools/train_epoch2_386+256/train_baseline.py
python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task1/task1_test.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task1_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task4/task4_test.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task4_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task5/task5_test.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task5_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task6/task6_test.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task6_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task9/task9_test.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task9_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path runs/save/train317_YOLOftS_dcn_dy_s1_t/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task1_2videos.json \
                              --pred_json runs/save/train317_YOLOftS_dcn_dy_s1_t/onnx_test_onnx/task1_2videos_pred.json 