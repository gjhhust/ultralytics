# python tools/train_epoch2_386+256/train_baseline.py
python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task1/task1_test.json \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task1_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task4/task4_test.json \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task4_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task5/task5_test.json \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task5_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task6/task6_test.json \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task6_pred.json 

python tools/run_yoloft_onnx.py --onnx_model_path /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/weights/best.onnx \
                              --model_type yoloft \
                              --json_path /data/jiahaoguo/datasets/gaode_6/annotations/task9/task9_test.json \
                              --pred_json /data/shuzhengwang/project/ultralytics/runs/save/train227_yoloft_dydcn_newdata/onnx_test/task9_pred.json 