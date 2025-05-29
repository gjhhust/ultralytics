python tools/my_tools/yoloft_conpresion.py | tee >(ts '[%Y-%m-%d %H:%M:%S]' >> logs/yoloft__time_$(date +%Y%m%d_%H%M).log)

python tools/my_tools/yoloft_baseline.py | tee >(ts '[%Y-%m-%d %H:%M:%S]' >> logs/yolov8s_ftv1_dcn_dy_3d_nwdloss$(date +%Y%m%d_%H%M).log)


