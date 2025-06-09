python tools/my_tools/yoloft_conpresion.py | tee >(ts '[%Y-%m-%d %H:%M:%S]' >> logs/yoloft_conv_$(date +%Y%m%d_%H%M).log)

python tools/my_tools/yoloft_baseline.py | tee >(ts '[%Y-%m-%d %H:%M:%S]' >> logs/yolovfts_yolo_visdorne$(date +%Y%m%d_%H%M).log)


