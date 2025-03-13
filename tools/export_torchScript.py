from ultralytics import YOLO, YOLOFT, YOLOWorld, NAS, SAM, FastSAM, RTDETR

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLOFT("/data/shuzhengwang/project/ultralytics/yoloftS_all_MDC_dy_test.pt")  # load a custom trained model

# Export the model
# model.export(format="onnx", imgsz=869, simplify=True)
model.export(format="torchscript", imgsz=896) # imgsz 参数需要与你的模型输入尺寸匹配

# import onnx
# torch.jit.trace
# from onnxruntime.quantization import quantize_dynamic, QuantType
# from onnxconverter_common import float16
# # 原 ONNX 模型路径
# onnx_model_path = "/data/shuzhengwang/project/ultralytics/runs/detect/train227/weights/epoch6.pt"
# onnx_model = onnx.load(onnx_model_path)

# # 量化为 FP16
# quantized_onnx_model = float16.convert_float_to_float16(onnx_model)

# # 保存量化后的模型
# quantized_onnx_model_path = onnx_model_path.replace('.onnx', '_fp16.onnx')
# onnx.save(quantized_onnx_model, quantized_onnx_model_path)

# print(f"Quantized model saved to {quantized_onnx_model_path}")