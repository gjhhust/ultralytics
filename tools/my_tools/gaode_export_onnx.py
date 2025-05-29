import torch
from ultralytics.nn.autobackend import AutoBackend

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoBackend('tools/my_tools/yolov8s_2.pt', device=device, dnn=False, fp16=False)
model.eval()

input_tensor = torch.randn(1, 3, 896, 896).to(device)

onnx_path = 'best.onnx'
torch.onnx.export(
    model,
    input_tensor,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"导出完成 {onnx_path}")