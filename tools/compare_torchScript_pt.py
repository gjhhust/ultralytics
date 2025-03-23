import torch
import numpy as np
from ultralytics import YOLO, YOLOFT
from torch.testing import assert_close
from PIL import Image
import os
def load_and_preprocess_image(image_path, img_size=896):
    """
    加载图片并进行预处理，包括调整大小和归一化
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # 转换为 (C, H, W)
    image_np = np.expand_dims(image_np, axis=0)  # 添加批次维度
    return image_np

def compare_pytorch_torchscript_outputs(torchscript_model_path, pytorch_model_path, image_path, model_type="yolo", compare_dir=None):
    """
    比较 PyTorch 模型和 TorchScript 模型的输出误差，使用 torch.testing.assert_close。

    Args:
        torchscript_model_path (str): TorchScript 模型文件的路径。
        pytorch_model_path (str): PyTorch 模型文件的路径。
        image_path (str): 输入图片的路径。
        model_type (str): 模型类型，'yolo' 或 'yoloft'。
    """
    os.makedirs(compare_dir, exist_ok=True)
    # 1. 加载 PyTorch 模型
    if model_type == "yoloft":
        pytorch_model = YOLOFT(pytorch_model_path)
    else:
        pytorch_model = YOLO(pytorch_model_path)
    pytorch_model.eval()

    # 2. 加载 TorchScript 模型
    torchscript_model = torch.jit.load(torchscript_model_path)
    torchscript_model.eval()

    # 3. 准备输入数据
    img_size = 896
    input_image_np = load_and_preprocess_image(image_path, img_size)
    input_image_torch = torch.tensor(input_image_np)

    if model_type == "yoloft":
        fmap2_np = np.random.rand(1, 224, 224, 64).astype(np.float32)
        fmap1_np = np.random.rand(1, 112, 112, 104).astype(np.float32)
        fmap0_np = np.random.rand(1, 56, 56, 192).astype(np.float32)
        fmaps_np = [fmap2_np, fmap1_np, fmap0_np]

        fmap2_torch = torch.tensor(fmap2_np)
        fmap1_torch = torch.tensor(fmap1_np)
        fmap0_torch = torch.tensor(fmap0_np)
        fmaps_torch = [fmap2_torch, fmap1_torch, fmap0_torch]

    # 保存输入数据到 npz 文件
    
    np.save(os.path.join(compare_dir, 'input_img.npy'), input_image_np)
    if model_type == "yoloft":
        np.save(os.path.join(compare_dir, 'fmap2_np.npy'), fmap2_np)
        np.save(os.path.join(compare_dir, 'fmap1_np.npy'), fmap1_np)
        np.save(os.path.join(compare_dir, 'fmap0_np.npy'), fmap0_np)
        

    # 4. 进行 PyTorch 推理
    with torch.no_grad():
        if model_type == "yoloft":
            pytorch_input = (input_image_torch, fmaps_torch)
            _ = pytorch_model(pytorch_input, device="cpu")

        elif model_type == "yolo":
            pytorch_input = input_image_torch
            _ = pytorch_model(pytorch_input, device="cpu")

        else:
            raise ValueError("model_type must be 'yolo' or 'yoloft'")

        model_sequrntial = pytorch_model.predictor.model.model.model
        y_pytorch, dt_pytorch, embeddings_pytorch = [], [], []
        x_pytorch = pytorch_input

        model_sequrntial.save = [i for i in range(len(model_sequrntial))]

        for m in model_sequrntial:
            if m.f != -1:
                x_pytorch_input = y_pytorch[m.f] if isinstance(m.f, int) else [x_pytorch if j == -1 else y_pytorch[j] for j in m.f]
            else:
                x_pytorch_input = x_pytorch

            x_pytorch = m(x_pytorch_input)
            y_pytorch.append(x_pytorch if m.i in model_sequrntial.save else None)

        if model_type == "yolo":
            pytorch_pred_output = x_pytorch

        elif model_type == "yoloft":
            pytorch_pred_output, pytorch_fmaps_outputs = x_pytorch, [f.permute(0, 2, 3, 1) for f in y_pytorch[13]]
            pytorch_pred_output = pytorch_pred_output[0]

    # 5. 进行 TorchScript 推理
    with torch.no_grad():
        if model_type == "yoloft":
            torchscript_output = torchscript_model(input_image_torch, *fmaps_torch)
            torchscript_pred_output = torchscript_output[0]
            torchscript_fmaps_outputs = torchscript_output[1]

        elif model_type == "yolo":
            torchscript_input = input_image_torch
            torchscript_output = torchscript_model(torchscript_input)
            torchscript_pred_output = torchscript_output

    # 保存模型输出到 npz 文件
    if model_type == "yoloft":
        np.save(os.path.join(compare_dir, 'pytorch_output.npy'), pytorch_pred_output.cpu().numpy())
        np.save(os.path.join(compare_dir, 'torchscript_output.npy'), torchscript_pred_output.cpu().numpy())
    else:
        np.save(os.path.join(compare_dir, 'pytorch_output.npy'), pytorch_pred_output.cpu().numpy())
        np.save(os.path.join(compare_dir, 'torchscript_output.npy'), torchscript_pred_output.cpu().numpy())

    # 6. 误差比较
    rtol_threshold = 1e-4
    atol_threshold = 1e-4

    try:
        assert_close(torchscript_pred_output, pytorch_pred_output, rtol=rtol_threshold, atol=atol_threshold)
        if model_type == "yoloft":
            for i in range(len(torchscript_fmaps_outputs)):
                assert_close(torchscript_fmaps_outputs[i], pytorch_fmaps_outputs[i], rtol=rtol_threshold, atol=atol_threshold)
        print(f"✅ 整体输出 `assert_close` 验证通过，相对误差阈值 (rtol)={rtol_threshold}, 绝对误差阈值 (atol)={atol_threshold}")

    except AssertionError as e:
        print(f"❌ 整体输出 `assert_close` 验证失败，误差超过阈值。开始逐层比较。")
        print(f"AssertionError 详情: {e}")


if __name__ == '__main__':
    torchscript_model_path = "runs/save/train227_yoloft_dydcn_newdata/weights/best.torchscript"
    pytorch_model_path = "runs/save/train227_yoloft_dydcn_newdata/weights/best.pt"
    compare_dir = "runs/save/train227_yoloft_dydcn_newdata/weights/compare"
    image_path = "test.jpg"  # 替换为实际的图片路径
    model_type = "yoloft"

    compare_pytorch_torchscript_outputs(torchscript_model_path, pytorch_model_path, image_path, model_type, compare_dir)