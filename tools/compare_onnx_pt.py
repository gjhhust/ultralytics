import torch,os
import numpy as np
from ultralytics import YOLO, YOLOFT
from openvino.runtime import Core
from torch.testing import assert_close

def compare_pytorch_onnx_outputs(onnx_model_path, pytorch_model_path, model_type="yolo", compare_dir=None):
    """
    比较 PyTorch 模型和 ONNX 模型的输出误差，使用 torch.testing.assert_close。

    Args:
        onnx_model_path (str): ONNX 模型文件的路径。
        model_type (str): 模型类型，'yolo' 或 'yoloft'。
    """
    # 1. 加载 PyTorch 模型 (这里为了对比，我们假设你已经导出了 ONNX 模型，
    #    并且知道原始 PyTorch 模型结构，这里简化操作，直接使用 YOLO 类，
    #    实际应用中你需要确保加载的 PyTorch 模型与导出 ONNX 的模型一致)
    # 假设你的原始 PyTorch 模型结构与 ultralytics 的 YOLO 类兼容
    os.makedirs(compare_dir, exist_ok=True)
    if model_type=="yoloft":
        pytorch_model = YOLOFT(pytorch_model_path) # 替换为你的 PyTorch 模型加载方式
    else:
        pytorch_model = YOLO(pytorch_model_path)
    
    pytorch_model.eval() # 设置为评估模式

    # 2. 加载 ONNX 模型
    core = Core()
    onnx_model = core.read_model(model=onnx_model_path)
    compiled_model = core.compile_model(model=onnx_model, device_name="CPU") # 或者 "GPU" 如果你想在 GPU 上运行

    # 3. 准备输入数据
    # 输入形状 (1, 3, 896, 896)
    img_size = 896
    input_image_np = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    input_image_torch = torch.tensor(input_image_np)

    # 特征图缓冲区初始化 (仅当 model_type == "yoloft" 时需要)
    if model_type == "yoloft":
        # , , 
        # fmap2_np = np.random.rand(1, 112, 112, 152).astype(np.float32)
        # fmap1_np = np.random.rand(1, 56, 56, 288).astype(np.float32)
        # fmap0_np = np.random.rand(1, 28, 28, 576).astype(np.float32)
        fmap2_np = np.random.rand(1, 112, 112, 104).astype(np.float32)
        fmap1_np = np.random.rand(1, 56, 56, 192).astype(np.float32)
        fmap0_np = np.random.rand(1, 56//2, 56//2, 384).astype(np.float32)
        fmaps_np = [fmap2_np, fmap1_np, fmap0_np]

        fmap2_torch = torch.tensor(fmap2_np)
        fmap1_torch = torch.tensor(fmap1_np)
        fmap0_torch = torch.tensor(fmap0_np)
        fmaps_torch = [fmap2_torch, fmap1_torch, fmap0_torch]


    np.save(os.path.join(compare_dir, 'input0.npy'), input_image_np)
    if model_type == "yoloft":
        np.save(os.path.join(compare_dir, 'input1.npy'), fmap2_np)
        np.save(os.path.join(compare_dir, 'input2.npy'), fmap1_np)
        np.save(os.path.join(compare_dir, 'input3.npy'), fmap0_np)

    # 4. 进行 PyTorch 推理
    with torch.no_grad():
        if model_type == "yoloft":
            # preds, save_fmaps = model((batch["img"], save_fmaps), augment=augment)
            # 模拟 yoloft 的输入结构，你需要根据实际情况调整
            pytorch_input = (input_image_torch, fmaps_torch) # 这里可能需要调整输入方式，取决于你的 preprocess 函数
            pytorch_output = pytorch_model(pytorch_input, device="cpu") # ultralytics 的 YOLO 模型通常只需要图像输入
            pytorch_pred = pytorch_output # 假设主要输出是这个，你需要根据实际模型调整

        elif model_type == "yolo":
            pytorch_input = input_image_torch # 假设只需要图像输入
            pytorch_output = pytorch_model(pytorch_input, device="cpu")
            pytorch_pred = pytorch_output # 假设主要输出是这个
        
        else:
            raise ValueError("model_type must be 'yolo' or 'yoloft'")

        # 4. 进行 PyTorch 推理 (模仿 ultralytics 模型 forward 过程)
    with torch.no_grad():
        model_sequrntial = pytorch_model.predictor.model.model.model
        y_pytorch, dt_pytorch, embeddings_pytorch = [], [], []  # outputs
        x_pytorch = pytorch_input # 初始输入为图像

        # if not hasattr(model_sequrntial, 'save'): # 确保 save 属性存在, 不存在则创建一个空列表
        model_sequrntial.save = [i for i in range(len(model_sequrntial))]

        for m in model_sequrntial:
            if m.f != -1:  # if not from previous layer
                x_pytorch_input = y_pytorch[m.f] if isinstance(m.f, int) else [x_pytorch if j == -1 else y_pytorch[j] for j in m.f]  # from earlier layers
            else:
                x_pytorch_input = x_pytorch # 初始层输入

            x_pytorch = m(x_pytorch_input)  # run layer

            y_pytorch.append(x_pytorch if m.i in model_sequrntial.save else None)  # save output

        if model_type == "yolo":
            pytorch_pred_output  = x_pytorch

        elif model_type == "yoloft":
            pytorch_pred_output, pytorch_fmaps_outputs = x_pytorch, [f.permute(0, 2, 3, 1) for f in y_pytorch[13][3:]] # 提取输出，假设索引 13 的输出是特征图
            pytorch_pred_output = pytorch_pred_output[0]
        
    # 5. 进行 ONNX 推理
    input_layer = next(iter(compiled_model.inputs))
    if model_type == "yoloft":
        inputs_onnx = {
            input_layer.any_name: input_image_np,
            "fmap2": fmap2_np,
            "fmap1": fmap1_np,
            "fmap0": fmap0_np
        }
        outputs_onnx = compiled_model(inputs=inputs_onnx)
        onnx_pred_output = outputs_onnx[compiled_model.output(0)] # 主要预测输出
        onnx_pred_fmaps = [outputs_onnx[compiled_model.output(1)], outputs_onnx[compiled_model.output(2)], outputs_onnx[compiled_model.output(3)]] # 主要预测输出

    elif model_type == "yolo":
        inputs_onnx = {
            input_layer.any_name: input_image_np
        }
        outputs_onnx = compiled_model(inputs=inputs_onnx)
        onnx_pred_output = outputs_onnx[compiled_model.output(0)] # 主要预测输出
    else:
        raise ValueError("model_type must be 'yolo' or 'yoloft'")


    # 6. 误差比较
    rtol_threshold = 1e-3
    atol_threshold = 1e-5

    try:
        np.save(os.path.join(compare_dir, 'onnx_pred_output0.npy'), onnx_pred_output)
        np.save(os.path.join(compare_dir, 'onnx_pred_output1.npy'), onnx_pred_fmaps[0])
        np.save(os.path.join(compare_dir, 'onnx_pred_output2.npy'), onnx_pred_fmaps[1])
        np.save(os.path.join(compare_dir, 'onnx_pred_output3.npy'), onnx_pred_fmaps[2])
        assert_close(torch.tensor(onnx_pred_output), torch.tensor(pytorch_pred_output), rtol=rtol_threshold, atol=atol_threshold)
        assert_close(torch.tensor(onnx_pred_fmaps[0]), torch.tensor(pytorch_fmaps_outputs[0]), rtol=rtol_threshold, atol=atol_threshold)
        assert_close(torch.tensor(onnx_pred_fmaps[1]), torch.tensor(pytorch_fmaps_outputs[1]), rtol=rtol_threshold, atol=atol_threshold)
        assert_close(torch.tensor(onnx_pred_fmaps[2]), torch.tensor(pytorch_fmaps_outputs[2]), rtol=rtol_threshold, atol=atol_threshold)
        print(f"✅ 整体输出 `assert_close` 验证通过，相对误差阈值 (rtol)={rtol_threshold}, 绝对误差阈值 (atol)={atol_threshold}")

    except AssertionError as e:
        print(f"❌ 整体输出 `assert_close` 验证失败，误差超过阈值。开始逐层比较。")
        print(f"AssertionError 详情: {e}")


if __name__ == '__main__':
    onnx_model_path = "/data/shuzhengwang/project/ultralytics/runs/save/train330_YOLOftS_all2_dy_s1_t/weights/best.onnx" # 替换为你的 ONNX 模型路径
    pytorch_model_path = "/data/shuzhengwang/project/ultralytics/runs/save/train330_YOLOftS_all2_dy_s1_t/weights/best.pt" 
    compare_dir = "runs/save/train330_YOLOftS_all2_dy_s1_t/weights/compare"
    model_type = "yoloft" # 或者 "yoloft" 根据你的模型类型设置
    
    # onnx_model_path = "/data/shuzhengwang/project/ultralytics/runs/save/train217_yolos_newdata/weights/best.onnx" # 替换为你的 ONNX 模型路径
    # pytorch_model_path = "/data/shuzhengwang/project/ultralytics/runs/save/train217_yolos_newdata/weights/best.pt" 
    # model_type = "yolo" # 或者 "yoloft" 根据你的模型类型设置

    compare_pytorch_onnx_outputs(onnx_model_path, pytorch_model_path, model_type, compare_dir)