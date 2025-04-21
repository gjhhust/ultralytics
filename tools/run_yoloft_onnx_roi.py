
from ultralytics.utils import LOGGER, ops, TQDM
import os, json
from pathlib import Path
import torch
from openvino.runtime import Core  # 导入 OpenVINO 核心库
import cv2
import onnxruntime as ort
import numpy as np
import os
import json
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional

class StreamBuffer_onnx(object):
    def __init__(self):
        super().__init__()
        self.bs = 0
        self.spatial_shape = None

    def update_memory(self, memory_last, frame_ids, spatial_shape):
        spatial_shape = torch.as_tensor(spatial_shape)
        
        memory_fmaps = [f.copy() for f in memory_last]

        b, dim, h, w = memory_last[0].shape
        assert len(frame_ids) == b, "frame_ids and memory_last should have the same batch size"
        self.bs = b
        
        
        if self.spatial_shape is None:
            self.spatial_shape = spatial_shape
            
        if not torch.equal(self.spatial_shape, spatial_shape):
            self.spatial_shape = spatial_shape 
            assert spatial_shape[-1] == spatial_shape[-2] # [1, C, H, W] H==W
            imagz = spatial_shape[-1]
            return (np.zeros([1, 64, imagz//4, imagz//4], dtype=np.float32), 
                                    np.zeros([1, 104, imagz//8, imagz//8], dtype=np.float32), 
                                    np.zeros([1, 192, imagz//16, imagz//16], dtype=np.float32))
            
        for i in range(self.bs):
            if frame_ids[i] == 0:
                for f in range(len(memory_last)):
                    memory_fmaps[f][i] = np.zeros_like(memory_last[f][i], device=memory_last[f][i].device)

        return memory_fmaps
    
class CocoVIDDataset(Dataset):
    def __init__(self, json_path: str, images_dir: str, imagz: int = 896):
        super(CocoVIDDataset, self).__init__()
        self.imagz = imagz
        self.images_dir = images_dir
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        # 提取图片信息
        self.image_info = coco_data['images']
        self.image_info.sort(key=lambda x: (x['video_id'], x['frame_id']))  # 按video_id和frame_id排序

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_info = self.image_info[idx]
        video_id = img_info['video_id']
        frame_id = img_info['frame_id']
        file_name = img_info['file_name']
        img_path = os.path.join(self.images_dir, file_name)
        
        # 加载图片
        img = Image.open(img_path).convert('RGB')
        
        # 获取图片尺寸
        orig_w, orig_h = img.size
        
        # 缩放到最长边为imagz
        if orig_w > orig_h:
            new_w = self.imagz
            new_h = int(orig_h * (self.imagz / orig_w))
        else:
            new_h = self.imagz
            new_w = int(orig_w * (self.imagz / orig_h))
            
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # 填充
        new_img = Image.new('RGB', (self.imagz, self.imagz), (114, 114, 114))
        paste_x = (self.imagz - new_w) // 2
        paste_y = (self.imagz - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # 转换为numpy数组并归一化
        img_np = np.array(new_img).astype(np.float32) / 255.0
        
        # 调整维度为 [C, H, W] 然后转换为 [1, C, H, W]
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        # img_np = np.expand_dims(img_np, axis=0)  # [C, H, W] -> [1, C, H, W]
        
        return img_np, file_name, frame_id, {
            'file_name': file_name,
            'orig_w': orig_w,
            'orig_h': orig_h,
            'new_w': new_w,
            'new_h': new_h,
            'paste_x': paste_x,
            'paste_y': paste_y
        }
        
    def get_original_image_and_bbox(self, img_np: np.ndarray, transform_info: dict, bbox: Optional[List[float]] = None) -> Tuple[Image.Image, Tuple[int, int], Optional[List[float]]]:
        # # 找到对应的变换信息
        # transform_info = next(info for info in self.transform_info if info['file_name'] == file_name)
        
        # 获取变换信息
        orig_w, orig_h = transform_info['orig_w'], transform_info['orig_h']
        new_w, new_h = transform_info['new_w'], transform_info['new_h']
        paste_x, paste_y = transform_info['paste_x'], transform_info['paste_y']
        
        # 复原图片
        img_np = img_np.squeeze(0)  # [1, C, H, W] -> [C, H, W]
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = img_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w, :]
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        # 复原bbox
        # 复原bbox
        if bbox is not None:
            # bbox: [n, 4] in normalized [0, 1] range
            n, _ = bbox.shape
            orig_bbox = np.zeros((n, 4))
            for i in range(n):
                x1, y1, x2, y2 = bbox[i]
                # x1 *= self.imagz
                # y1 *= self.imagz
                # x2 *= self.imagz
                # y2 *= self.imagz
                
                x1 -= paste_x
                y1 -= paste_y
                x2 -= paste_x
                y2 -= paste_y
                
                x1 = x1 / new_w * orig_w
                y1 = y1 / new_h * orig_h
                x2 = x2 / new_w * orig_w
                y2 = y2 / new_h * orig_h
                
                # 将xyxy转换为xywh
                orig_bbox[i] = [x1, y1, x2-x1, y2-y1]
            return img, (orig_w, orig_h), orig_bbox
        else:
            return img, (orig_w, orig_h), None

def custom_collate(batch):
    img_nps = []
    file_names = []
    frame_ids = []
    transform_infos = []
    
    for img_np, file_name, frame_id, transform_info in batch:
        img_nps.append(img_np)
        file_names.append(file_name)
        frame_ids.append(frame_id)
        transform_infos.append(transform_info)
    
    return np.stack(img_nps), file_names, frame_ids, transform_infos

def preprocess(buffer, img_np, frame_id, save_fmaps, imagz = 896, roi=None):
    """Preprocesses batch of images for YOLO training."""
    if roi is not None:
        # 裁剪图片
        x1, y1, x2, y2 = roi
        img_np = img_np[:, :, y1:y2, x1:x2]
        # 调整大小
        h, w = img_np.shape[2:]
        img_np = cv2.resize(img_np.transpose(0, 2, 3, 1).squeeze(0), (imagz, imagz)).transpose(2, 0, 1)[np.newaxis, :]

    if save_fmaps is not None:
        fmaps_old = buffer.update_memory(save_fmaps, [frame_id], img_np.shape)
        batch = (img_np, fmaps_old)
    else:
        batch = (img_np, (np.zeros([1, imagz//4, imagz//4, 64], dtype=np.float32), 
                          np.zeros([1, imagz//8, imagz//8, 104], dtype=np.float32), 
                          np.zeros([1, imagz//16, imagz//16, 192], dtype=np.float32)))

    return batch
    
def postprocess(preds, conf=0.001, iou=0.7, single_cls=False, agnostic_nms=False, max_det=300):
    """Apply Non-maximum suppression to prediction outputs.
    conf: default val 0.001, predic 0.25
    """
    return ops.non_max_suppression(
        preds,
        conf,
        iou,
        labels=[],
        multi_label=True,
        agnostic=single_cls or agnostic_nms,
        max_det=max_det,
    )

class CocoEvaluators:
    def __init__(self, eval_ann_json="/data/jiahaoguo/dataset/gaode_6/annotations/mini_val/gaode_6_mini_val.json", 
                 class_map = [0,1],#将模型预测的分类映射为 index: value，这里相当于没有映射
                 nc=2,
                 image_dir = None,
                 save_dir=None,
                 show = False):
        """Initialize evaluation metrics for YOLO."""
        self.eval_ann_json = eval_ann_json
        with open(eval_ann_json, 'r', encoding='utf-8') as f:
            self.gt_cocodata = json.load(f)
        
        assert image_dir is not None and show, "show mode need image_dir"
        assert save_dir is not None and show, "show mode need save_dir"
        
        self.image_dir = image_dir
        # 图片名称找到对应的image_id
        self.image_name_map_id = {}
        for image in self.gt_cocodata["images"]:
            self.image_name_map_id[image["file_name"]] = image["id"]

        self.class_map = {int(index):value  for index, value in enumerate(class_map)}
        self.nc = nc
        self.jdict = []
        
        self.show = show
        
        self.save_dir = save_dir
        self.show = show
        self.video_name = None
        self.video_writer = None

    def update_metrics(self, preds, file_names):
        """Metrics."""
        for si, pred in enumerate(preds):
            # Predictions
            if self.nc == 1:
                pred[:, 5] = 0
            # Save
            self.pred_to_json(pred, file_names[si])
    
    def from_coco_get_image_id(self,file_name_mapping_id,im_file):
        if file_name_mapping_id:
            return file_name_mapping_id.get(im_file, 0)
        assert False, f"error in function from_coco_get_image_id, {im_file} is not in {self.data['eval_ann_json']}"
        
    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        # stem = Path(filename).stem
        # image_id = int(stem) if stem.isnumeric() else 
        if os.sep in self.gt_cocodata["images"][0]["file_name"]:
            path, file = os.path.split(filename)
            file_name = os.path.join(os.path.basename(path), file)
        else:
            file_name = os.path.basename(filename)
        image_id = self.from_coco_get_image_id(self.image_name_map_id,file_name)
        
        # box = ops.xyxy2xywh(predn[:, :4])  # xywh
        # box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        if self.show:
            cur_video_name = os.path.dirname(file_name)
            if self.video_name is None or self.video_name != cur_video_name:
                # 如果视频名称更换或首次处理，关闭之前的视频写入器并创建新的
                if self.video_writer is not None:
                    self.video_writer.release()

                self.video_name = cur_video_name
                output_video_path = os.path.join(self.save_dir, f"{cur_video_name}.mp4")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

                # 读取第一张图片以获取图像尺寸
                image_path = os.path.join(self.image_dir, file_name)
                img = cv2.imread(image_path)
                height, width, _ = img.shape

                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

            image_path = os.path.join(self.image_dir, file_name)
            img = cv2.imread(image_path)
            box = predn[:, :4]  # xywh
            for p, b in zip(predn.tolist(), box.tolist()):
                x, y, w, h = b
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                # 获取类别名称，假设 class_map 是一个将类别 ID 映射到类别名称的字典
                class_name = self.class_map[int(p[5])]
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 绘制类别名称和置信度
                text = f"{class_name}: {round(p[4], 5)}"
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 将绘制后的图像写入视频
            self.video_writer.write(img)
            
        box = predn[:, :4]# xywh
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
    
    def eval_json(self, pred_json):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        # anno_json = (
        #     self.data["path"]
        #     / "annotations"
        #     / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        # )  # annotations
        anno_json = Path(self.eval_ann_json)  # annotations
        pkg = "pycocotools"
        LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

            from pycocotools.coco import COCO  # noqa
            # from pycocotools.cocoeval import COCOeval  # noqa
            from ultralytics.data.cocoeval import COCOeval  # noqa

            anno = COCO(str(anno_json))  # init annotations api
            pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
            val = COCOeval(anno, pred, "bbox")
            # val.params.imgIds = self.imgIDS
            val.evaluate()
            val.accumulate()
            val.summarize()
            # update mAP50-95 and mAP50
        except Exception as e:
            LOGGER.warning(f"{pkg} unable to run: {e}")

    def save_json(self, save_dir):
        save_dir = Path(save_dir)
        preds_json = save_dir / "predictions.json"
        with open(preds_json, "w") as f:
            LOGGER.info(f"Saving {f.name}...")
            json.dump(self.jdict, f)  # flatten and save
        self.eval_json(preds_json)  # update stats
        
def calculate_roi(boxes, img_shape, expand_ratio=0.5, min_size=512):
    """计算包含所有边界框的最大 ROI 区域并进行适当扩大，确保 ROI 至少为 min_size x min_size"""
    if len(boxes) == 0:
        return None

    # 提取所有边界框的坐标信息
    x1_values = boxes[:, 0]
    y1_values = boxes[:, 1]
    x2_values = boxes[:, 0] + boxes[:, 2]
    y2_values = boxes[:, 1] + boxes[:, 3]

    # 找出最小的 x1 和 y1，以及最大的 x2 和 y2
    x1 = np.min(x1_values)
    y1 = np.min(y1_values)
    x2 = np.max(x2_values)
    y2 = np.max(y2_values)

    # 计算扩展量
    width_expand = expand_ratio * (x2 - x1)
    height_expand = expand_ratio * (y2 - y1)

    # 扩展 ROI
    x1 = int(max(0, x1 - width_expand))
    y1 = int(max(0, y1 - height_expand))
    x2 = int(min(img_shape[1], x2 + width_expand))
    y2 = int(min(img_shape[0], y2 + height_expand))

    # 计算当前 ROI 的宽度和高度
    roi_width = x2 - x1
    roi_height = y2 - y1

    # 如果 ROI 小于 min_size，沿中心进行扩展
    if roi_width < min_size or roi_height < min_size:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_width = max(roi_width, min_size)
        new_height = max(roi_height, min_size)

        x1 = max(0, center_x - new_width // 2)
        y1 = max(0, center_y - new_height // 2)
        x2 = min(img_shape[1], center_x + new_width // 2)
        y2 = min(img_shape[0], center_y + new_height // 2)

    return (x1, y1, x2, y2)

def sliding_average_boxes(boxes, history_boxes, window_size):
    """对检测框进行滑动平均计算"""
    history_boxes.append(boxes)
    if len(history_boxes) > window_size:
        history_boxes.pop(0)
    all_bboxes = []
    for bboxes in history_boxes:
        all_bboxes.extend(bboxes)
    all_boxes = np.concatenate(all_bboxes, axis=0)
    avg_boxes = np.mean(all_boxes, axis=0, keepdims=True)
    return avg_boxes


def load_onnx_model(onnx_model_path, device='cpu', gpu_id=0, cpu_threads=4):
    """
    加载ONNX模型并返回推理会话。

    :param onnx_model_path: ONNX模型的路径
    :param device: 设备类型，'cpu' 或 'gpu'
    :param gpu_id: GPU设备编号
    :param cpu_threads: CPU线程数
    :return: ONNX推理会话
    """
    session_options = ort.SessionOptions()
    if device == 'gpu':
        # providers = [('CUDAExecutionProvider', {'device_id': gpu_id})]
        providers = ['CUDAExecutionProvider']
    elif device == 'cpu':
        providers = ['CPUExecutionProvider']
        session_options.intra_op_num_threads = cpu_threads
        session_options.inter_op_num_threads = cpu_threads
    else:
        raise ValueError("Invalid device type. Choose either 'cpu' or 'gpu'.")

    session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    
    # import onnxruntime as ort
 
    # import onnxruntime as ort
 
    # 创建一个推理会话
    # session = ort.InferenceSession(r"YOLO_t22_best.onnx", providers=['CUDAExecutionProvider'])
    
    # 检查是否使用了CUDA
    providers = session.get_providers()
    print(f"Available providers: {providers}")
    
    # 获取当前执行程序的是否使用GPU设备
    device = ort.get_device()
    print(f"Current device: {device}")

    return session

def run_model(session, input, augment=False):
    # 获取输入输出的名称
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    # 构造输入字典
    inputs = {
        input_names[0]: input[0],  # 对应图像输入
        input_names[1]: input[1][0],  # 对应fmap2
        input_names[2]: input[1][1],  # 对应fmap1
        input_names[3]: input[1][2]   # 对应fmap0
    }
    # # for i in outputs:
    # #     print(i.shape)
    # # (1, 6, 66640)
    # # (1, 64, 224, 224)
    # # (1, 104, 112, 112)
    # # (1, 192, 56, 56)
    
    # 进行推理
    pred, fmap2, fmap1, fmap0 = session.run(output_names, inputs)
    
    return (pred, (fmap2, fmap1, fmap0))


json_path = "/data/jiahaoguo/dataset/gaode_6/annotations/mini_val/gaode_6_mini_val.json"
images_dir = "/data/jiahaoguo/dataset/gaode_6/images"
onnx_model_path = "/data/shuzhengwang/project/ultralytics/runs/save/train201_DCN_32.9/weights/last.onnx"
imagz = 896



dataset = CocoVIDDataset(json_path, images_dir, imagz=imagz) # 加载数据集
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

# 加载ONNX模型
# session = load_onnx_model(onnx_model_path, device='gpu', gpu_id=0, cpu_threads=8)
# 读取 ONNX 模型并编译为 OpenVINO 模型
core = Core()
model = core.read_model(model=onnx_model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")  # 可以将 "CPU" 替换为 "GPU" 以使用 GPU 加速

buffer = StreamBuffer_onnx() # 保存特征图缓冲区
coco_evaluator = CocoEvaluators(eval_ann_json="/data/jiahaoguo/dataset/gaode_6/annotations/mini_val/gaode_6_mini_val.json", 
                         class_map = [0,1],#将模型预测的分类映射为 index: value，这里相当于没有映射
                         nc=2,
                         show=True,
                         image_dir="/data/jiahaoguo/dataset/gaode_6/images/",
                         save_dir="/data/shuzhengwang/project/ultralytics/results/shows")

# 新增变量用于保存历史检测框
history_boxes = []
# 滑动平均的窗口大小
history_window_size = 10
save_fmaps = None
for i, batch in enumerate(TQDM(dataloader, total=len(dataloader))):
    img_np, file_names, frame_ids, transform_infos = batch
    
    # 计算 ROI 区域
    if len(history_boxes) > 0:
        avg_boxes = sliding_average_boxes(history_boxes[-1], history_boxes, history_window_size)
        roi = calculate_roi(avg_boxes, img_np.shape[2:])
    else:
        roi = None
        
    batch = preprocess(buffer, img_np, frame_ids[0], save_fmaps, imagz, roi)

    # 获取模型的输入层
    input_layer = next(iter(compiled_model.inputs))

    # 构造输入字典
    inputs = {
        input_layer.any_name: batch[0],  # 对应图像输入
        "fmap2": batch[1][0],  
        "fmap1": batch[1][1],  
        "fmap0": batch[1][2]   
    }

    # 进行推理
    outputs = compiled_model(inputs)
    pred = outputs[compiled_model.output(0)]
    fmap2 = outputs[compiled_model.output(1)]
    fmap1 = outputs[compiled_model.output(2)]
    fmap0 = outputs[compiled_model.output(3)]

    preds = (pred, (fmap2, fmap1, fmap0))

    # save_fmaps = [np.transpose(f, (0, 3, 1, 2)) for f in save_fmaps]
    preds = [pred.numpy() for pred in postprocess(torch.from_numpy(preds[0]), conf=0.001)]  # val 0.001, pred 0.25

    # 保存当前检测框
    if len(preds) > 0:
        current_boxes = preds
        if roi is not None:
            # 正确获取 roi 信息
            x1, y1, x2, y2 = roi
            # 计算 roi 区域的宽度和高度
            roi_w = x2 - x1
            roi_h = y2 - y1
            # 计算缩放因子，这里假设 pred 是 roi 区域 resize 到 img_np 尺寸后的检测结果
            scale_x = img_np.shape[3] / roi_w
            scale_y = img_np.shape[2] / roi_h
            for pred in current_boxes:
                # 先将检测框从 img_np 尺寸还原到 roi 尺寸,再偏移
                pred[:, 0] = pred[:, 0] / scale_x + x1
                pred[:, 1] = pred[:, 1] / scale_y + y1
                pred[:, 2] = pred[:, 2] / scale_x + x1
                pred[:, 3] = pred[:, 3] / scale_y + y1
        history_boxes.append(current_boxes)
        preds = current_boxes
        
    for pred in preds:
        img_orige, (orig_w, orig_h), pred[:, :4] = dataset.get_original_image_and_bbox(img_np, transform_infos[0], pred[:, :4])
    coco_evaluator.update_metrics(preds, file_names)

coco_evaluator.save_json(os.path.dirname(onnx_model_path))


#####################session 推理###############################
# save_fmaps = None

# for i, batch in enumerate(TQDM(dataloader, total=len(dataloader))):
#     img_np, file_names, frame_ids, transform_infos = batch
#     batch = preprocess(buffer, img_np, frame_ids[0], save_fmaps)
#     preds, save_fmaps = run_model(session, batch)
#     # save_fmaps = [np.transpose(f, (0, 3, 1, 2)) for f in save_fmaps]
#     preds = [pred.numpy() for pred in postprocess(torch.from_numpy(preds), conf=0.001)]  #val 0.001, pred 0.25
    
#     for pred in preds:
#         img_orige, (orig_w, orig_h), pred[:, :4] = dataset.get_original_image_and_bbox(img_np, transform_infos[0], pred[:, :4])
#     coco_evaluator.update_metrics(preds, file_names)

# coco_evaluator.save_json(os.path.dirname(onnx_model_path))


