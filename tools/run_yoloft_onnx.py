
from ultralytics.utils import LOGGER, ops, TQDM
import os, json
from pathlib import Path
import torch
import argparse
import time
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

def preprocess(buffer, img_np, frame_id, save_fmaps, imagz = 896):
        """Preprocesses batch of images for YOLO training."""
        if save_fmaps is not None:
            fmaps_old = buffer.update_memory(save_fmaps, [frame_id], img_np.shape)
            batch = (img_np, fmaps_old)
        else:
            batch = (img_np, (np.zeros([1, 64, imagz//4, imagz//4], dtype=np.float32), 
                                    np.zeros([1, 104, imagz//8, imagz//8], dtype=np.float32), 
                                    np.zeros([1, 192, imagz//16, imagz//16], dtype=np.float32)))

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
    def __init__(self, eval_ann_json="/data/jiahaoguo/datasets/gaode_6/annotations/mini_val/gaode_6_mini_val.json", 
                 class_map = [0,1],#将模型预测的分类映射为 index: value，这里相当于没有映射
                 nc=2):
        """Initialize evaluation metrics for YOLO."""
        self.eval_ann_json = eval_ann_json
        with open(eval_ann_json, 'r', encoding='utf-8') as f:
            self.gt_cocodata = json.load(f)
            
        # 图片名称找到对应的image_id
        self.image_name_map_id = {}
        for image in self.gt_cocodata["images"]:
            self.image_name_map_id[image["file_name"]] = image["id"]

        self.class_map = {int(index):value  for index, value in enumerate(class_map)}
        self.nc = nc
        self.jdict = []

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


json_path = "/data/jiahaoguo/datasets/gaode_6/annotations/mini_val/gaode_6_mini_val.json"
images_dir = "/data/jiahaoguo/datasets/gaode_6/images"
onnx_model_path = "/data/shuzhengwang/project/ultralytics/runs/detect/train68/weights/last.onnx"
imagz = 896



dataset = CocoVIDDataset(json_path, images_dir, imagz=imagz) # 加载数据集
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

session = ort.InferenceSession(onnx_model_path)
buffer = StreamBuffer_onnx() # 保存特征图缓冲区
coco_evaluator = CocoEvaluators(eval_ann_json="/data/jiahaoguo/datasets/gaode_6/annotations/mini_val/gaode_6_mini_val.json", 
                         class_map = [0,1],#将模型预测的分类映射为 index: value，这里相当于没有映射
                         nc=2)


save_fmaps = None

for i, batch in enumerate(TQDM(dataloader, total=len(dataloader))):
    img_np, file_names, frame_ids, transform_infos = batch
    batch = preprocess(buffer, img_np, frame_ids[0], save_fmaps)
    preds, save_fmaps = run_model(session, batch)
    preds = [pred.numpy() for pred in postprocess(torch.from_numpy(preds), conf=0.001)]  #val 0.001, pred 0.25
    
    for pred in preds:
        img_orige, (orig_w, orig_h), pred[:, :4] = dataset.get_original_image_and_bbox(img_np, transform_infos[0], pred[:, :4])
    coco_evaluator.update_metrics(preds, file_names)

coco_evaluator.save_json(os.path.dirname(onnx_model_path))


# 随机生成示例输入数据，替换为实际数据
# img = np.random.randn(1, 3, 896, 896).astype(np.float32)  # 输入图像
# fmap2 = np.random.randn(1, 64, 224, 224).astype(np.float32)  # 特征图2
# fmap1 = np.random.randn(1, 104, 112, 112).astype(np.float32)  # 特征图1
# fmap0 = np.random.randn(1, 192, 56, 56).astype(np.float32)  # 特征图0

# input_names = [input.name for input in session.get_inputs()]
# output_names = [output.name for output in session.get_outputs()]
# # 构造输入字典
# inputs = {
#     input_names[0]: img,  # 对应图像输入
#     input_names[1]: fmap2,  # 对应fmap2
#     input_names[2]: fmap1,  # 对应fmap1
#     input_names[3]: fmap0   # 对应fmap0
# }

# outputs = session.run(output_names, inputs)
# # for i in outputs:
# #     print(i.shape)
# # (1, 6, 66640)
# # (1, 64, 224, 224)
# # (1, 104, 112, 112)
# # (1, 192, 56, 56)

# # 输出结果
# print("Predictions:", outputs[0])  # 预测结果
# print("Feature Maps:", outputs[1])  # 特征图

