
import time
from tqdm import tqdm
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


def coco80_to_coco91_class():
    r"""
    Converts 80-index (val2014) to 91-index (paper).
    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.

    Examples:
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    
def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (torch.Tensor): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (torch.Tensor): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (torch.Tensor): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

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
            return (np.zeros([1, imagz//8, imagz//8, 104], dtype=np.float32), 
            np.zeros([1, imagz//16, imagz//16, 192], dtype=np.float32),
            np.zeros([1, imagz//32, imagz//32, 384], dtype=np.float32))
            
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
        if json_path:
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            # 提取图片信息
            self.image_info = coco_data['images']
            if "videos" in coco_data:
                self.image_info.sort(key=lambda x: (x['video_id'], x['frame_id']))  # 按video_id和frame_id排序
        else:
            print("Warning: No json file provided, only image files will be used")
            # 遍历images_dir获取video_name/frames.jpg信息
            self.image_info = []
            for video_name in os.listdir(images_dir):
                video_dir = os.path.join(images_dir, video_name)
                # 获取video_dir下的jpg图片并排序，00000.jpg、00001.jpg、...
                frames = sorted(os.listdir(video_dir))
                for i, frame in enumerate(frames):
                    if frame.endswith('.jpg'):
                        # frame_id = int(frame.split('.')[0])
                        self.image_info.append({
                            'video_id': video_name,
                            'frame_id': i,
                            'file_name': os.path.join(video_name, frame)
                        })
            
        
    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_info = self.image_info[idx]
        video_id = img_info.get('video_id', 0)
        frame_id = img_info.get('frame_id', 0)
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
            "video_name": video_id,
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
        img_np = cv2.resize(img_np, (orig_w, orig_h))
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
            return img_np, (orig_w, orig_h), orig_bbox
        else:
            return img_np, (orig_w, orig_h), None

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
    
def postprocess(preds, conf=0.001, iou=0.7, single_cls=False, agnostic_nms=False, max_det=300):
    """Apply Non-maximum suppression to prediction outputs.
    conf: default val 0.001, predic 0.25
    """
    return non_max_suppression(
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
        print(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
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
            print(f"{pkg} unable to run: {e}")

    def save_json(self, preds_json):
        # save_dir = Path(save_dir)
        with open(preds_json, "w") as f:
            print(f"Saving {f.name}...")
            json.dump(self.jdict, f)  # flatten and save
        self.eval_json(preds_json)  # update stats
        
def calculate_roi(boxes, img_shape, expand_ratio=0.2):
    """计算 ROI 区域并进行适当扩大"""
    if len(boxes) == 0:
        return None
    x1 = int(max(0, min(boxes[:, 0]) - expand_ratio * (max(boxes[:, 2]) - min(boxes[:, 0]))))
    y1 = int(max(0, min(boxes[:, 1]) - expand_ratio * (max(boxes[:, 3]) - min(boxes[:, 1]))))
    x2 = int(min(img_shape[2], max(boxes[:, 2]) + expand_ratio * (max(boxes[:, 2]) - min(boxes[:, 0]))))
    y2 = int(min(img_shape[3], max(boxes[:, 3]) + expand_ratio * (max(boxes[:, 3]) - min(boxes[:, 1]))))
    return (x1, y1, x2, y2)

def sliding_average_boxes(boxes, history_boxes, window_size):
    """对检测框进行滑动平均计算"""
    history_boxes.append(boxes)
    if len(history_boxes) > window_size:
        history_boxes.pop(0)
    all_boxes = np.concatenate(history_boxes, axis=0)
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--onnx_model_path", type=str, default="yolov8s.onnx")
parser.add_argument("--model_type", type=str, default='yolo')
parser.add_argument("--json_path", type=str, default='/data/shuzhengwang/project/ultralytics/ultralytics/data/datasets/coco_sample/annotations/instances_val2017_sample.json')
parser.add_argument("--images_dir", type=str, default="/data/shuzhengwang/project/ultralytics/ultralytics/data/datasets/coco_sample/images")
parser.add_argument("--imagz", type=int, default=640)
parser.add_argument("--pred_json", type=str, default="./results.json")

parser.add_argument("--show", type=bool, default=False) #when show, will not eval and save pred_json
parser.add_argument("--show_dir", type=str, default="/data/jiahaoguo/datasets/gaode_6/show_dir_4/") #when show, will save show_dir
parser.add_argument("--conf", type=float, default=0.001) #when eval conf=0.001, pred 0.25
parser.add_argument("--iou", type=float, default=0.5) #iou threshold when eval 0.5, pred 0.2

parser.add_argument("--save_fmaps", type=bool, default=False) #保存推理过程中的fmaps，用于int8量化过程中校准数据
args = parser.parse_args()

json_path = args.json_path
images_dir = args.images_dir
onnx_model_path = args.onnx_model_path
imagz = args.imagz

print("model_test: ", args.onnx_model_path)
print("json_path: ", args.json_path)
if not args.show:
    print("pred_json: ", args.pred_json)
    os.makedirs(os.path.dirname(args.pred_json), exist_ok=True)
    print(f"pred json will save in {os.path.dirname(args.pred_json)}")

dataset = CocoVIDDataset(json_path, images_dir, imagz=imagz) # 加载数据集
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

# 加载ONNX模型
# session = load_onnx_model(onnx_model_path, device='gpu', gpu_id=0, cpu_threads=8)
# 读取 ONNX 模型并编译为 OpenVINO 模型
core = Core()
model = core.read_model(model=onnx_model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

if not args.show:
    coco_evaluator = CocoEvaluators(eval_ann_json=args.json_path, 
                            class_map = coco80_to_coco91_class(),#将模型预测的分类映射为 index: value，这里相当于没有映射
                            nc=80)
else:
    video_write = None
    # cls_id映射调色盘,低饱和度多种颜色
    color_maps = [
            
            (128, 128, 0),    # 橄榄色
            (128, 0, 0),      # 深红
            (0, 128, 0),      # 深绿
            (0, 0, 128),      # 深蓝
            (128, 128, 128),  # 灰色
            (128, 0, 128),    # 紫色
            (0, 128, 128),    # 青色
            (64, 64, 64),     # 深灰
            (192, 192, 192),  # 浅灰
            (64, 0, 0),       # 深棕
            (0, 64, 0),       # 深绿
            (0, 0, 64),       # 深蓝
        ]

save_fmaps_orige = [
            np.zeros([1, imagz//8, imagz//8, 104], dtype=np.float32), 
            np.zeros([1, imagz//16, imagz//16, 192], dtype=np.float32),
            np.zeros([1, imagz//32, imagz//32, 384], dtype=np.float32), ]

for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
    img_np, file_names, frame_ids, transform_infos = batch
    
    if frame_ids[0] == 0:
        save_fmaps = [b.copy() for b in save_fmaps_orige]
        
    # 获取模型的输入层
    input_layer = next(iter(compiled_model.inputs))

    if args.model_type == "yoloft":
        # 构造输入字典
        inputs = {
            input_layer.any_name: img_np,  # 对应图像输入
            "fmap2": save_fmaps[0],  
            "fmap1": save_fmaps[1],  
            "fmap0": save_fmaps[2]   
        }
        # 进行推理
        outputs = compiled_model(inputs)
        pred = outputs[compiled_model.output(0)]
        fmap2 = outputs[compiled_model.output(1)]
        fmap1 = outputs[compiled_model.output(2)]
        fmap0 = outputs[compiled_model.output(3)]

        save_fmaps = [fmap2, fmap1, fmap0]
        preds = (pred, save_fmaps)
        
    elif args.model_type == "yolo":
        # 构造输入字典
        inputs = {
            input_layer.any_name: img_np,  # 对应图像输入
        }
        # 进行推理
        outputs = compiled_model(inputs)
        pred = outputs[compiled_model.output(0)]

        preds = (pred, )
    else:
        AssertionError("model typr must in [yolo yoloft]")

    # save_fmaps = [np.transpose(f, (0, 3, 1, 2)) for f in save_fmaps]
    preds = [pred.numpy() for pred in postprocess(torch.from_numpy(preds[0]), conf=args.conf, iou=args.iou)]  # val 0.001, pred 0.25

    for pred in preds:
        img_orige, (orig_w, orig_h), pred[:, :4] = dataset.get_original_image_and_bbox(img_np, transform_infos[0], pred[:, :4])
        
    if args.show:
        if frame_ids[0] == 0:
            if video_write is not None:
                video_write.release()
                
            video_name = transform_infos[0]["video_name"]
            os.makedirs(args.show_dir, exist_ok=True)
            video_write = cv2.VideoWriter(os.path.join(args.show_dir, f"{video_name}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 5, (orig_w, orig_h))
            print(f"video_write: {os.path.join(args.show_dir, f'{video_name}.mp4')}")
            
        for predn in preds:
            box = predn[:, :4]# xywh
            for p, b in zip(predn.tolist(), box.tolist()):
                x1, y1, w, h = [int(x) for x in b]
                x2, y2 = x1+w, y1+h
                score = round(p[4], 2)
                cls_id = int(p[5])
                cv2.rectangle(img_orige, (x1, y1), (x2, y2), color_maps[cls_id], 2)
                # 调整文本位置以适应特别小的目标框
                # 类别ID
                scale = 0.35
                (text_width, text_height), _ = cv2.getTextSize(str(cls_id), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                cv2.putText(img_orige, str(cls_id), (max(x1-text_width, 5), max(y1, 5)), cv2.FONT_HERSHEY_SIMPLEX, scale, color_maps[cls_id], 1)
                # 得分
                (text_width, text_height), _ = cv2.getTextSize(str(score), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                cv2.putText(img_orige, str(score), (max(x2, 5), max(y1, 5)), cv2.FONT_HERSHEY_SIMPLEX, scale, color_maps[cls_id], 1)
                
        # os.makedirs(os.path.join(args.show_dir, video_name), exist_ok=True)
        # cv2.imwrite(os.path.join(args.show_dir, video_name, f"{frame_ids[0]:06d}.jpg"), img_orige)
        video_write.write(img_orige)
    else:
        coco_evaluator.update_metrics(preds, file_names)

coco_evaluator.save_json(args.pred_json)


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


