import json
import torch
from torchvision.ops import nms
from collections import defaultdict
import time

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def non_max_suppression(
        predictions,
        conf_thres=0.25,
        iou_thres=0.7,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
        classes=None,
        agnostic=False
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes.

    Arguments:
        predictions (List[dict]): List of predictions for each image.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
        max_det (int): The maximum number of boxes to keep after NMS.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all classes will be considered as one.

    Returns:
        (List[dict]): List of kept boxes after NMS.
    """
    output = []

    # 按照image_id分组
    grouped_data = defaultdict(list)
    for item in predictions:
        grouped_data[item['image_id']].append(item)

    for image_id, preds in grouped_data.items():
        # 按类别分组
        category_grouped_data = defaultdict(list)
        for pred in preds:
            category_grouped_data[pred['category_id']].append(pred)
        
        for category_id, category_preds in category_grouped_data.items():
            bboxes = torch.tensor([xywh_to_xyxy(pred['bbox']) for pred in category_preds], dtype=torch.float32)
            scores = torch.tensor([pred['score'] for pred in category_preds], dtype=torch.float32)
            
            # Apply confidence threshold
            mask = scores > conf_thres
            bboxes = bboxes[mask]
            scores = scores[mask]
            category_preds = [pred for i, pred in enumerate(category_preds) if mask[i]]

            # If none remain process next image
            if len(scores) == 0:
                continue
            
            # Apply NMS
            keep = nms(bboxes, scores, iou_thres)
            keep = keep[:max_det]  # limit detections

            for idx in keep:
                pred = category_preds[idx]
                pred['bbox_xyxy'] = [int(b) for b in bboxes[idx].tolist()]
                output.append(pred)

    return output

# 读取json文件
with open("/data/shuzhengwang/project/ultralytics/runs/save/train81_9.5/val24/predictions.json", 'r') as f:
    data = json.load(f)

# 执行NMS
results = non_max_suppression(data, conf_thres=0.25, iou_thres=0.45, max_det=300)

# 保存结果到新的json文件中
with open('/data/shuzhengwang/project/ultralytics/runs/save/train81_9.5/val24/prediction_nms.json', 'w') as f:
    json.dump(results, f, indent=2)

print("NMS处理完成，并保存为 .json")
