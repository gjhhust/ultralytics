# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import os,json
from ultralytics import SAM, YOLO
from collections import defaultdict
import torch
import cv2
from tqdm import tqdm

def results_compose(batch):
    """
    将 list[dict] 转换为 dict[list]，不依赖字典键的顺序。
    
    Args:
        batch: List[Dict], 输入字典列表，所有字典必须有相同的键（顺序无关）。
    
    Returns:
        Dict[List], 键来自输入字典，值为对应位置的值列表。
    """
    if not batch:
        return {}
    
    # 获取第一个字典的所有键（顺序无关）
    keys = set(batch[0].keys())
    
    # 验证所有字典的键是否一致
    for d in batch[1:]:
        if set(d.keys()) != keys:
            raise ValueError("输入字典的键不一致！")
    
    # 初始化结果字典
    new_batch = {k: [] for k in keys}
    
    # 填充值列表
    for d in batch:
        for k in keys:
            new_batch[k].append(d[k])
    
    return new_batch

import torch

def xywh2xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """
    将 [n, 4] 的 xywh 格式边界框转换为 xyxy 格式
    Args:
        bboxes: torch.Tensor, shape [n, 4], 格式为 [x_top, y_left, width, height]
    Returns:
        torch.Tensor, shape [n, 4], 格式为 [x1, y1, x2, y2]
    """
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.tensor(bboxes)
    
    # 确保输入是二维张量 [n, 4]
    if bboxes.dim() == 1:
        bboxes = bboxes.unsqueeze(0)
    
    # 转换逻辑
    x1 = bboxes[..., 0]  # x1 = x_top
    y1 = bboxes[..., 1]  # y1 = y_left
    x2 = bboxes[..., 0] + bboxes[..., 2]  # x2 = x_top + width
    y2 = bboxes[..., 1] + bboxes[..., 3]  # y2 = y_left + height
    
    return torch.stack([x1, y1, x2, y2], dim=-1)

def auto_annotate(
    images_dir,
    ann_file,
    sam_model="sam_b.pt",
    device="",
    output_dir=None,
    show = False
):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model; default is 0.25.
        iou (float): IoU threshold for filtering overlapping boxes in detection results; default is 0.45.
        imgsz (int): Input image resize dimension; default is 640.
        max_det (int): Limits detections per image to control outputs in dense scenes.
        classes (list): Filters predictions to specified class IDs, returning only relevant detections.
        output_dir (str | None): Directory to save the annotated results. If None, a default directory is created.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")

    Notes:
        - The function creates a new directory for output if not specified.
        - Annotation results are saved as text files with the same names as the input images.
        - Each line in the output text file represents a detected object with its class ID and segmentation points.
    """
    with open(ann_file, 'rb') as f:
        anno_data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    sam_model = SAM(sam_model)

    name_to_id = {vid["id"]: vid["name"] for vid in anno_data["videos"]}

    det_results = defaultdict(lambda: defaultdict(list))
    for ann in anno_data["annotations"]:
        det_results[ann["video_id"]][ann["frame_id"]].append(ann)

    for video_id, current_det_results in tqdm(det_results.items(),total=len(det_results), desc=f"Annotating videos"):
        video_name = name_to_id[video_id]
        video_dir = os.path.join(images_dir, video_name)
        current_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(current_output_dir, exist_ok=True)

    # det_results = det_model(
    #     data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    # )
        if show:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
            video_writer = cv2.VideoWriter(
                filename=os.path.join(output_dir, f"{video_name}.mp4"),
                fourcc=fourcc,
                fps=25,
                frameSize=(1024,1024),
            )
    
        for frame_id, result in tqdm(current_det_results.items(), total=len(current_det_results), desc=f"{video_name}: "):
            frame_name = f"{frame_id:07d}.jpg"
            orig_img = cv2.imread(os.path.join(video_dir, frame_name))
            assert orig_img is not None, f"Failed to load image {os.path.join(video_dir, f'{frame_id:07d}.jpg')}"
            result_ = results_compose(result)
            class_ids = result_['category_id']  # noqa
            if len(class_ids):
                boxes = xywh2xyxy(torch.tensor(result_['bbox'])).int().tolist()  # Boxes object for bbox outputs
                sam_results = sam_model(orig_img, bboxes=boxes, verbose=False, save=False, device=device)
                if show:
                    _, im = sam_results[0].save(filename="result.jpg")
                    video_writer.write(im)
                segments = sam_results[0].masks.xyn  # noqa

                with open(f"{Path(current_output_dir) / Path(frame_name).stem}.txt", "w") as f:
                    for i in range(len(segments)):
                        s = segments[i]
                        if len(s) == 0:
                            continue
                        segment = map(str, segments[i].reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
        if show:
            video_writer.release()


def auto_annotate_by_model(
    images_dir,
    ann_file=None,
    sam_model="sam_b.pt",
    device="",
    output_dir=None,
    show=False,
):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates
    segmentation masks using a SAM model. The resulting annotations are saved as text files.

    Args:
        data (str): Path to a folder containing images to be annotated.
        det_model (str): Path or name of the pre-trained YOLO detection model.
        sam_model (str): Path or name of the pre-trained SAM segmentation model.
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').
        conf (float): Confidence threshold for detection model; default is 0.25.
        iou (float): IoU threshold for filtering overlapping boxes in detection results; default is 0.45.
        imgsz (int): Input image resize dimension; default is 640.
        max_det (int): Limits detections per image to control outputs in dense scenes.
        classes (list): Filters predictions to specified class IDs, returning only relevant detections.
        output_dir (str | None): Directory to save the annotated results. If None, a default directory is created.

    Examples:
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")

    Notes:
        - The function creates a new directory for output if not specified.
        - Annotation results are saved as text files with the same names as the input images.
        - Each line in the output text file represents a detected object with its class ID and segmentation points.
    """
    det_model = "runs/detect/train343/weights/best.pt"
    imgsz=1024
    conf=0.25
    iou=0.45
    max_det=300
    classes = None

    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    # 方法1：使用os.listdir+os.path.isdir
    for item in tqdm(os.listdir(images_dir), total=294):
        full_path = os.path.join(images_dir, item)
        video_output_dir = os.path.join(output_dir, item)

        if show:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
            video_writer = cv2.VideoWriter(
                filename=os.path.join(output_dir, f"{item}.mp4"),
                fourcc=fourcc,
                fps=25,
                frameSize=(1024,1024),
            )
    
        data = Path(full_path)
        if not video_output_dir:
            video_output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
        Path(video_output_dir).mkdir(exist_ok=True, parents=True)

        det_results = det_model(
            data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
        )

        for result in det_results:
            class_ids = result.boxes.cls.int().tolist()  # noqa
            if len(class_ids):
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
                sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)

                if show:
                    _, im = sam_results[0].save(filename="result.jpg")
                    video_writer.write(im)
                segments = sam_results[0].masks.xyn  # noqa

                with open(f"{Path(video_output_dir) / Path(result.path).stem}.txt", "w") as f:
                    for i in range(len(segments)):
                        s = segments[i]
                        if len(s) == 0:
                            continue
                        segment = map(str, segments[i].reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
        if show:
            video_writer.release()

# auto_annotate("/data/shuzhengwang/datasets/XS-VID/images", 
#               ann_file="/data/shuzhengwang/datasets/XS-VID/annotations/fix/train.json", 
#               output_dir = "/data/shuzhengwang/datasets/XS-VID/annotations/segment_yolox/", show=True)
auto_annotate_by_model("/data/shuzhengwang/datasets/XS-VID/images", 
              ann_file="/data/shuzhengwang/datasets/XS-VID/annotations/fix/train.json", 
              output_dir = "/data/shuzhengwang/datasets/XS-VID/annotations/segment_yolox_model/",
              sam_model="sam2_b.pt",
              show=True)