# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import os,json
from ultralytics import SAM, YOLO
from collections import defaultdict
import torch
import cv2
from tqdm import tqdm

import os
import re
from pathlib import Path
from eval.tao import TaoEval
import logging
# 获取根日志记录器
root_logger = logging.getLogger()
# 修改日志级别为INFO
root_logger.setLevel(logging.INFO)


def get_current_run_dir(output_dir="output_dir"):
    """
    自动检测output_dir下的run*文件夹，创建下一个编号的run文件夹，并返回路径。
    
    参数:
        output_dir (str): 父目录路径，默认为"output_dir"
    
    返回:
        str: 新创建的run文件夹的完整路径（如"output_dir/run4"）
    """
    # 确保父目录存在
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 正则匹配所有run{number}文件夹并提取数字
    pattern = re.compile(r'^run(\d+)$')  # 匹配run开头+纯数字的文件夹名
    run_numbers = [
        int(match.group(1)) 
        for d in output_path.iterdir() 
        if d.is_dir() and (match := pattern.match(d.name))
    ]
    
    # 计算下一个编号（如果没有匹配项则从0开始）
    next_num = max(run_numbers) + 1 if run_numbers else 0
    
    # 创建新文件夹
    current_run_dir = output_path / f"run{next_num}"
    current_run_dir.mkdir(exist_ok=False)  # 如果已存在会报错，避免覆盖
    
    return str(current_run_dir)

def xyxy2xywh(xyxy):
    """
    将边界框从 [x1, y1, x2, y2] 格式转换为 [x, y, w, h] 格式
    
    参数:
        xyxy (list/tuple/numpy.ndarray): 输入边界框，格式为 [x1, y1, x2, y2]
    
    返回:
        list: 转换后的边界框，格式为 [x, y, w, h]
    """
    x1, y1, x2, y2 = xyxy  # 解构坐标
    x = x1                  # 左上角x坐标
    y = y1                  # 左上角y坐标
    w = x2 - x1             # 宽度
    h = y2 - y1             # 高度
    return [x, y, w, h]

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

def map_track_ids(result_anns):
    video_track_mapping = {}
    new_result_anns = []
    new_track_id = 0
    for ann in result_anns:
        video_id = ann['video_id']
        track_id = ann['track_id']
        unique_id = (video_id, track_id)
        if unique_id not in video_track_mapping:
            # 为每个视频的每个 track_id 分配一个新的唯一 ID
            video_track_mapping[unique_id] = {
                'track_id': new_track_id,
                "category_id": ann['category_id'],
            }
            new_track_id += 1

        new_ann = ann.copy()
        new_ann['track_id'] = video_track_mapping[unique_id]["track_id"]
        new_ann['category_id'] = video_track_mapping[unique_id]["category_id"]
        new_result_anns.append(new_ann)
    return new_result_anns

def track_annotate(
    images_dir,
    ann_file,
    detect_model="hyper-yoloS_best.pt",
    tracker_yaml="bytetrack.yaml",
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
    model = YOLO(detect_model)
    with open(ann_file, 'rb') as f:
        anno_data = json.load(f)
    current_run_dir = get_current_run_dir(output_dir)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"tracking images_dir:{images_dir}, and results will save in {current_run_dir}")

    name_to_id = {vid["id"]: vid["name"] for vid in anno_data["videos"]}

    video_images = defaultdict(list)
    for img in anno_data["images"]:
        video_images[img["video_id"]].append(img)
    # 对每个 video_id 的图像列表按照 frame_id 排序
    for video_id in video_images:
        video_images[video_id].sort(key=lambda x: x["frame_id"])

    track_json_results = []
    for video_id, frames in tqdm(video_images.items(),total=len(video_images), desc=f"tracking videos"):
        video_name = name_to_id[video_id]

        if show:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
            video_writer = cv2.VideoWriter(
                filename=os.path.join(current_run_dir, f"{video_name}.mp4"),
                fourcc=fourcc,
                fps=25,
                frameSize=(1024,1024),
            )

        for frame_info in tqdm(frames, total=len(frames), desc=f"{video_name}: "):
            frame_path = os.path.join(images_dir, frame_info["file_name"])
            # orig_img = cv2.imread(frame_path)
            # assert orig_img is not None, f"Failed to load image {frame_path}"

            result = model.track(frame_path, tracker=tracker_yaml, persist=True)[0]

            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                cls = result.boxes.cls.cpu().tolist()
                conf = result.boxes.conf.cpu().tolist()
                track_ids = result.boxes.id.int().cpu().tolist()
                for i, box in enumerate(boxes):
                    track_json_results.append({
                        "image_id" : frame_info["id"],
                        "category_id" : cls[i],
                        "bbox" : xyxy2xywh(box),
                        "score" : conf[i],
                        "track_id": track_ids[i],
                        "video_id": video_id
                    })

                if show:
                    annotated_frame = result.plot()
                    video_writer.write(annotated_frame)

        if show:
            video_writer.release()
    
    track_json_results = map_track_ids(track_json_results)
    result_path = os.path.join(current_run_dir, "tracks.json")
    with open(result_path, "w") as f:
        json.dump(track_json_results, f)
        
    return result_path

ann_file = "/data/jiahaoguo/ultralytics_yoloft/ultralytics/tools/gaode_time/tao_gt/task1_test.json"

result_path = track_annotate(images_dir = "/data/jiahaoguo/dataset/gaode_6/images/", 
                detect_model="runs/save/train217_yolos_newdata/weights/best.pt",
                tracker_yaml="bytetrack.yaml",
                ann_file=ann_file, 
                output_dir = "runs/track/", show=False)
    
    
    # TAO uses logging to print results. Make sure logging is set to show INFO
    # messages, or you won't see any evaluation results.
print(f"Evaluating detect_model is ")
tao_eval = TaoEval(ann_file,
                result_path, 
                )
tao_eval.run()
tao_eval.print_results()

