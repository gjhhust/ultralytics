from ultralytics.models.sam import SAM2VideoPredictor
import cv2
import os
import tempfile
import shutil
import json
from tqdm import tqdm
def xywh_to_xyxy(bbox):
    """
    将 [x1, y1, w, h] 格式的 bbox 转换为 [x1, y1, x2, y2] 格式的整数列表。

    参数:
        bbox (list[float]): 输入的 bbox，格式为 [x1, y1, w, h]。

    返回:
        list[int]: 转换后的 bbox，格式为 [x1, y1, x2, y2]。
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def frames_to_video(frames_dir, output_path, fps=25):
    # 获取帧文件列表
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(
        frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # 读取第一帧以获取尺寸
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # 定义视频编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入每一帧到视频文件
    for frame_file in tqdm(frame_files, desc=f"保存视频中: {os.path.basename(output_path)}"):
        frame = cv2.imread(frame_file)
        out.write(frame)

    # 释放 VideoWriter 对象
    out.release()


def process_frames_in_directory(images_dir, annotation_file):
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 读取注释文件
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)

    videos = annotation_data.get('videos', [])
    annotations = annotation_data.get('annotations', [])

    for video in tqdm(videos, desc=f"总视频处理进度"):
        video_id = video['id']
        video_folder_path = os.path.join(images_dir, video['name'])

        assert os.path.isdir(video_folder_path)
        # 生成输出视频的路径
        output_video_path = os.path.join(
            "/data/shuzhengwang/project/ultralytics/runs/sam2", f"{video_id}.mp4")
        # 将帧转换为视频
        if not os.path.exists(output_video_path):
            frames_to_video(video_folder_path, output_video_path)
        print(f"Video saved to {output_video_path}")
        
        # 记录每个 instance_id 出现的第一帧数、bbox 和 category_id
        frame_info = {}
        instance_seen = set()
        for annotation in annotations:
            if annotation['video_id'] == video_id:
                instance_id = annotation['instance_id']
                frame_id = annotation['frame_id']
                bbox = xywh_to_xyxy(annotation['bbox'])
                category_id = annotation['category_id']

                if instance_id not in instance_seen:
                    if frame_id not in frame_info:
                        frame_info[frame_id] = {
                            "bboxes": [],
                            "labels": [],
                            "instance_ids": []
                        }
                    frame_info[frame_id]["bboxes"].append(bbox)
                    frame_info[frame_id]["labels"].append(category_id)
                    frame_info[frame_id]["instance_ids"].append(instance_id)
                    instance_seen.add(instance_id)
                    
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt", device=[0])
        predictor = SAM2VideoPredictor(overrides=overrides)

        # Run inference with single point
        results = predictor(source=output_video_path, frame_prompt_info=frame_info, muti_frame_prompt=True)
        
        print(f"Instance first frame info for video {video_id}: {len(results)}")
        
        # 这里可以添加后续处理代码，使用 output_video_path 和 instance_first_frame
        print("后续处理代码可以在这里添加...")
    
    # 删除临时目录
    shutil.rmtree(temp_dir)
    print("临时目录已删除")


if __name__ == "__main__":
    images_dir = "/data/shuzhengwang/datasets/XS-VID/images"
    annotation_file = "/data/shuzhengwang/datasets/XS-VID/annotations/test.json"
    process_frames_in_directory(images_dir, annotation_file)
    
    
# Create SAM2VideoPredictor


# # Run inference with multiple points
# results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

# # Run inference with multiple points prompt per object
# results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

# # Run inference with negative points prompt
# results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])