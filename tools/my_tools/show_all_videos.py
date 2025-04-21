import json
import cv2
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# 配置路径
pred_path = '/data/shuzhengwang/project/ultralytics/runs/save/train81_9.5/val24/prediction_nms.json'
anno_path = "/data/jiahaoguo/dataset/gaode_6/annotations/mini_val/gaode_6_mini_val.json"
image_base = "/data/jiahaoguo/dataset/gaode_6/images/"
output_dir = "./visualization_videos/"
os.makedirs(output_dir, exist_ok=True)

# 加载注释数据
with open(anno_path, 'r') as f:
    coco_anno = json.load(f)

# 构建image_id到文件名的映射
image_id_map = {img['id']: img['file_name'] for img in coco_anno['images']}

# 加载预测结果
with open(pred_path, 'r') as f:
    predictions = json.load(f)

# 按视频和帧号组织预测结果
video_frame_dict = defaultdict(lambda: defaultdict(list))

for pred in predictions:
    image_id = pred['image_id']
    file_name = image_id_map[image_id]
    
    # 解析视频名称和帧号（假设文件名格式：video_name/frame_xxxxxx.jpg）
    video_name, frame_part = os.path.split(file_name)
    frame_num = int(os.path.splitext(frame_part)[0])
    
    # 存储预测信息
    video_frame_dict[video_name][frame_num].append({
        'bbox': pred['bbox'],  # [x,y,w,h]
        'score': pred['score'],
        'category_id': pred['category_id'],
        "file_name": file_name,
    })

# 可视化参数设置
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)  # COCO类别颜色
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 25  # 视频帧率
VIDEO_SIZE = (1280, 720)  # 视频分辨率（根据实际调整）

# 处理每个视频
for video_name, frame_data in tqdm(video_frame_dict.items(), desc="Processing Videos"):
    # 获取所有帧号并排序
    sorted_frames = sorted(frame_data.keys())
    
    # 创建视频写入对象
    video_writer =  None
    
    
    
    # 遍历每一帧
    for frame_num in tqdm(sorted_frames, desc=f"Processing {video_name}"):
        # 获取对应图片路径
        file_name = video_frame_dict[video_name][frame_num][0]["file_name"]  # 假设同视频文件名格式一致
        img_path = os.path.join(image_base, file_name)
        
        if not os.path.exists(img_path):
            print(f"Missing frame: {img_path}")
            continue
            
        # 读取图片并调整尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        if video_writer is None:
           video_writer =  cv2.VideoWriter(
                os.path.join(output_dir, f"{video_name}.mp4"),
                cv2.VideoWriter_fourcc(*'mp4v'),
                FPS,
                img.shape[1::-1]
            )
        
        # 绘制预测框
        for pred in frame_data[frame_num]:
            x, y, w, h = map(int, pred['bbox'])
            cat_id = pred['category_id']
            score = pred['score']
            
            # 绘制矩形和文字
            color = tuple(map(int, COLORS[cat_id]))
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            label = f"Cls:{cat_id} {score:.2f}"
            cv2.putText(img, label, (x, y-10), FONT, 0.5, color, 1)
        
        # 写入视频帧
        video_writer.write(img)
    
    # 释放视频写入资源
    video_writer.release()

print(f"All videos saved to {output_dir}")