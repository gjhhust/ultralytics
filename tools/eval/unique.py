import json
import os
import numpy as np
from collections import defaultdict
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

with open("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/track/run20/tracks_yolov8l.json", 'rb') as f:
    pred_data = json.load(f)

pred_data = map_track_ids(pred_data)    


with open("/data/jiahaoguo/ultralytics_yoloft/ultralytics/runs/track/run20/tracks_yolov8l_unique.json", 'w') as f:
    json.dump(pred_data, f)
