import json
import os
import numpy as np
from collections import defaultdict
def map_track_ids(result_anns, gt_data):
    
    
    image_video_id = {}
    for image in gt_data['images']:
        image_video_id[image['id']] = image['video_id']
    
    anns_videos_map = defaultdict(list)
    for ann in result_anns:
        ann['video_id'] = image_video_id[ann['image_id']]
        anns_videos_map[ann['video_id']].append(ann)
        
    new_result_anns=[]
    for video_id, video_anns in anns_videos_map.items():
        for i, ann in enumerate(video_anns):
            ann['track_id'] = video_id

    return new_result_anns

gt = '/data/jiahaoguo/ultralytics_yoloft/ultralytics/tools/gaode_time/tao_gt/task1_test.json'
with open("runs/detect/val133/predictions.json", 'rb') as f:
    pred_data = json.load(f)
with open(gt, 'rb') as f:
    gt_data = json.load(f)
pred_data = map_track_ids(pred_data, gt_data)    


with open("/data/jiahaoguo/ultralytics_yoloft/ultralytics/tools/gaode_time/results/predictions_unique.json", 'w') as f:
    json.dump(pred_data, f)
