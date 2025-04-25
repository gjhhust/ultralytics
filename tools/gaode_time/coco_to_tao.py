import json
from collections import defaultdict
def generate_new_json(input_json_path, output_json_path):
    # 读取输入 JSON 文件
    with open(input_json_path, 'r') as infile:
        data = json.load(infile)

    for image in data["images"]:
        image["frame_index"] = image["frame_id"]

    # 复制 videos 部分
    for video in data["videos"]:
        video["width"] = 800
        video["height"] = 600
        video["neg_category_ids"] =  []
        video["not_exhaustive_category_ids"] = []
    
    # 初始化全局 track_id 和 tracks 列表
    global_track_id = 0
    tracks = []
    inscentence_to_track = {}

    video_id_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        video_id_to_anns[ann["video_id"]].append(ann)
    
    video_id_to_images = defaultdict(list)
    for img in data["images"]:
        video_id_to_images[img["video_id"]].append(ann)

    annota_new = []
    for video_id in list(video_id_to_anns.keys()):
        anns = video_id_to_anns[video_id]
        imgs = video_id_to_images[video_id]
        if len(anns) > len(imgs):
            continue
        
        tracks.append({
                "id": global_track_id,
                "category_id": anns[0]["category_id"],
                "video_id": video_id
            })
        
        for ann in anns:
            ann["track_id"] = global_track_id
            annota_new.append(ann)
            
        global_track_id+=1
            
            

    # 将 tracks 添加到 JSON 数据中
    data["tracks"] = tracks
    data["annotations"] = annota_new
    # 写入输出 JSON 文件
    with open(output_json_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    input_json_path = '/data/jiahaoguo/dataset/gaode_6/annotations/task1/task1_test.json'
    output_json_path = '/data/jiahaoguo/ultralytics_yoloft/ultralytics/tools/gaode_time/tao_gt/task1_test.json'
    generate_new_json(input_json_path, output_json_path)
