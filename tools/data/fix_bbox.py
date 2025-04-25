import json
json_path = '/data/jiahaoguo/datasets/XS-VIDv2/annotations/jsonv2/test_segment.json'
with open(json_path, 'r') as f:
    data = json.load(f)

def check_ann(ann, img_h=1024, img_w=1024):
    # 检查 bbox 是否有效
    x1, y1, w, h = ann['bbox']
    x1 = max(min(x1, img_w-1), 0)
    y1 = max(min(y1, img_h-1), 0)
    x2 = min(x1 + w, img_w-1)
    y2 = min(y1 + h, img_h-1)
    # w = x2 - x1
    # h = y2 - y1
    return [x1, y1, x2 - x1, y2 - y1]

anns = []
for ann in data['annotations']:
    ann["bbox"] = check_ann(ann)
    if ann["bbox"][2]<=0 or ann["bbox"][3]<=0:
        print(f"Error: bbox is negative: {ann}")
        continue
            # continue
    anns.append(ann)

for i, ann in enumerate(anns):
    ann['id'] = i
data['annotations'] = anns

with open(json_path, 'w') as f:
    json.dump(data, f)