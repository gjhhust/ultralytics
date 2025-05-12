from pycocotools.coco import COCO  # noqa
# from pycocotools.cocoeval import COCOeval  # noqa
from ultralytics.data.cocoeval_xs_vid import COCOeval  # noqa
import os, json, pickle
# anno_json = "/data/jiahaoguo/datasets/speed_merge/merge_test_1.json"
# pred_json = "/data/jiahaoguo/YOLOFT/yoloft/train53/predictions.json"
# anno_json = "/data/jiahaoguo/datasets/gaode_6/annotations/mini_val/gaode_6_mini_val.json"
anno_json = "/data/jiahaoguo/datasets/XS-VIDv2/annotations/fix/test.json"
# anno_json = "/data/jiahaoguo/datasets/gaode_6/annotations/task1_2videos.json"
pred_json = "/data/jiahaoguo/ultralytics_yoloft/ultralytics/results/[Dfine-L]results_54.10.json"

# 27.8 38.5 34.0 0.5 9.5 53.2 26.3 66.4 43.3
# 8.9 13.8 9.2 1.0 11.0 19.3 3.2 27.5 27.6

def xyxy2xywh(bbox):
    x,y,x2,y2 = bbox
    return [x, y, x2-x, y2-y]
# 加载 pickle 文件
if os.path.splitext(pred_json)[-1] == ".pkl":
    results = []
    with open(pred_json, 'rb') as f:
        pred_data = pickle.load(f)
    with open(anno_json, 'rb') as f:
        anno_data = json.load(f)

    for i, img in enumerate(anno_data["images"]):
        pred_bboxes = pred_data['det_bboxes'][i]
        for category_id, bboxes in enumerate(pred_bboxes):
            for bbox in bboxes.tolist():
                results.append({
                    "image_id":img["id"],
                    "bbox": xyxy2xywh(bbox[:4]),
                    "category_id": category_id,
                    "score":bbox[-1]
                })
    # 假设 pred_data 是一个符合 COCO 格式的列表，将其保存为 JSON 文件
    pred_json = os.path.splitext(pred_json)[0] + ".json"
    with open(pred_json, 'w') as f:
        json.dump(results, f)

anno = COCO(str(anno_json))  # init annotations api
pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
eval = COCOeval(anno, pred, 'bbox')

# 仅对类别1进行评估
# eval.params.catIds = [1]  # 设定类别ID为1

eval.evaluate()
eval.accumulate()
eval.summarize()
def print_formatted_stats(stats):
    # 数字乘以100后保留一位小数
    formatted_stats = [f"{num * 100:.1f}" for num in stats]
    # 使用空格连接数组中的元素并打印
    print(" ".join(formatted_stats))
stats_result =  print_formatted_stats(eval.stats)
# print(eval.stats)
print(pred_json)







# from pycocotools.coco import COCO  # noqa
# from ultralytics.data.cocoeval import COCOeval  # noqa

# anno_json = "/data/jiahaoguo/datasets/speed_merge/merge_test_1.json"
# pred_json = "/data/jiahaoguo/YOLOFT/yoloft/train53/predictions.json"

# # 加载 COCO 数据集的注释和预测
# anno = COCO(str(anno_json))  # 初始化注释 API
# pred = anno.loadRes(str(pred_json))  # 初始化预测 API

# # 实例化 COCOeval
# eval = COCOeval(anno, pred, 'bbox')

# # 过滤指定范围内的 images
# eval.params.imgIds = [img_id for img_id in anno.getImgIds() if 20793 <= img_id <= 31521]

# # 进行评估
# eval.evaluate()
# eval.accumulate()
# eval.summarize()

# print(eval.stats)
# print(pred_json)
