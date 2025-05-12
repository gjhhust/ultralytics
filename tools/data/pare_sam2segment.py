import cv2
import numpy as np

# 读取"/data/jiahaoguo/datasets/MOSE/train/Annotations/001ca3cb/00000.png"的分割标注图
ann = cv2.imread("/data/jiahaoguo/datasets/MOSE/train/Annotations/001ca3cb/00000.png")
unique = np.unique(ann)
print(unique)