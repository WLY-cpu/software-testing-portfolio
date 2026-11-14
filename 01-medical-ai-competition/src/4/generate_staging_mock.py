import os
import cv2
import pandas as pd
import numpy as np
import random
LESION_IMG_DIR = '../3/data/images/train'
LESION_LABEL_DIR = '../3/data/labels/train'
OUTPUT_CSV = 'staging_labels.csv'
random.seed(42)
np.random.seed(42)
samples = []
for label_file in os.listdir(LESION_LABEL_DIR):
    if not label_file.endswith('.txt'):
        continue
    img_name = label_file.replace('.txt', '.png')
    img_path = os.path.join(LESION_IMG_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    with open(os.path.join(LESION_LABEL_DIR, label_file), 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id, xc, yc, bw, bh = map(float, parts)
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        size_gt3cm = random.randint(0, 1)
        chest_wall = random.randint(0, 1)
        obstruct_pneumonia = random.randint(0, 1)
        lymph_node = random.randint(0, 1)
        contralateral = random.randint(0, 1)
        bone_dest = random.randint(0, 1)
        has_tumor = 1 if int(cls_id) in (8, 9) else 0
        samples.append({
            'filename': img_name,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'size_gt3cm': size_gt3cm,
            'chest_wall_invasion': chest_wall,
            'obstructive_pneumonia': obstruct_pneumonia,
            'lymph_node_enlargement': lymph_node,
            'contralateral_lesion': contralateral,
            'bone_destruction': bone_dest,
            'has_tumor': has_tumor
        })
df = pd.DataFrame(samples)
df.to_csv(OUTPUT_CSV, index=False)
print(f"生成 {len(df)} 个病灶标注样本，保存至 {OUTPUT_CSV}")
