import os
import cv2
import numpy as np
import random
from tqdm import tqdm
REAL_DIR = '../1/data/real'
OUTPUT_DIR = './data/splicing'
TRAIN_RATIO = 0.8
def make_spliced_xray(real_img1, real_img2):
    h, w = real_img1.shape[:2]
    splice_type = random.choice(['vertical', 'horizontal', 'patch'])
    if splice_type == 'vertical':
        x_split = random.randint(w//3, 2*w//3)
        left_part = real_img1[:, :x_split]
        right_part = real_img2[:, x_split:]
        spliced = np.hstack([left_part, right_part])
    elif splice_type == 'horizontal':
        y_split = random.randint(h//3, 2*h//3)
        top_part = real_img1[:y_split, :]
        bottom_part = real_img2[y_split:, :]
        spliced = np.vstack([top_part, bottom_part])
    else:
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(x1+w//4, w), random.randint(y1+h//4, h)
        patch = real_img1[y1:y2, x1:x2]
        spliced = real_img2.copy()
        spliced[y1:y2, x1:x2] = patch
    spliced = cv2.GaussianBlur(spliced, (3,3), 0.5)
    return spliced
def main():
    random.seed(42)
    np.random.seed(42)
    real_images = []
    for root, _, files in os.walk(REAL_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                real_images.append(os.path.join(root, f))
    if len(real_images) < 2:
        raise RuntimeError("需要至少两张真实胸片才能生成拼接图！")
    print(f"找到 {len(real_images)} 张真实胸片用于生成拼接数据")
    random.shuffle(real_images)
    split_idx = int(len(real_images) * TRAIN_RATIO)
    train_files = real_images[:split_idx]
    val_files = real_images[split_idx:]
    for split, files in [('train', train_files), ('val', val_files)]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, '0_real'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, '1_spliced'), exist_ok=True)
        for f in tqdm(files, desc=f'Copying {split} real'):
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(OUTPUT_DIR, split, '0_real', os.path.basename(f)), img)
        for i, f in enumerate(tqdm(files, desc=f'Generating {split} spliced')):
            img1 = cv2.imread(f)
            if img1 is None:
                continue
            img1 = cv2.resize(img1, (512, 512))
            other_files = [x for x in files if x != f]
            if not other_files:
                other_files = files
            partner = random.choice(other_files)
            img2 = cv2.imread(partner)
            img2 = cv2.resize(img2, (512, 512))
            spliced = make_spliced_xray(img1, img2)
            out_name = f'spliced_{i}.png'
            cv2.imwrite(os.path.join(OUTPUT_DIR, split, '1_spliced', out_name), spliced)
    print("拼接数据集生成完成！")
if __name__ == '__main__':
    main()
