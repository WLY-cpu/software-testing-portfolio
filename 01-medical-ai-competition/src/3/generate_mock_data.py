import os
import cv2
import numpy as np
import random
"""生成模拟 YOLO 数据集（本地联调用，不参与正式训练）。"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
REAL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../1/data/authenticity/train/0_real"))
N_TRAIN, N_VAL = 20, 5
def load_real_images():
    """优先复用第一关训练集的真实胸片，没有则返回空列表。"""
    if not os.path.isdir(REAL_DIR):
        return []
    files = sorted(f for f in os.listdir(REAL_DIR) if f.lower().endswith(".png"))
    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(REAL_DIR, f))
        if img is not None:
            imgs.append(img)
        if len(imgs) >= N_TRAIN + N_VAL:
            break
    return imgs
def synthetic_image(h=512, w=512):
    """无真实胸片时生成带纹理的合成图像，保证流程可跑通。"""
    img = np.random.randint(60, 130, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 18)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 2), 0, 0, 360,
                (160, 160, 160), -1)
    return img
def main():
    random.seed(42)
    np.random.seed(42)
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATA_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA_ROOT, "labels", split), exist_ok=True)
    real_imgs = load_real_images()
    print(f"找到 {len(real_imgs)} 张真实胸片，其余将使用合成图像。")
    idx = 0
    for split, n in (("train", N_TRAIN), ("val", N_VAL)):
        for _ in range(n):
            idx += 1
            img = real_imgs[idx - 1].copy() if idx <= len(real_imgs) else synthetic_image()
            h, w = img.shape[:2]
            img_path = os.path.join(DATA_ROOT, "images", split, f"mock_{idx:04d}.png")
            label_path = os.path.join(DATA_ROOT, "labels", split, f"mock_{idx:04d}.txt")
            cv2.imwrite(img_path, img)
            num_boxes = random.randint(0, 3)
            with open(label_path, "w", encoding="utf-8") as f:
                for _ in range(num_boxes):
                    cls_id = random.choice(list(range(17)))
                    bw = random.randint(30, min(100, w // 2))
                    bh = random.randint(30, min(100, h // 2))
                    bx = random.randint(0, max(0, w - bw))
                    by = random.randint(0, max(0, h - bh))
                    x_center = (bx + bw / 2) / w
                    y_center = (by + bh / 2) / h
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} "
                            f"{bw / w:.6f} {bh / h:.6f}\n")
    print(f"模拟数据集生成完毕: {DATA_ROOT} (train={N_TRAIN}, val={N_VAL})")
if __name__ == "__main__":
    main()
