import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageEnhance
REAL_DIR = "./data/real"
OUTPUT_DIR = "./data/authenticity"
NONHUMAN_SOURCE_DIR = "./data/nonhuman_raw"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
SCREEN_CAPTURE_COUNT = 3
BODY_PHANTOM_COUNT = 1
def add_lcd_moire(img, angle_range=(-30,30), spacing_range=(5,15)):
    h, w = img.shape[:2]
    angle = np.deg2rad(np.random.uniform(*angle_range))
    spacing = np.random.uniform(*spacing_range)
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    grid = 0.5 + 0.5 * np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / spacing)
    grid = np.clip(grid, 0.8, 1.2).astype(np.float32)
    img = img.astype(np.float32) * np.expand_dims(grid, -1)
    return np.clip(img, 0, 255).astype(np.uint8)
def add_glare_patch(img, size_range=(50,150)):
    h, w, _ = img.shape
    size = np.random.randint(*size_range)
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    mask = np.zeros((h, w), dtype=np.float32)
    y1, y2 = max(0, cy-size//2), min(h, cy+size//2)
    x1, x2 = max(0, cx-size//2), min(w, cx+size//2)
    mask[y1:y2, x1:x2] = np.random.uniform(0.2, 0.6)
    img = img.astype(np.float32) + mask[:,:,None] * 255
    return np.clip(img, 0, 255).astype(np.uint8)
def screen_capture_pipeline(img):
    if random.random() > 0.5:
        scale = random.uniform(0.7, 0.95)
        new_size = (int(img.shape[1]*scale), int(img.shape[0]*scale))
        img = cv2.resize(img, new_size)
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    if random.random() > 0.7:
        img = add_lcd_moire(img, angle_range=(-40,40), spacing_range=(3,20))
    if random.random() > 0.3:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(random.uniform(0.4, 1.6))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if random.random() > 0.5:
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), random.uniform(-8,8), random.uniform(0.85,1.15))
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if random.random() > 0.4:
        img = add_glare_patch(img)
    if random.random() > 0.3:
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 85)])
        img = cv2.imdecode(enc, 1)
    return img
def simulate_phantom(img):
    img = cv2.GaussianBlur(img, (15,15), random.uniform(2,5))
    img = cv2.addWeighted(img, random.uniform(0.5, 0.8), np.zeros_like(img), 0, random.uniform(10, 40))
    img = cv2.convertScaleAbs(img, alpha=random.uniform(0.7, 0.9), beta=random.randint(-10,10))
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img
def load_nonhuman_paths(source_dir):
    paths = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(root, f))
    return paths
def get_patient_id(filename):
    base = os.path.basename(filename)
    if '_' in base:
        return base.split('_')[0]
    else:
        return os.path.splitext(base)[0]
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    real_images = []
    for root, _, files in os.walk(REAL_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                real_images.append(os.path.join(root, f))
    if len(real_images) == 0:
        raise RuntimeError(f"在 {REAL_DIR} 中未找到真实胸片！请放入PNG/JPG文件。")
    patient_groups = defaultdict(list)
    for img_path in real_images:
        pid = get_patient_id(img_path)
        patient_groups[pid].append(img_path)
    patient_ids = list(patient_groups.keys())
    print(f"找到 {len(patient_ids)} 位患者，共 {len(real_images)} 张真实胸片。")
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * TRAIN_RATIO)
    train_pids = set(patient_ids[:split_idx])
    val_pids = set(patient_ids[split_idx:])
    print(f"训练患者: {len(train_pids)} 位，验证患者: {len(val_pids)} 位。")
    nonhuman_paths = []
    if os.path.exists(NONHUMAN_SOURCE_DIR):
        nonhuman_paths = load_nonhuman_paths(NONHUMAN_SOURCE_DIR)
        print(f"找到 {len(nonhuman_paths)} 张非人体自然图像。")
    else:
        print("未找到非人体图像源目录，将使用随机纹理生成非人体数据。")
    for split in ['train', 'val']:
        for cls in ['0_real', '1_fake', '2_nonhuman']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    stats = defaultdict(int)
    for split, pids in [('train', train_pids), ('val', val_pids)]:
        print(f"\n处理 {split} 集...")
        for pid in tqdm(pids, desc=f"{split} 患者"):
            for img_path in patient_groups[pid]:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (512, 512))
                base_name = f"{pid}_{os.path.basename(img_path).split('.')[0]}.png"
                real_out = os.path.join(OUTPUT_DIR, split, '0_real', base_name)
                cv2.imwrite(real_out, img)
                stats[f'{split}_real'] += 1
                for i in range(SCREEN_CAPTURE_COUNT):
                    fake_img = screen_capture_pipeline(img.copy())
                    fake_name = f"{pid}_screen_{i}_{os.path.basename(img_path).split('.')[0]}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, split, '1_fake', fake_name), fake_img)
                    stats[f'{split}_fake_screen'] += 1
                for i in range(BODY_PHANTOM_COUNT):
                    phantom_img = simulate_phantom(img.copy())
                    phantom_name = f"{pid}_phantom_{i}_{os.path.basename(img_path).split('.')[0]}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, split, '1_fake', phantom_name), phantom_img)
                    stats[f'{split}_fake_phantom'] += 1
        nonhuman_needed = stats[f'{split}_real']
        if nonhuman_paths:
            selected = random.sample(nonhuman_paths, min(nonhuman_needed, len(nonhuman_paths)))
            for idx, nh_path in enumerate(selected):
                nh_img = cv2.imread(nh_path)
                if nh_img is None:
                    continue
                nh_img = cv2.resize(nh_img, (512, 512))
                nh_name = f"nonhuman_{idx:06d}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, split, '2_nonhuman', nh_name), nh_img)
                stats[f'{split}_nonhuman'] += 1
        else:
            for i in range(nonhuman_needed):
                texture = np.random.randint(0, 256, (512,512,3), dtype=np.uint8)
                texture = cv2.GaussianBlur(texture, (5,5), random.uniform(1,4))
                if random.random() > 0.5:
                    texture = cv2.applyColorMap(texture, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(OUTPUT_DIR, split, '2_nonhuman', f"random_{i:06d}.png"), texture)
                stats[f'{split}_nonhuman'] += 1
    print("\n====== 数据集生成统计 ======")
    for key, val in sorted(stats.items()):
        print(f"{key}: {val} 张")
    print("============================")
if __name__ == '__main__':
    main()
