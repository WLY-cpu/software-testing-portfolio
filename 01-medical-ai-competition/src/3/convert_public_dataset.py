import os
import sys
import argparse
import random
import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
CLASS_NAME_TO_ID = {
    "纤维条索影": 0, "片状模糊影": 1, "空气支气管征": 2,
    "间质性改变": 3, "肺水肿": 4, "肺不张": 5,
    "肺气肿": 6, "支气管扩张": 7, "肺部肿块": 8,
    "肺结节": 9, "空洞": 10, "肺内钙化": 11,
    "胸膜增厚": 12, "胸腔积液": 13, "气胸": 14,
    "纵隔增宽": 15, "心脏增大": 16,
}
def read_annotation_csv(csv_path):
    """读取标注 CSV，兼容坐标字段未加引号（内部含逗号）的导出文件。"""
    raw = None
    used_enc = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                raw = list(csv.reader(f))
            used_enc = enc
            break
        except (UnicodeDecodeError, csv.Error):
            continue
    if raw is None:
        raise ValueError(f"CSV 读取失败: {csv_path}（尝试过 utf-8 / gbk）")
    if not raw:
        raise ValueError(f"CSV 为空: {csv_path}")
    header = [str(h).strip() for h in raw[0]]
    data_rows = []
    for line in raw[1:]:
        if not line or all(not str(c).strip() for c in line):
            continue
        if len(line) == len(header):
            data_rows.append(line)
        elif len(line) > len(header):
            data_rows.append(line[:len(header) - 1] +
                             [",".join(str(c) for c in line[len(header) - 1:])])
        else:
            continue
    df = pd.DataFrame(data_rows, columns=header)
    print(f"      CSV 解析: encoding={used_enc}, {len(df)} 行标注记录")
    return df
def load_dicom_as_rgb(dicom_path: str) -> np.ndarray:
    """读取 DICOM 文件，返回 RGB 图像 (H, W, 3)。"""
    try:
        import pydicom
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "WindowWidth") and hasattr(ds, "WindowCenter"):
            ww = float(ds.WindowWidth)
            wc = float(ds.WindowCenter)
            low = wc - ww / 2
            high = wc + ww / 2
            arr = np.clip(arr, low, high)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return arr
    except Exception as e:
        print(f"  [WARN] 无法读取 DICOM: {dicom_path}  {e}")
        return None
def yolo_normalize(coord_str: str, img_w: int, img_h: int):
    """解析 XLSX 中的 Coordinates 字段，返回 YOLO 格式 (cls xc yc w h)。"""
    coord_str = coord_str.strip().replace("[[", "").replace("]]", "").replace("[", "").replace("]", "")
    parts = [float(p) for p in coord_str.replace(",", " ").split()]
    if len(parts) >= 4:
        x1, y1, x2, y2 = parts[0], parts[1], parts[2], parts[3]
    elif len(parts) == 2:
        return None
    else:
        return None
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0 or bw > img_w * 0.95 or bh > img_h * 0.95:
        return None
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    norm_w = bw / img_w
    norm_h = bh / img_h
    return (x_center, y_center, norm_w, norm_h)
def main():
    parser = argparse.ArgumentParser(description="赛道八 公共数据集 → YOLO 格式转换")
    parser.add_argument("--csv", type=str, default="",
                        help="标注 CSV 文件路径（优先于 --xlsx）")
    parser.add_argument("--xlsx", type=str, default="/data/public/track8/annotation/annotations_track8.xlsx", help="标注 XLSX 文件路径")
    parser.add_argument("--dicom_dir", type=str,
                        default="/data/public/track8/original",
                        help="DICOM 原始图像根目录")
    parser.add_argument("--output_dir", type=str, default="3/data",
                        help="输出目录 (YOLO 格式)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    xlsx_path = args.xlsx
    dicom_root = Path(args.dicom_dir)
    output_root = Path(args.output_dir)
    df = None
    src_desc = ""
    if args.csv and os.path.exists(args.csv):
        try:
            df = read_annotation_csv(args.csv)
            src_desc = args.csv
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    else:
        if not os.path.exists(xlsx_path):
            local_path = os.path.join(output_root, "annotations_track8.xlsx")
            if os.path.exists(local_path):
                xlsx_path = local_path
            else:
                print(f"[ERROR] 标注文件不存在: {xlsx_path}")
                print(f"       请通过 --csv 或 --xlsx 指定标注文件路径")
                sys.exit(1)
        try:
            df = pd.read_excel(xlsx_path)
            src_desc = xlsx_path
        except ImportError:
            print("[ERROR] 缺少 openpyxl，无法读取 XLSX。请先 pip install openpyxl，或改用 --csv。")
            sys.exit(1)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[1/5] 读取标注文件: {src_desc}")
    print(f"      共 {len(df)} 行标注记录")
    print(f"      字段: {list(df.columns)[:10]}...")
    required_cols = ["AccessionNumber", "RoiName", "ANNOTATION->Coordinates"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] 缺少必要列: {missing}")
        print(f"       可用列: {list(df.columns)}")
        sys.exit(1)
    print(f"[2/5] 按 AccessionNumber 分组...")
    study_groups = defaultdict(list)
    for _, row in df.iterrows():
        acc = str(row["AccessionNumber"]).strip()
        study_groups[acc].append(row)
    print(f"      共 {len(study_groups)} 个检查")
    print(f"[3/5] 划分训练/验证集 (val_ratio={args.val_ratio})...")
    all_accs = list(study_groups.keys())
    random.shuffle(all_accs)
    val_count = max(1, int(len(all_accs) * args.val_ratio))
    val_accs = set(all_accs[:val_count])
    train_accs = set(all_accs[val_count:])
    print(f"      训练: {len(train_accs)} 个检查, 验证: {len(val_accs)} 个检查")
    for split_name, acc_set in [("train", train_accs), ("val", val_accs)]:
        img_out_dir = output_root / "images" / split_name
        label_out_dir = output_root / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[4/5] 生成 {split_name} 集...")
        converted = 0
        skipped_no_dicom = 0
        skipped_no_label = 0
        for acc in tqdm(acc_set, desc=f"  {split_name}"):
            rows = study_groups[acc]
            study_dir = dicom_root / acc
            dcm_files = []
            if study_dir.exists():
                dcm_files = list(study_dir.rglob("*.dcm")) + list(study_dir.rglob("*.IMA"))
            if not dcm_files:
                png_candidates = list(img_out_dir.glob(f"{acc}*.png"))
                if png_candidates:
                    dcm_files = [png_candidates[0]]
            if not dcm_files:
                skipped_no_dicom += 1
                continue
            dcm_path = str(dcm_files[0])
            img = None
            img_h, img_w = 0, 0
            cached_png = img_out_dir / f"{acc}.png"
            if cached_png.exists():
                img = cv2.imread(str(cached_png))
                if img is not None:
                    img_h, img_w = img.shape[:2]
            if img is None:
                if dcm_path.lower().endswith(".png"):
                    img = cv2.imread(dcm_path)
                else:
                    img = load_dicom_as_rgb(dcm_path)
                if img is None:
                    skipped_no_dicom += 1
                    continue
                img_h, img_w = img.shape[:2]
                cv2.imwrite(str(cached_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            label_lines = []
            for row in rows:
                roi_name = str(row["RoiName"]).strip()
                cls_id = CLASS_NAME_TO_ID.get(roi_name, -1)
                if cls_id < 0:
                    continue
                coord_str = str(row["ANNOTATION->Coordinates"])
                yolo_vals = yolo_normalize(coord_str, img_w, img_h)
                if yolo_vals is None:
                    continue
                xc, yc, nw, nh = yolo_vals
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            if not label_lines:
                skipped_no_label += 1
                continue
            label_path = label_out_dir / f"{acc}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))
            converted += 1
        print(f"      {split_name}: 成功 {converted}, "
              f"无DICOM {skipped_no_dicom}, 无有效标注 {skipped_no_label}")
    print(f"[5/5] 更新 dataset.yaml...")
    yaml_content = f"""path: {str(output_root.resolve())}
train: images/train
val: images/val
nc: 17
names:
"""
    for i in range(17):
        name = [k for k, v in CLASS_NAME_TO_ID.items() if v == i][0]
        yaml_content += f"  {i}: '{name}'\n"
    dataset_yaml = Path(__file__).resolve().parent / "dataset.yaml"
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"      dataset.yaml 已更新: {dataset_yaml}")
    print(f"\n{'='*50}")
    print(f"转换完成！数据集统计：")
    for split in ["train", "val"]:
        img_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        n_img = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
        n_label = len(list(label_dir.glob("*.txt")))
        print(f"  {split}: {n_img} 张图片, {n_label} 个标注文件")
    print(f"{'='*50}")
if __name__ == "__main__":
    main()
