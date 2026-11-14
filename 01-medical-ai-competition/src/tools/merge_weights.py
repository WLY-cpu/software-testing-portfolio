"""容器工具：把个人空间下载的权重 txt 分片还原为二进制权重文件。
用法：
  python merge_weights.py --dir . --out weights
"""
import argparse
import base64
import json
import os
import re
def main():
    ap = argparse.ArgumentParser(description="权重 txt 分片 → 二进制还原")
    ap.add_argument("--dir", default=".", help="分片 txt 所在目录（含 weights_manifest.json.txt）")
    ap.add_argument("--out", default="weights", help="输出目录")
    args = ap.parse_args()
    manifest_path = os.path.join(args.dir, "weights_manifest.json.txt")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] 找不到清单: {manifest_path}")
        return 1
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    for item in manifest["weights"]:
        parts = []
        for part_name in item["files"]:
            part_path = os.path.join(args.dir, part_name)
            if not os.path.exists(part_path):
                print(f"[ERROR] 缺少分片: {part_path}")
                return 1
            with open(part_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = re.findall(r'_ = "([A-Za-z0-9+/=]*)"', text)
            parts.append("".join(chunks))
        b64 = "".join(parts)
        data = base64.b64decode(b64)
        out_path = os.path.join(args.out, item["file"])
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  + {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")
    print("权重还原完成。")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
