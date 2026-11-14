import json
import os
import sys
import threading
import time
import urllib.request
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch
from path_utils import setup_module_paths, resource_path
setup_module_paths()
from competition_output import generate_all_outputs, generate_duplicate_pairs_jsonl
from pipeline import CompetitionPipeline
from train_logger import TrainLogger, log_inference_step
ANSWER_BASE = Path("/2026aicompetition/workspace/answer")
DEFAULT_CALLBACK = "http://10.1.13.241:80/api/competition/inference/callback/"
TEAM_ID = "2054451790241292288"
_pipeline = None
_pipeline_lock = threading.Lock()
def load_dicom_as_bgr(dicom_path):
    try:
        import pydicom
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr
    except Exception:
        img = cv2.imread(dicom_path)
        return img
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[server] 初始化推理管线 (device={device})...")
                _pipeline = CompetitionPipeline(device=device)
                print("[server] 推理管线就绪。")
    return _pipeline
def callback_platform(callback_url, request_id, evaluation_id, pred_path, max_retries=3):
    payload = {
        "request_id": request_id,
        "evaluationId": evaluation_id,
        "predPath": pred_path,
    }
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                callback_url, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30):
                pass
            print(f"[callback] 成功回调平台 (attempt={attempt+1})")
            return True
        except Exception as e:
            print(f"[callback] 回调失败 (attempt={attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False
def run_inference(request_id, team_id, track_code, evaluation_id, dataset_path, callback_url):
    print(f"[inference] 开始推理 request_id={request_id}, evaluation_id={evaluation_id}")
    print(f"[inference] 数据集路径: {dataset_path}")
    start_time = time.time()
    pipeline = get_pipeline()
    try:
        pipeline.norm_gate.dup_detector.reset_db()
    except Exception:
        pass
    testset_path = Path(dataset_path)
    if not testset_path.exists():
        print(f"[inference] 错误: 数据集路径不存在: {dataset_path}")
        return
    if os.path.isdir("/2026aicompetition/workspace"):
        log_dir = "/2026aicompetition/workspace/logs"
    else:
        log_dir = os.path.join(os.getcwd(), "logs")
    logger = TrainLogger(os.path.join(log_dir, "inference.jsonl"))
    log_inference_step(logger, data_source=str(dataset_path))
    image_files = []
    supported_ext = {".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for f in testset_path.rglob("*"):
        if f.suffix.lower() in supported_ext and f.is_file():
            rel = f.relative_to(testset_path)
            parts = list(rel.parts)
            if parts and parts[0].lower() == "original":
                parts = parts[1:]
            if len(parts) >= 3:
                study_uid = parts[-3]
                series_uid = parts[-2]
            elif len(parts) == 2:
                study_uid = parts[-2]
                series_uid = Path(parts[-1]).stem
            else:
                study_uid = Path(parts[-1]).stem
                series_uid = study_uid
            image_files.append({
                "path": str(f), "study_uid": study_uid, "series_uid": series_uid,
            })
    print(f"[inference] 共发现 {len(image_files)} 个图像文件")
    if not image_files:
        print("[inference] 未找到可处理的图像文件")
        pred_path = f"/2026aicompetition/workspace/answer/{evaluation_id}"
        callback_platform(callback_url, request_id, evaluation_id, pred_path)
        return
    from collections import defaultdict
    accession_groups = defaultdict(list)
    for img_info in image_files:
        accession_groups[img_info["study_uid"]].append(img_info)
    dup_pairs = []
    for acc_idx, (accession_number, imgs) in enumerate(accession_groups.items()):
        img_info = imgs[0]
        series_uid = img_info["series_uid"]
        print(f"[inference] [{acc_idx+1}/{len(accession_groups)}] "
              f"处理 {accession_number}/{series_uid} ({len(imgs)} 张)")
        study_start = time.time()
        img_bgr = load_dicom_as_bgr(img_info["path"])
        if img_bgr is None:
            print(f"[inference] 跳过文件: {img_info['path']}")
            continue
        h, w = img_bgr.shape[:2]
        result = pipeline.process(img_info["path"], accession_number,
                                  exam_timestamp=None)
        if "error" in result:
            print(f"[inference] 管线错误: {result['error']}")
            continue
        auth = result.get("authenticity", {})
        norm = result.get("normality", {})
        detections = result.get("detections", [])
        splice_prob = 0.0
        if norm and "details" in norm:
            splice_prob = norm["details"].get("cnn_splice_prob", 0.0)
            dup_info = norm["details"].get("duplicate") or {}
            if dup_info.get("is_duplicate"):
                matched_acc = dup_info.get("matched_uid")
                pair_prob = float(dup_info.get("pair_prob", 0.99))
                if matched_acc and matched_acc != accession_number:
                    dup_pairs.append((accession_number, matched_acc, pair_prob))
        processing_ms = int((time.time() - study_start) * 1000)
        pixel_spacing = [0.194, 0.194]
        try:
            if ".dcm" in img_info["path"].lower():
                import pydicom
                ds = pydicom.dcmread(img_info["path"], stop_before_pixels=True)
                if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
                    row_spacing = float(ds.PixelSpacing[0])
                    col_spacing = float(ds.PixelSpacing[1]) \
                        if len(ds.PixelSpacing) > 1 else row_spacing
                    pixel_spacing = [col_spacing, row_spacing]
        except Exception:
            pass
        generate_all_outputs(
            team_id=team_id,
            evaluation_id=evaluation_id,
            answer_base=str(ANSWER_BASE),
            accession_number=accession_number,
            image_size={"width": w, "height": h},
            pixel_spacing=pixel_spacing,
            auth_result=auth,
            norm_result={"splice_prob": splice_prob, "normality": norm},
            detections=detections,
            processing_time_ms=processing_ms,
        )
        print(f"[inference] 耗时: {processing_ms}ms, 病灶: {len(detections)}个")
    if dup_pairs:
        answer_dir = ANSWER_BASE / str(evaluation_id)
        generate_duplicate_pairs_jsonl(dup_pairs, str(answer_dir))
    logger.close()
    total_time = time.time() - start_time
    print(f"[inference] 全部完成，总耗时: {total_time:.1f}s")
    pred_path = f"/2026aicompetition/workspace/answer/{evaluation_id}"
    callback_platform(callback_url, request_id, evaluation_id, pred_path)
class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        body = b'{"status": "active"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        if self.path != "/call":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw)
            request_id = payload["request_id"]
            team_id = payload.get("team_id", TEAM_ID)
            track_code = payload.get("track_code", "")
            input_data = payload["input"]
            evaluation_id = input_data.get("evaluation_id") or input_data["evaluationId"]
            dataset_path = input_data["dataset_path"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            print(f"[server] 请求解析失败: {e}")
            self.send_error(400, "invalid request body")
            return
        callback_url = os.environ.get("CALLBACK_URL", DEFAULT_CALLBACK)
        body = b'{"status": "received"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        thread = threading.Thread(
            target=run_inference,
            args=(request_id, team_id, track_code, evaluation_id, dataset_path, callback_url),
            daemon=True,
        )
        thread.start()
        print(f"[server] 推理已加入后台: request_id={request_id}")
    def log_message(self, format, *args):
        pass
def serve(host: str = "0.0.0.0", port: int = 8000,
          callback: Optional[str] = None, team_id: str = TEAM_ID) -> None:
    """启动推理服务（供 zong.py --server 与独立运行共用）。"""
    if callback:
        os.environ.setdefault("CALLBACK_URL", callback)
    print("[server] 正在初始化推理管线...")
    get_pipeline()
    server = ThreadingHTTPServer((host, port), InferenceHandler)
    print(f"[server] 推理服务已启动: http://{host}:{port}")
    print(f"[server] 健康检查: GET /health")
    print(f"[server] 推理接口: POST /call")
    print(f"[server] 队伍ID: {team_id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 服务关闭。")
        server.server_close()
def parse_args():
    parser = argparse.ArgumentParser(description="赛道八 推理服务")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--callback", type=str, default=DEFAULT_CALLBACK, help="回调地址")
    parser.add_argument("--team-id", type=str, default=TEAM_ID, help="队伍ID")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    serve(host=args.host, port=args.port,
          callback=args.callback, team_id=args.team_id)
