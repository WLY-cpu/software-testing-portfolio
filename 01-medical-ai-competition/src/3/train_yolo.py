import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ultralytics import YOLO
from train_logger import TrainLogger, log_training_step, log_validation_step
def make_loss_logger(logger):
    """构造训练日志回调（ultralytics 通过 add_callback 注册）。"""
    state = {"epoch": 0, "step": 0}
    def on_train_epoch_start(trainer):
        state["epoch"] = getattr(trainer, "epoch", 0)
    def on_train_batch_end(trainer):
        state["step"] += 1
        if state["step"] % 50 == 0:
            loss = getattr(trainer, "loss", None)
            if loss is None:
                loss = getattr(trainer, "tloss", 0.0)
            try:
                loss = float(loss)
            except Exception:
                loss = 0.0
            lr = 0.001
            try:
                lr = trainer.optimizer.param_groups[0]["lr"]
            except Exception:
                pass
            log_training_step(logger, state["epoch"], state["step"], loss, lr)
    def on_val_end(trainer):
        val_loss = 0.0
        try:
            val_loss = float(trainer.metrics.get("val/loss", 0.0))
        except Exception:
            try:
                val_loss = float(getattr(trainer.metrics, "val/loss", 0.0))
            except Exception:
                val_loss = 0.0
        lr = 0.001
        try:
            lr = trainer.optimizer.param_groups[0]["lr"]
        except Exception:
            pass
        checkpoint_path = None
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            checkpoint_path = os.path.join(str(save_dir), "weights", "last.pt")
        log_validation_step(
            logger, state["epoch"], state["step"], val_loss, lr,
            checkpoint_path=checkpoint_path,
        )
    return on_train_epoch_start, on_train_batch_end, on_val_end
def find_weights(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="赛道八 17类病灶检测 YOLO 训练")
    parser.add_argument("--data", default=os.path.join(script_dir, "dataset.yaml"),
                        help="数据集 YAML 路径")
    parser.add_argument("--weights", default="weights/yolov8m.pt",
                        help="预训练权重路径；不存在时依次尝试 yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--name", default="cxr_lesion", help="训练输出目录名")
    parser.add_argument("--plots", action="store_true",
                        help="生成训练图表（需中文字体；无外网容器建议关闭）")
    args = parser.parse_args()
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[train] 计算设备: {device}")
    weights_path = find_weights([
        args.weights,
        "yolov8m.pt",
        os.path.join(script_dir, "yolov8m.pt"),
    ])
    if weights_path:
        print(f"[train] 使用预训练权重: {weights_path}")
        model = YOLO(weights_path)
    else:
        print("[WARN] 未找到 yolov8m.pt，无外网环境无法自动下载，"
              "将使用 yolov8m.yaml 从零训练。")
        print("       建议先把 yolov8m.pt（约50MB）传到 weights/yolov8m.pt 再训练。")
        model = YOLO("yolov8m.yaml")
    if os.path.isdir("/2026aicompetition/workspace"):
        log_dir = "/2026aicompetition/workspace/logs"
    else:
        log_dir = os.path.join(os.getcwd(), "logs")
    logger = TrainLogger(os.path.join(log_dir, "yolo_train.jsonl"))
    cb_epoch_start, cb_batch_end, cb_val_end = make_loss_logger(logger)
    model.add_callback("on_train_epoch_start", cb_epoch_start)
    model.add_callback("on_train_batch_end", cb_batch_end)
    model.add_callback("on_val_end", cb_val_end)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=1e-3,
        optimizer="AdamW",
        cos_lr=True,
        device=device,
        name=args.name,
        exist_ok=True,
        amp=False,
        plots=args.plots,
    )
    logger.close()
    print(f"训练完成，日志已保存: logs/yolo_train.jsonl")
if __name__ == "__main__":
    main()
