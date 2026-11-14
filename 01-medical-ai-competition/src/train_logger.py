import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
class TrainLogger:
    """JSONL 格式训练日志记录器。"""
    def __init__(self, log_path: str = "logs/train.jsonl"):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
    def log(
        self,
        epoch: Optional[int],
        step: Optional[int],
        phase: str,
        mode: str = "training",
        loss: Optional[float] = None,
        lr: Optional[float] = None,
        data_source: str = "official/train_v1",
        checkpoint: Optional[str] = None,
    ) -> None:
        """写入一条训练日志。
        参数：
          epoch: 当前 epoch（推理阶段为 null）
          step:  当前 step（推理阶段为 null）
          phase: train / val / test
          mode:  training / inference
          loss:  损失值
          lr:    学习率
          data_source: 数据来源
          checkpoint:  保存的 checkpoint 路径（如无则为 null）
        """
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") +
                         f"{datetime.now().microsecond // 1000:03d}Z",
            "epoch": epoch,
            "step": step,
            "phase": phase,
            "mode": mode,
            "loss": loss,
            "lr": lr,
            "data_source": data_source,
            "checkpoint": checkpoint,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
    def __del__(self):
        self.close()
def log_training_step(
    logger: TrainLogger,
    epoch: int,
    step: int,
    loss: float,
    lr: float,
    phase: str = "train",
):
    """记录单步训练日志。"""
    logger.log(
        epoch=epoch,
        step=step,
        phase=phase,
        mode="training",
        loss=round(loss, 4),
        lr=round(lr, 6),
        data_source="official/train_v1",
        checkpoint=None,
    )
def log_validation_step(
    logger: TrainLogger,
    epoch: int,
    step: int,
    loss: float,
    lr: float,
    checkpoint_path: Optional[str] = None,
):
    """记录验证步骤日志。"""
    logger.log(
        epoch=epoch,
        step=step,
        phase="val",
        mode="training",
        loss=round(loss, 4),
        lr=round(lr, 6),
        data_source="official/val_v1",
        checkpoint=checkpoint_path,
    )
def log_inference_step(
    logger: TrainLogger,
    data_source: str = "official/test_v1",
):
    """记录推理阶段日志。"""
    logger.log(
        epoch=None,
        step=None,
        phase="test",
        mode="inference",
        loss=None,
        lr=None,
        data_source=data_source,
        checkpoint=None,
    )
