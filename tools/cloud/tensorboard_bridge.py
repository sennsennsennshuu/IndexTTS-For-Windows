"""只读训练记录并生成 TensorBoard events，不加载模型、不占用训练 GPU。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


SOURCES = {
    "pilot": Path("training/runs/overfit/metrics.jsonl"),
    "formal": Path("training/runs/minnan-lora/metrics.jsonl"),
}
STATE = Path("logs/tensorboard_bridge_state.json")
LOGDIR = Path("training/tensorboard")
TAGS = {
    "loss_per_token": "train/loss_per_token",
    "grad_norm": "train/grad_norm",
    "lr_lora": "train/lr_lora",
    "lr_row": "train/lr_language",
    "val_loss": "validation/loss",
    "gpu_allocated_mb": "gpu/allocated_mib",
    "gpu_reserved_mb": "gpu/reserved_mib",
    "tokens": "train/tokens",
    "optloss_mean": "train/optimization_loss",
    "samples": "train/samples_per_step",
}


def write_json(path: Path, value: dict) -> None:
    """先写临时文件再提交，避免查看进程读到半截状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_complete_records(path: Path, offset: int):
    """只消费完整 JSONL 行；训练器正在追加的半行留待下轮读取。"""
    with path.open("rb") as stream:
        stream.seek(offset)
        while True:
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                return
            if line.strip():
                yield stream.tell(), json.loads(line.decode("utf-8"))


def scalar_values(record: dict) -> dict[str, float]:
    """拒绝 NaN／Inf，防止面板把异常训练指标显示成正常曲线。"""
    result = {}
    for key, tag in TAGS.items():
        value = record.get(key)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"训练指标不是有限数：{key}")
        result[tag] = number
    if record.get("event") == "validation_initial" and "validation/loss" in result:
        result["validation/initial_loss"] = result["validation/loss"]
    return result


class Bridge:
    def __init__(self):
        # 使用 TensorBoard 自带的轻量事件写入器，避免导入 torch 或创建 CUDA 上下文。
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.Event, self.Summary, self.EventFileWriter = Event, Summary, EventFileWriter
        self.state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        self.writers = {}

    def add(self, run: str, step: int, values: dict[str, float]) -> None:
        if not values:
            return
        if run not in self.writers:
            self.writers[run] = self.EventFileWriter(str(LOGDIR / run))
        summary = self.Summary(value=[self.Summary.Value(tag=tag, simple_value=value)
                                      for tag, value in values.items()])
        # 原 JSONL 未记录绝对训练时间；这里只记录导入时间，横轴 step 保留真实优化步数。
        self.writers[run].add_event(self.Event(wall_time=time.time(), step=step, summary=summary))

    def poll(self) -> dict:
        for run, source in SOURCES.items():
            if not source.exists():
                continue
            identity = [source.stat().st_dev, source.stat().st_ino]
            cursor = self.state.get(run, {})
            offset = int(cursor.get("offset", 0))
            # 日志被重建或 events 被移走时显式报错，不能静默沿用错误偏移。
            if offset and (cursor.get("identity") != identity or source.stat().st_size < offset
                           or not any((LOGDIR / run).glob("events.out.tfevents.*"))):
                raise RuntimeError(f"{run} 日志身份发生变化，需要保留现场后重建独立 run")
            count = int(cursor.get("records", 0))
            validation_steps = set(cursor.get("validation_steps", []))
            if "validation_steps" not in cursor and offset:
                # 从早期桥接状态升级时仅扫描已消费的前缀，补齐去重信息。
                for end, record in read_complete_records(source, 0):
                    if end > offset:
                        break
                    if record.get("val_loss") is not None:
                        validation_steps.add(int(record["step"]))
            for end, record in read_complete_records(source, offset):
                step, values = int(record["step"]), scalar_values(record)
                if record.get("val_loss") is not None:
                    if step in validation_steps:
                        values.pop("validation/loss", None)
                    validation_steps.add(step)
                self.add(run, step, values)
                offset, count = end, count + 1
            # 早期试训尚未把中途验证追加到 JSONL；只从仍然存在的真实 checkpoint 补齐，
            # 不插值、不推测已经被清理的检查点，也不重复写入 JSONL 已有的同一步。
            for checkpoint in sorted((source.parent / "checkpoints").glob("step_*/checkpoint.json")):
                metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
                step = int(metadata["step"])
                if metadata.get("val_loss") is not None and step not in validation_steps:
                    self.add(run, step, scalar_values({"val_loss": metadata["val_loss"]}))
                    validation_steps.add(step)
            self.state[run] = {"offset": offset, "identity": identity, "records": count,
                               "validation_steps": sorted(validation_steps)}

        source = Path("logs/extraction_progress.json")
        if source.exists():
            try:
                progress = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # 提取器原地更新 JSON，偶尔会碰到写入窗口；下一轮重新读，不改训练器。
                progress = None
            if progress is not None:
                signature = [progress["done"], progress["total"], progress.get("complete", False)]
                if signature != self.state.get("features"):
                    total, done = int(progress["total"]), int(progress["done"])
                    values = {"progress/done": done, "progress/total": total,
                              "progress/percent": 100 * done / max(1, total)}
                    for key in ("new", "reused", "elapsed", "gpu_peak_mib"):
                        if key in progress:
                            values["progress/" + key] = float(progress[key])
                    self.add("features", done, values)
                    self.state["features"] = signature
        # 先把 events 刷入文件，再提交偏移；正常重启不会重复导入历史记录。
        for writer in self.writers.values():
            writer.flush()
        self.state.update({"pid": os.getpid(), "updated_unix": time.time()})
        write_json(STATE, self.state)
        return self.state

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    Path("logs").mkdir(exist_ok=True)
    # 云端使用进程锁，防止两个桥接器同时写入并重复消费相同记录。
    import fcntl
    with Path("logs/tensorboard_bridge.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        bridge = Bridge()
        try:
            while True:
                state = bridge.poll()
                print(json.dumps(state, ensure_ascii=False), flush=True)
                if args.once:
                    return
                time.sleep(max(1, args.interval))
        finally:
            bridge.close()


if __name__ == "__main__":
    main()
