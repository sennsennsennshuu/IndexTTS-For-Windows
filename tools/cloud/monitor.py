"""只读监控本项目训练、GPU 和数据盘，将真实采样写入日志及 TensorBoard。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from tensorboard_bridge import read_complete_records, write_json


def process_info(pid: int, script: str | None = None) -> dict:
    """核对实际 /proc 身份；旧 PID 或单独存在的状态文件不能证明作业存活。"""
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        arguments = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
        cwd_matches = (proc / "cwd").resolve() == Path.cwd()
        script_matches = script is None or script in arguments
        return {"pid": pid, "alive": fields[0] not in ("Z", "X"), "state": fields[0],
                "identity_matches": cwd_matches and script_matches, "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "alive": False, "identity_matches": False}


def read_progress(path: Path) -> dict | None:
    """提取进度由原进程直接更新；短暂写入窗口不当作训练失败。"""
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            time.sleep(0.05)
    raise ValueError(f"无法读取完整进度：{path}")


def collect_metrics(state: dict) -> dict:
    """按字节偏移增量读取真实正式训练日志，保留最近 100 步和全部验证点。"""
    path = Path("training/runs/minnan-lora/metrics.jsonl")
    if not path.exists():
        return {"available": False}
    identity = [path.stat().st_dev, path.stat().st_ino]
    offset = int(state.get("offset", 0))
    if offset and (state.get("identity") != identity or path.stat().st_size < offset):
        raise RuntimeError("正式训练日志被替换或截短，保留监控现场并人工核对")
    recent = state.get("recent", [])
    validations = state.get("validations", [])
    invalid = state.get("invalid", [])
    for end, row in read_complete_records(path, offset):
        invalid_fields = []
        for key in ("loss_per_token", "grad_norm", "lr_lora", "lr_row", "val_loss"):
            if row.get(key) is not None and not math.isfinite(float(row[key])):
                invalid.append({"step": row.get("step"), "field": key})
                invalid_fields.append(key)
        if invalid_fields:
            # 非有限记录保留在训练原日志，并将字段位置记入告警；不让 NaN 污染均值和 JSON。
            offset = end
            continue
        if "loss_per_token" in row:
            recent.append({key: row[key] for key in ("step", "loss_per_token", "grad_norm",
                                                    "lr_lora", "lr_row", "elapsed", "samples") if key in row})
            recent = recent[-100:]
        if row.get("val_loss") is not None:
            validations.append({"step": row["step"], "loss": row["val_loss"], "event": row.get("event")})
        offset = end
    state.update({"identity": identity, "offset": offset, "recent": recent,
                  "validations": validations, "invalid": invalid})
    finite_losses = [float(row["loss_per_token"]) for row in recent if math.isfinite(float(row["loss_per_token"]))]
    baseline = next((float(row["loss"]) for row in validations if row.get("event") == "validation_initial"), None)
    return {"available": True, "last": recent[-1] if recent else None,
            "rolling_loss_100": sum(finite_losses) / len(finite_losses) if finite_losses else None,
            "rolling_count": len(finite_losses), "baseline_validation": baseline,
            "last_validation": validations[-1] if validations else None,
            "recent_validation": [row for row in validations if row.get("event") != "validation_initial"][-3:],
            "best_validation": min((float(row["loss"]) for row in validations), default=None),
            "invalid": invalid}


def assess(snapshot: dict) -> list[dict]:
    """监控只报告证据，不因告警擅自停止训练或修改学习率。"""
    alerts = []
    def add(code: str, message: str, severity: str = "warning"):
        alerts.append({"code": code, "severity": severity, "message": message})
    terminal = snapshot["stage"] in ("complete", "failed")
    process = snapshot["pipeline_process"]
    if snapshot["stage"] == "failed":
        add("pipeline_failed", "流水线已记录失败，需要检查阶段日志", "critical")
    elif not terminal and (not process["alive"] or not process["identity_matches"]):
        add("pipeline_process_missing", "流水线实际进程缺失或身份不匹配", "critical")
    if not terminal and snapshot["progress_age_seconds"] > 600:
        add("progress_stalled", "超过十分钟没有进度变化，需检查实际进程和阶段日志")
    free = snapshot["disk"]["free_gib"]
    if free < 10:
        add("disk_low", f"tmp 数据盘仅剩 {free:.2f} GiB", "critical" if free < 3 else "warning")
    gpu = snapshot.get("gpu", {})
    if gpu.get("temperature_c", 0) >= 85:
        add("gpu_hot", "GPU 温度达到监控阈值 85 摄氏度")
    if snapshot.get("gpu_error"):
        add("gpu_telemetry_unavailable", "暂时无法读取 GPU 遥测，不据此判定训练失败")
    metrics = snapshot["training"]
    if metrics.get("invalid"):
        add("nonfinite_metric", "正式训练记录出现 NaN 或 Inf", "critical")
    values = [float(row["loss"]) for row in metrics.get("recent_validation", [])]
    baseline = metrics.get("baseline_validation")
    if len(values) == 3 and baseline is not None and all(value > baseline for value in values):
        add("validation_above_baseline", "最近三次验证均高于基础模型验证 loss，需要检查泛化表现")
    if len(values) == 3 and values[0] < values[1] < values[2]:
        add("validation_rising", "最近三次验证 loss 连续上升，需要结合训练曲线检查过拟合")
    return alerts


def collect(state: dict) -> dict:
    now = time.time()
    pipeline = read_progress(Path("logs/pipeline_status.json"))
    if pipeline is None:
        raise FileNotFoundError("缺少流水线状态")
    stage = pipeline["stage"]
    process = process_info(int(pipeline["pid"]), "tools/cloud/pipeline.py")
    children = []
    if process["alive"] and process["identity_matches"]:
        try:
            child_ids = (Path("/proc") / str(pipeline["pid"]) / "task" / str(pipeline["pid"]) / "children").read_text().split()
            children = [process_info(int(pid)) for pid in child_ids]
        except FileNotFoundError:
            pass
    extraction = read_progress(Path("logs/extraction_progress.json"))
    formal = read_progress(Path("training/runs/minnan-lora/progress.json"))
    progress = formal if stage == "formal_training" else extraction if stage == "full_extraction" else None
    value_key = "step" if stage == "formal_training" else "done"
    value = int((progress or {}).get(value_key, 0))
    token = [stage, value]
    previous = state.get("progress")
    if previous is None or previous["token"] != token:
        state["last_advance_unix"] = now
    speed = None
    if previous and previous["token"][0] == stage and now > previous["unix"]:
        delta = value - previous["token"][1]
        if delta >= 0:
            speed = delta / (now - previous["unix"])
    state["progress"] = {"token": token, "unix": now}
    disk = shutil.disk_usage(".")
    snapshot = {"checked_unix": now, "monitor_pid": os.getpid(), "stage": stage,
                "pipeline_process": process, "children": children, "extraction": extraction,
                "formal_progress": formal, "progress_age_seconds": now - state["last_advance_unix"],
                "speed_per_second": speed, "speed_unit": "optimizer_steps" if stage == "formal_training" else "audio_files",
                "disk": {"free_gib": disk.free / 2**30, "total_gib": disk.total / 2**30,
                         "used_percent": 100 * disk.used / disk.total},
                "training": collect_metrics(state.setdefault("metrics", {}))}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=8)
        values = result.stdout.strip().splitlines()[0].split(",")
        snapshot["gpu"] = dict(zip(("utilization_percent", "used_mib", "total_mib", "temperature_c", "power_w"),
                                    (float(value.strip()) for value in values)))
        if not all(math.isfinite(value) for value in snapshot["gpu"].values()):
            snapshot.pop("gpu")
            raise ValueError("GPU 遥测含非有限数值")
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        snapshot["gpu_error"] = type(error).__name__
    snapshot["alerts"] = assess(snapshot)
    return snapshot


def scalar_snapshot(snapshot: dict) -> dict[str, float]:
    values = {f"gpu/{key}": value for key, value in snapshot.get("gpu", {}).items()}
    values.update({f"disk/{key}": value for key, value in snapshot["disk"].items()})
    values.update({"health/alert_count": len(snapshot["alerts"]),
                   "health/critical_count": sum(item["severity"] == "critical" for item in snapshot["alerts"]),
                   "health/progress_age_seconds": snapshot["progress_age_seconds"],
                   "health/pipeline_alive": int(snapshot["pipeline_process"]["alive"]),
                   "health/worker_count": sum(item["alive"] for item in snapshot["children"])})
    if snapshot["speed_per_second"] is not None:
        values["speed/" + snapshot["speed_unit"] + "_per_second"] = snapshot["speed_per_second"]
    for key in ("rolling_loss_100", "baseline_validation", "best_validation"):
        if snapshot["training"].get(key) is not None:
            values["formal/" + key] = snapshot["training"][key]
    return {key: float(value) for key, value in values.items() if math.isfinite(float(value))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    Path("logs").mkdir(exist_ok=True)
    import fcntl
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    with Path("logs/monitor.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = Path("logs/monitor_state.json")
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        writer = EventFileWriter("training/tensorboard/system")
        try:
            while True:
                try:
                    snapshot = collect(state)
                    step = int(state.get("sample", 0)) + 1
                    values = [Summary.Value(tag=key, simple_value=value) for key, value in scalar_snapshot(snapshot).items()]
                    writer.add_event(Event(wall_time=snapshot["checked_unix"], step=step, summary=Summary(value=values)))
                    writer.flush()
                    state["sample"] = step
                    write_json(path, state)
                    write_json(Path("logs/monitor_latest.json"), snapshot)
                    with Path("logs/monitor_history.jsonl").open("a", encoding="utf-8") as history:
                        history.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + "\n")
                    print(json.dumps({"sample": step, "stage": snapshot["stage"], "alerts": snapshot["alerts"]}, ensure_ascii=False), flush=True)
                    if snapshot["stage"] in ("complete", "failed"):
                        return
                except Exception as error:
                    # 观测失败保留旧快照并单独记录错误，不将临时观测错误伪装成训练失败。
                    write_json(Path("logs/monitor_error.json"), {"checked_unix": time.time(),
                               "type": type(error).__name__, "message": str(error)})
                    print(f"monitor_error: {type(error).__name__}: {error}", flush=True)
                time.sleep(max(5, args.interval))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
