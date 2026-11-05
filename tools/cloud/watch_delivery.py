"""在本机后台同步真实云端状态，训练结束后校验并取回适配器，不发送外部消息。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect
from delivery_checks import delivery_ready, validate_package

REMOTE_ROOT = "/root/autodl-tmp/IndexTTS25_Minnan/"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(sftp, relative: str):
    with sftp.open(REMOTE_ROOT + relative, "r") as stream:
        return json.loads(stream.read().decode("utf-8"))


def check_todo(needle: str) -> None:
    """只更新任务对应的一行，不重写用户的其他待办。"""
    path = Path("docs/TODO.md")
    text = path.read_text(encoding="utf-8")
    text = text.replace("- [ ] " + needle, "- [x] " + needle)
    path.write_text(text, encoding="utf-8", newline="\n")


def receive_delivery(sftp, status: dict) -> dict:
    expected = status.get("sha256", "")
    if status.get("package") != "training/runs/minnan-delivery.tar.gz" or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("云端交付路径或摘要不符合约定")
    root = Path("models/extra_language/zh-nan-tw")
    root.mkdir(parents=True, exist_ok=True)
    final = root / "model"
    receipt_path = final / "delivery_receipt.json"
    if receipt_path.is_file() and json.loads(receipt_path.read_text(encoding="utf-8"))["sha256"] == expected:
        validate_package(final)
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    archive_path = Path("cache/cloud_delivery/minnan-delivery.tar.gz.part")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(REMOTE_ROOT + status["package"], str(archive_path))
    with archive_path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("正式模型下载摘要不符，保留现场，不交付")
    candidate = root / (".candidate-" + uuid.uuid4().hex)
    candidate.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(candidate, filter="data")
    if not (candidate / "adapter/adapter_model.safetensors").is_file() or not (candidate / "extra_state.pt").is_file():
        raise ValueError("交付包缺少 LoRA 或台语语言嵌入")
    manifest = validate_package(candidate)
    receipt = {"sha256": actual, "bytes": archive_path.stat().st_size, "local_directory": str(final),
               "remote_checkpoint": status.get("best"), "received_unix": time.time(),
               "formal_training_complete": True, "native_listening_passed": False,
               "paired_cases": manifest["paired_cases"], "full_sampling_audit_passed": True}
    write_json(candidate / "delivery_receipt.json", receipt)
    # 固定语言包入口；只有新候选逐文件验收通过，才切换当前模型。
    previous = None
    if final.exists():
        validate_package(final)
        old_receipt = json.loads((final / "delivery_receipt.json").read_text(encoding="utf-8"))
        previous = root / "history" / old_receipt["sha256"][:12]
        previous.parent.mkdir(parents=True, exist_ok=True)
        if previous.exists():
            raise FileExistsError("历史模型目录已存在，保留当前模型及候选，不覆盖历史证据")
        for path in (final, previous):
            path.resolve().relative_to(Path.cwd().resolve())
        final.replace(previous)
    try:
        candidate.replace(final)
    except OSError:
        if previous is not None:
            previous.replace(final)
        raise
    archive_path.unlink()
    write_json(Path("docs/cloud_minnan_delivery.json"), receipt)
    check_todo("下载并验证模型与试听样例，整理复现说明及剩余听感验收项。")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    Path("logs").mkdir(exist_ok=True)
    # 锁跟随进程释放，避免重启后台同步时出现两个实例竞争交付目录。
    lock = Path("logs/cloud_delivery_watch.lock").open("a+b")
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    Path("logs/cloud_delivery_watch.pid").write_text(str(os.getpid()), encoding="utf-8")
    while True:
        try:
            with connect() as client, client.open_sftp() as sftp:
                sftp.get_channel().settimeout(60)
                remote = read_json(sftp, "logs/pipeline_status.json")
                snapshot = {"watcher_pid": os.getpid(), "checked_unix": time.time(), "remote": remote}
                for name, path in (("extraction", "logs/extraction_progress.json"),
                                   ("formal_training", "training/runs/minnan-lora/progress.json"),
                                   ("monitoring", "logs/monitor_latest.json"),
                                   ("finalization", "logs/final_delivery_status.json"),
                                   ("pilot_training", "training/runs/overfit/progress.json")):
                    try:
                        snapshot[name] = read_json(sftp, path)
                    except FileNotFoundError:
                        pass
                formal = snapshot.get("formal_training", {})
                if int(formal.get("step", 0)) > 0:
                    check_todo("启动正式训练，记录进程、日志、检查点和验证结果。")
                if delivery_ready(remote):
                    snapshot["delivery"] = receive_delivery(sftp, remote)
                elif remote["stage"] == "complete":
                    snapshot["waiting_for_paired_delivery"] = True
                write_json(Path("logs/cloud_training_live.json"), snapshot)
                print(json.dumps({"stage": remote["stage"], "checked_unix": snapshot["checked_unix"]}), flush=True)
                if remote["stage"] == "failed" or "delivery" in snapshot:
                    return
        except Exception as error:
            # 网络重连保留上次有效状态；不打印凭据，不把连接失败当作训练完成。
            write_json(Path("logs/cloud_delivery_watch_error.json"),
                       {"type": type(error).__name__, "message": str(error), "time": time.time()})
            print(type(error).__name__, str(error), flush=True)
        if args.once:
            return
        for _ in range(max(1, args.interval // 10)):
            time.sleep(10)


if __name__ == "__main__":
    main()
