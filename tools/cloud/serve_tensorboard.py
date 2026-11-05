"""在 tmp 项目内启动独立 TensorBoard 与日志桥接器，保留现有训练进程。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


def start(name: str, arguments: list[str], environment: dict[str, str], previous: dict) -> int:
    """只复用身份、工作目录和启动时间均匹配的既有进程，不停止其他服务。"""
    proc = Path("/proc") / str(previous.get("pid", 0))
    if proc.exists():
        command = (proc / "cmdline").read_bytes().split(b"\0")
        expected = [item.encode() for item in arguments]
        if (command[:len(expected)] == expected and (proc / "cwd").resolve() == Path.cwd()
                and (proc / "stat").read_text().split()[21] == previous.get("start_ticks")):
            return int(previous["pid"])
    if name == "server":
        # 发现端口被其他服务占用时明确停止启动，不覆盖平台的 6007 TensorBoard。
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 6008))
    with Path(f"logs/tensorboard_{name}.log").open("ab") as output:
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    return process.pid


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[2])
    # TensorBoard 本身会使用临时目录；在导入／启动前将其定向到项目数据盘。
    for relative in ("cache/tmp", "cache/tensorboard", "logs"):
        Path(relative).mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"TMPDIR": str(Path("cache/tmp").resolve()),
                        "TMP": str(Path("cache/tmp").resolve()),
                        "TEMP": str(Path("cache/tmp").resolve()),
                        "XDG_CACHE_HOME": str(Path("cache/tensorboard").resolve()),
                        "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    state_path = Path("logs/tensorboard_services.json")
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    commands = {
        "monitor": ["even/bin/python", "-u", "-B", "tools/cloud/monitor.py", "--interval", "30"],
        "bridge": ["even/bin/python", "-u", "-B", "tools/cloud/tensorboard_bridge.py", "--interval", "10"],
        "server": ["even/bin/python", "-u", "-B", "-m", "tensorboard.main",
                   "--logdir", "training/tensorboard", "--host", "127.0.0.1", "--port", "6008",
                   "--reload_interval", "10", "--load_fast=false"],
    }
    services = {}
    for name, arguments in commands.items():
        pid = start(name, arguments, environment, previous.get(name, {}))
        services[name] = {"pid": pid, "start_ticks": (Path("/proc") / str(pid) / "stat").read_text().split()[21]}
        state_path.write_text(json.dumps({**previous, **services}, indent=2), encoding="utf-8")
    # 真实 HTTP 成功才返回可用状态；提前退出的进程和错误日志均保留供排查。
    for _ in range(50):
        for name, metadata in services.items():
            stat = Path("/proc") / str(metadata["pid"]) / "stat"
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                raise RuntimeError(f"{name} 启动失败，请检查 logs/tensorboard_{name}.log")
        try:
            with urlopen("http://127.0.0.1:6008/data/logdir", timeout=2) as response:
                health = json.load(response)
            print(json.dumps({"services": services, "http": health}), flush=True)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("TensorBoard HTTP 尚未就绪，请检查项目内服务日志")


if __name__ == "__main__":
    main()
