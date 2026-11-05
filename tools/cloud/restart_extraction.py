"""在批量方案验证后恢复特征任务；仅终止本项目的指定流水线及其提取子进程。"""

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main() -> None:
    root = Path("/root/autodl-tmp/IndexTTS25_Minnan")
    os.chdir(root)
    status = json.loads(Path("logs/pipeline_status.json").read_text())
    if status["stage"] != "full_extraction":
        raise RuntimeError("只允许在全量提取阶段恢复，不打断已开始的正式参数更新")
    pid = int(status["pid"])
    owner = Path(f"/proc/{pid}")
    if (owner / "cwd").resolve() != root or b"tools/cloud/pipeline.py" not in (owner / "cmdline").read_bytes():
        raise RuntimeError("进程身份或工作目录不符，拒绝终止")
    children = [int(value) for value in (owner / "task" / str(pid) / "children").read_text().split()]
    targets = []
    for child in children:
        directory = Path(f"/proc/{child}")
        if (directory / "cwd").resolve() == root and b"tools/cloud/extract.py" in (directory / "cmdline").read_bytes():
            targets.append(child)
    if len(targets) != 1:
        raise RuntimeError("未找到唯一的本项目特征提取子进程，保留现场")
    # 先终止父进程，避免它将正常的受控恢复写成失败；已完成特征均为原子提交。
    os.kill(pid, signal.SIGTERM)
    os.kill(targets[0], signal.SIGTERM)
    for _ in range(30):
        if not owner.exists() or "State:\tZ" in (owner / "status").read_text():
            break
        time.sleep(1)
    else:
        raise RuntimeError("原流水线未退出，不启动第二个实例")
    with Path("logs/pipeline.log").open("a") as stream:
        process = subprocess.Popen(["even/bin/python", "-u", "tools/cloud/pipeline.py", "--from-stage", "full_extraction"],
                                   stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({"terminated_pipeline": pid, "terminated_extractor": targets[0], "new_pipeline": process.pid,
                      "feature_cache_preserved": True}), flush=True)


if __name__ == "__main__":
    main()
