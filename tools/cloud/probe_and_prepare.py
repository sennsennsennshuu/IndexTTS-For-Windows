"""在用户指定的真实 GPU 服务器准备独立目录，不修改原有项目。"""

from pathlib import Path
import json
import os
import subprocess
import sys


root = Path("/root/autodl-tmp/IndexTTS25_Minnan")
root.mkdir(parents=True, exist_ok=True)
os.chdir(root)
for name in ("tools", "training", "models", "cache", "logs", "outputs", "test", "vendor"):
    Path(name).mkdir(exist_ok=True)
Path("cache/tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str((root / "cache/tmp").resolve())
if not Path("even/bin/python").exists():
    # 复用当前服务的 Python 解释器版本，包安装在项目 even/，不改动服务的全局包。
    subprocess.run([sys.executable, "-m", "venv", "even"], check=True)
inventory = {
    "root": str(root),
    "python_base": sys.version,
    "python_environment": "even/bin/python",
    "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], capture_output=True, text=True, check=True).stdout,
    "storage": subprocess.run(["df", "-h", "."], capture_output=True, text=True, check=True).stdout,
}
Path("logs/cloud_probe.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(inventory, ensure_ascii=False, indent=2))
