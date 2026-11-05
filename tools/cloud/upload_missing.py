"""仅补传云端缺失的原音频，保留源文件并逐项校验 SHA-256。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import connect, execute, upload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="logs/cloud_data_mapping.json", type=Path)
    parser.add_argument("--maximum-gib", type=float, default=3.0)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    missing = report["missing"]
    files = []
    total = 0
    for entry in missing:
        supplied = Path(entry["source_audio"])
        source_root = Path("../../minan-tw/source")
        # 本地 manifest 的 audio 相对 source_root；云端映射后才相对训练根目录。
        # 兼容显式带 ../../minan-tw/source 的清单，同时始终检查真实解析目标。
        source = supplied if supplied.is_file() else source_root / supplied
        # 用户指定的 source 是唯一允许补传的原始目录，不跟随目录链接读其他文件。
        allowed = source_root.resolve()
        if not source.resolve().is_relative_to(allowed):
            raise ValueError(f"补传路径超出用户指定语料目录：{source}")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["audio_sha256"]:
            raise ValueError(f"源音频自清单生成后发生变化：{source}")
        total += source.stat().st_size
        files.append((source, f"training/audio/{actual}{source.suffix.lower()}"))
    if total > args.maximum_gib * 2**30:
        raise ValueError(f"缺失音频 {total / 2**30:.2f} GiB 超过单批补传大小，需要拆批而非无限生成大包")
    print(json.dumps({"files": len(files), "bytes": total}), flush=True)
    if not files:
        return
    package = Path("cache/cloud_research/missing_audio.tar")
    with tarfile.open(package, "w") as tar:
        for source, destination in files:
            tar.add(source, arcname=destination, recursive=False)
    with connect() as client:
        upload(client, package, "/root/autodl-tmp/IndexTTS25_Minnan/cache/missing_audio.tar")
        status = execute(client, "cd /root/autodl-tmp/IndexTTS25_Minnan && tar -xf cache/missing_audio.tar && even/bin/python -u tools/cloud/map_data.py")
        if status:
            raise SystemExit(status)
    # 确认映射成功才移除当前任务产生的本地补传包；原始音频始终保留。
    package.unlink()


if __name__ == "__main__":
    main()
