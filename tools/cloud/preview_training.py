"""固定当前最佳适配器，在验证集生成成对汉字／台罗的基座和微调试听对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from heldout_cases import select_paired_cases


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pin_best() -> tuple[Path, dict]:
    """把所需文件原子固定到单独目录，避免试听期间被正常检查点保留策略清理。"""
    best = json.loads(Path("training/runs/minnan-lora/best.json").read_text(encoding="utf-8"))
    source = Path(best["path"])
    source.resolve().relative_to(Path("training/runs/minnan-lora/checkpoints").resolve())
    final = Path("training/evaluation_checkpoints") / f"step_{best['step']:06d}"
    if final.exists():
        receipt = json.loads((final / "pin_receipt.json").read_text(encoding="utf-8"))
        if not all(digest(final / name) == expected for name, expected in receipt["files"].items()):
            raise ValueError("已经固定的试听检查点摘要变化，保留现场并停止")
        return final, receipt
    candidate = final.with_name(final.name + f".{os.getpid()}.part")
    candidate.mkdir(parents=True)
    shutil.copytree(source / "adapter", candidate / "adapter")
    for name in ("extra_state.pt", "checkpoint.json"):
        shutil.copy2(source / name, candidate / name)
    hashes = {}
    for path in sorted(candidate.rglob("*")):
        if path.is_file():
            relative = path.relative_to(candidate).as_posix()
            hashes[relative] = digest(path)
            if hashes[relative] != digest(source / relative):
                raise ValueError("试听检查点复制摘要不一致，保留候选文件")
    receipt = {"source": source.as_posix(), "step": best["step"], "validation_loss": best["val_loss"],
               "pinned_unix": time.time(), "files": hashes}
    (candidate / "pin_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    candidate.replace(final)
    return final, receipt


def free_gpu_mib() -> float:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=8)
    return float(result.stdout.splitlines()[0].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=int, default=1)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    # 基座与微调推理串行执行，留出充分显存余量；余量不足就等待后续再试听。
    if free_gpu_mib() < 12 * 1024:
        raise RuntimeError("可用显存不足 12 GiB，本轮不启动并行试听")
    manifest = Path("training/data/minnan/valid.cloud.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = select_paired_cases(rows, utterances=args.utterances, split="valid")
    checkpoint, pinned = pin_best()
    output = Path("outputs/validation_previews") / f"step_{pinned['step']:06d}"
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.json"
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in ("baseline", "adapted"):
        if free_gpu_mib() < 12 * 1024:
            raise RuntimeError("显存余量变化，停止启动下一轮试听；训练进程不受影响")
        command = ["even/bin/python", "-u", "-B", "tools/cloud/evaluate.py", "--cases", str(cases_path),
                   "--output", str(output / mode)]
        if mode == "adapted":
            command.extend(["--adapter", str(checkpoint)])
        print(json.dumps({"preview": mode, "step": pinned["step"], "status": "starting"}), flush=True)
        with (output / f"{mode}.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    receipt = {"checkpoint": checkpoint.as_posix(), "step": pinned["step"], "split": "valid",
               "manifest_sha256": digest(manifest), "cases": len(cases), "modes": ["baseline", "adapted"],
               "automated_audio_checks_passed": True, "native_listening_passed": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
