"""训练终态后补做八条成对测试、原始录音和全程采样审计，再发布私有交付包。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from delivery_checks import digest, validate_cases, validate_report
from heldout_cases import select_paired_cases
from tensorboard_bridge import write_json


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state(stage: str, **details) -> None:
    result = {"stage": stage, "pid": os.getpid(), "updated_unix": time.time(), **details}
    write_json(Path("logs/final_delivery_status.json"), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def run(stage: str, command: list[str]) -> None:
    state(stage, command=command)
    with Path(f"logs/final_delivery_{stage}.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    from runtime import configure
    configure()
    pipeline = read("logs/pipeline_status.json")
    progress = read("training/runs/minnan-lora/progress.json")
    ready = (pipeline["stage"] == "complete" and progress.get("status") == "complete"
             and progress.get("step") == 15000 and progress.get("total") == 15000)
    if args.check_only:
        print(json.dumps({"ready": ready, "pipeline_stage": pipeline["stage"], "step": progress["step"]}))
        return
    if not ready:
        raise RuntimeError("正式训练及父流水线尚未完成，暂不执行最终合成或发布")
    best = read("training/runs/minnan-lora/best.json")
    if not best["val_loss"] < pipeline["baseline_validation_loss"]:
        raise ValueError("最佳验证未优于基础模型，暂不发布为已改善的适配器")
    checkpoint = Path(best["path"])
    checkpoint.resolve().relative_to(Path("training/runs/minnan-lora/checkpoints").resolve())
    status_path = Path("logs/final_delivery_status.json")
    if status_path.exists():
        previous = read(status_path)
        if previous.get("stage") == "complete" and previous.get("best") == best:
            if digest(Path(previous["package"])) != previous["sha256"]:
                raise ValueError("现有最终包摘要变化，保留现场并停止")
            print(json.dumps(previous), flush=True)
            return
    run("sampling_audit", ["even/bin/python", "-B", "tools/cloud/audit_sampling.py", "--checkpoint",
                           "training/runs/minnan-lora/checkpoints/step_015000"])
    audit_path = Path("logs/sampling_audit_step_015000.json")
    audit = read(audit_path)
    if audit["optimizer_step"] != 15000 or audit["sampler_cursor"] != 120000 or not audit["all_text_views_seen"]:
        raise ValueError("最终采样与双输入覆盖没有通过")
    rows = [json.loads(line) for line in Path("training/data/minnan/test.cloud.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = select_paired_cases(rows, utterances=4)
    validate_cases(cases)
    cases_path = Path("training/data/minnan/heldout_paired_cases.json")
    if cases_path.exists() and read(cases_path) != cases:
        raise ValueError("最终 test 清单与事先固定的选样不一致")
    write_json(cases_path, cases)
    output = Path("outputs/heldout_paired")
    request = {"adapter_sha256": digest(checkpoint / "adapter/adapter_model.safetensors"),
               "extra_state_sha256": digest(checkpoint / "extra_state.pt"), "cases_sha256": digest(cases_path)}
    cached = (output / "generation_request.json").is_file() and read(output / "generation_request.json") == request
    if not cached or not (output / "report.json").is_file():
        run("paired_audio", ["even/bin/python", "-u", "-B", "tools/cloud/evaluate.py", "--cases", str(cases_path),
                             "--adapter", str(checkpoint), "--output", str(output)])
        write_json(output / "generation_request.json", request)
    report = read(output / "report.json")
    validate_report(cases, report, str(checkpoint))
    from evaluate import waveform_report
    for index, case in enumerate(cases):
        waveform_report(output / f"{index:02d}_{case['name']}.wav")
    # 保存未转换的 test 原录音和真实参考，合成音色以参考录音为条件。
    originals = {}
    for case in cases:
        for kind, source_key, hash_key in (("target", "target_audio", "target_audio_sha256"),
                                           ("reference", "reference_audio", "reference_audio_sha256")):
            expected = case[hash_key]
            if expected in originals:
                if kind not in originals[expected]["roles"]:
                    originals[expected]["roles"].append(kind)
                continue
            source = Path(case[source_key])
            source.resolve().relative_to(Path.cwd().parent)
            if digest(source) != expected:
                raise ValueError("test 原始音频或参考内容变化")
            destination = output / "original_audio" / f"{expected}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != expected:
                raise ValueError("原始录音复制摘要不符")
            originals[expected] = {"file": f"original_audio/{expected}.wav", "roles": [kind], "converted": False}
    write_json(output / "original_audio.json", originals)
    state("packaging", best=best)
    meta = Path("outputs/final_delivery_meta")
    meta.mkdir(parents=True, exist_ok=True)
    description = ("# 台语 IndexTTS 2.5 LoRA 训练交付\n\n"
                   "本包包含最佳 GPT LoRA、minnan 语言行、完整训练指标及四组汉字／带调台罗测试。\n"
                   "基础模型必须与 base_model_manifest.json 相同，使用项目 tools/cloud/evaluate.py 加载 adapter 与 extra_state.pt。\n"
                   "云端复现命令：even/bin/python -B tools/cloud/evaluate.py --cases training/data/minnan/heldout_paired_cases.json "
                   f"--adapter {checkpoint.as_posix()} --output outputs/reproduced_minnan\n\n"
                   "heldout_audio/cases.json 记录实际输入，original_audio.json 定位未转换的目标及参考录音。\n"
                   "原录音用于核对发音内容，合成音色由 reference_audio 决定。工程检查通过，台语母语者听感仍待评审。\n"
                   "TensorBoard 历史位于 tensorboard/；完整采样核验位于 sampling_audit.json。\n")
    (meta / "README.md").write_text(description, encoding="utf-8")
    sources = {path.as_posix(): digest(path) for path in Path("tools/cloud").glob("*.py")}
    write_json(meta / "tool_sources.json", sources)
    files = []
    def add(source: Path, relative: str):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"交付源不是普通文件：{source}")
        files.append((source, relative))
    for path in sorted((checkpoint / "adapter").rglob("*")):
        if path.is_file():
            add(path, "adapter/" + path.relative_to(checkpoint / "adapter").as_posix())
    for source, name in ((checkpoint / "extra_state.pt", "extra_state.pt"),
                         (Path("training/configs/formal.json"), "formal.json"),
                         (Path("training/base_model_manifest.json"), "base_model_manifest.json"),
                         (Path("training/data/minnan/data_report.json"), "data_report.json"),
                         (Path("training/runs/minnan-lora/metrics.jsonl"), "training_metrics.jsonl"),
                         (Path("training/runs/minnan-lora/best.json"), "best.json"),
                         (audit_path, "sampling_audit.json"), (meta / "README.md", "README.md"),
                         (meta / "tool_sources.json", "tool_sources.json")):
        add(source, name)
    for folder, prefix in ((output, "heldout_audio"), (Path("training/tensorboard"), "tensorboard")):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                add(path, prefix + "/" + path.relative_to(folder).as_posix())
    for name in ("monitor_history.jsonl", "formal_first_checkpoint_audit.json"):
        path = Path("logs") / name
        if path.is_file():
            snapshot = meta / "monitoring" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot)
            add(snapshot, "monitoring/" + name)
    manifest = {"format": 1, "model_type": "IndexTTS-2.5 GPT LoRA with minnan language row",
                "paired_heldout_verified": True, "paired_cases": 8, "sampling_audit_steps": 15000,
                "best": best, "native_listening_passed": False,
                "files": [{"path": name, "bytes": source.stat().st_size, "sha256": digest(source)} for source, name in files]}
    manifest_path = meta / "delivery_manifest.json"
    write_json(manifest_path, manifest)
    add(manifest_path, "delivery_manifest.json")
    package = Path("training/runs/minnan-delivery.tar.gz")
    candidate = package.with_name(package.name + ".paired.part")
    with tarfile.open(candidate, "w:gz") as archive:
        for source, name in files:
            archive.add(source, arcname=name, recursive=False)
    # 重新解码归档内容核验每个成员，之后才允许替换旧流水线包并开放下载门禁。
    with tarfile.open(candidate, "r:gz") as archive:
        expected_names = {name for _, name in files}
        if {member.name for member in archive.getmembers()} != expected_names:
            raise ValueError("最终归档成员与清单不一致")
        for entry in manifest["files"]:
            stream = archive.extractfile(entry["path"])
            checksum = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
            if checksum.hexdigest() != entry["sha256"]:
                raise ValueError("最终归档成员摘要不符")
    candidate.replace(package)
    checksum = digest(package)
    previous_sha = pipeline.get("sha256")
    pipeline.update({"package": package.as_posix(), "sha256": checksum, "paired_heldout_verified": True,
                     "paired_cases": 8, "sampling_audit_steps": 15000, "superseded_package_sha256": previous_sha,
                     "finalized_utc": datetime.now(timezone.utc).isoformat(),
                     "delivery_scope": "trained_adapter_paired_test_audio_and_full_sampling_audit"})
    write_json(Path("logs/pipeline_status.json"), pipeline)
    state("complete", package=package.as_posix(), sha256=checksum, bytes=package.stat().st_size,
          paired_heldout_verified=True, paired_cases=8, sampling_audit_steps=15000, best=best,
          native_listening_passed=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state("failed", error_type=type(error).__name__, error=str(error))
        raise
