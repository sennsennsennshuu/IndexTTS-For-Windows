"""云端训练流水线：先实测小样本可学习性，再提取全量特征并执行正式训练。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from heldout_cases import select_paired_cases

STAGES = ["cloud_regressions", "data_audit", "pilot_extraction", "codec_check", "baseline_audio", "overfit_training",
          "adapted_pilot_audio", "resume_check", "full_extraction", "formal_training", "heldout_audio"]
FIRST_STAGE = 0


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def state(stage: str, **extra) -> None:
    value = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **extra}
    write_json(Path("logs/pipeline_status.json"), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_stage(stage: str, command: list[str]) -> None:
    if STAGES.index(stage) < FIRST_STAGE:
        print(f"Reuse previously verified stage: {stage}", flush=True)
        return
    state(stage, command=command)
    with Path(f"logs/{stage}.log").open("a", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())


def load_rows(path: str):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: str, rows) -> None:
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_pilot() -> tuple[dict, list[dict]]:
    """从各来源挑选固定小集合，保持已有 split，不以训练句冒充泛化测试。"""
    train = load_rows("training/data/minnan/train.cloud.jsonl")
    valid = load_rows("training/data/minnan/valid.cloud.jsonl")
    selected = []
    for rows, split, quotas in ((train, "train", {"leku": 4, "sutiau": 2, "cv": 2}),
                                (valid, "valid", {"leku": 2, "sutiau": 1, "cv": 1})):
        chosen = set()
        for source, quota in quotas.items():
            # 优先完整短句，使验证开销可控，仍保留真实文本与完整音频。
            candidates = sorted((row for row in rows if row["source"] == source and 1.0 <= float(row["duration"]) <= 8.0),
                                key=lambda row: (abs(float(row["duration"]) - 3.0), row["audio_sha256"]))
            for row in candidates:
                if row["audio_sha256"] not in chosen:
                    chosen.add(row["audio_sha256"])
                    quota -= 1
                    if quota == 0:
                        break
        subset = [row for row in rows if row["audio_sha256"] in chosen]
        if not subset:
            raise RuntimeError(f"试运行 {split} 数据为空")
        write_rows(f"training/data/minnan/pilot_{split}.cloud.jsonl", subset)
        if split == "train":
            selected = subset
    leku = [row for row in selected if row["source"] == "leku" and row["text_type"] == "hanzi"]
    if len(leku) < 2:
        raise RuntimeError("缺少两条已精确配对的例句，不能生成 codec 验证")
    codec_cases = [{"name": f"codec_{index}", "target_audio": row["audio"], "reference_audio": leku[(index + 1) % len(leku)]["audio"]}
                   for index, row in enumerate(leku[:2])]
    cases = []
    for index, row in enumerate(leku[:2]):
        views = [candidate for candidate in selected if candidate["audio_sha256"] == row["audio_sha256"]]
        for view in views:
            cases.append({"name": f"pilot_{index}_{view['text_type']}", "text": view["text"],
                          "reference_audio": leku[(index + 1) % len(leku)]["audio"],
                          "evaluation_scope": "seen_training_text_for_overfit_only"})
    write_json(Path("training/data/minnan/codec_cases.json"), codec_cases)
    write_json(Path("training/data/minnan/pilot_cases.json"), cases)
    config = {"train_manifest": "training/data/minnan/pilot_train.cloud.jsonl",
              "valid_manifest": "training/data/minnan/pilot_valid.cloud.jsonl",
              "output_dir": "training/runs/overfit", "steps": 200, "grad_accum": 1,
              "validate_every": 50, "save_every": 50, "max_valid": 8, "warmup_steps": 10,
              "lr": 0.0002, "row_lr": 0.00003, "rank": 32, "alpha": 64,
              "dropout": 0.05, "seed": 20261105, "gradient_checkpointing": True, "sdpa": True}
    write_json(Path("training/configs/overfit.json"), config)
    return config, cases


def main(from_stage: str = "cloud_regressions") -> None:
    global FIRST_STAGE
    FIRST_STAGE = STAGES.index(from_stage)
    if Path.cwd().resolve() != Path("/root/autodl-tmp/IndexTTS25_Minnan"):
        raise RuntimeError("流水线只允许在指定 tmp 训练项目内执行")
    Path("logs").mkdir(exist_ok=True)
    lock = Path("logs/pipeline.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for key, value in {"TMPDIR": "cache/tmp", "HF_HOME": "cache/huggingface", "XDG_CACHE_HOME": "cache",
                       "NUMBA_CACHE_DIR": "cache/numba", "MPLCONFIGDIR": "cache/matplotlib"}.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(Path(value).resolve())
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    Path("logs/pipeline.pid").write_text(str(os.getpid()), encoding="utf-8")
    python = "even/bin/python"
    for marker in ("logs/setup_complete", "logs/runtime_complete", "logs/models_complete"):
        if not Path(marker).exists():
            raise RuntimeError(f"前置准备未完成：{marker}")
    if FIRST_STAGE > STAGES.index("data_audit"):
        audit = json.loads(Path("logs/data_audit.json").read_text(encoding="utf-8"))
        if not audit.get("passed"):
            raise RuntimeError("从中间阶段恢复前必须已有通过的数据审计")
    if FIRST_STAGE > STAGES.index("baseline_audio"):
        for report_name in ("codec_check", "baseline"):
            report = json.loads(Path(f"outputs/{report_name}/report.json").read_text(encoding="utf-8"))
            if not report.get("cases") or not all(case["finite"] for case in report["cases"]):
                raise RuntimeError("缺少已确认的重建／基础合成回执，不能跳过前置阶段")
    run_stage("cloud_regressions", [python, "-I", "-B", "-m", "unittest", "discover", "-s", "test", "-p", "test_cloud*.py", "-v"])
    run_stage("data_audit", [python, "-u", "tools/cloud/audit_data.py"])
    make_pilot()
    run_stage("pilot_extraction", [python, "-u", "tools/cloud/extract.py", "--manifests",
                                  "training/data/minnan/pilot_train.cloud.jsonl", "training/data/minnan/pilot_valid.cloud.jsonl"])
    run_stage("codec_check", [python, "-u", "tools/cloud/evaluate.py", "--codec-only", "--cases", "training/data/minnan/codec_cases.json", "--output", "outputs/codec_check"])
    run_stage("baseline_audio", [python, "-u", "tools/cloud/evaluate.py", "--cases", "training/data/minnan/pilot_cases.json", "--output", "outputs/baseline"])
    overfit_command = [python, "-u", "tools/cloud/train.py", "--config", "training/configs/overfit.json"]
    checkpoints = sorted(Path("training/runs/overfit/checkpoints").glob("step_*"))
    if checkpoints:
        overfit_command.extend(["--resume", str(checkpoints[-1])])
    run_stage("overfit_training", overfit_command)
    metrics = load_rows("training/runs/overfit/metrics.jsonl")
    losses = [float(row["loss_per_token"]) for row in metrics if "loss_per_token" in row and row.get("step")]
    if len(losses) < 40:
        raise RuntimeError("小样本训练没有足够的真实更新记录")
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    gate = {"initial_mean_ce": first, "final_mean_ce": last, "relative_drop": 1 - last / first,
            "passed": last < first * 0.75, "native_listening_passed": False}
    if FIRST_STAGE > STAGES.index("overfit_training"):
        # 200→201 步续训后从提取阶段恢复时，保留原 200 步可学习性回执。
        gate = json.loads(Path("logs/overfit_gate.json").read_text(encoding="utf-8"))
    else:
        write_json(Path("logs/overfit_gate.json"), gate)
    if not gate["passed"]:
        raise RuntimeError("小样本可学习性门未通过，停止正式训练并保留现场")
    progress = json.loads(Path("training/runs/overfit/progress.json").read_text(encoding="utf-8"))
    checkpoint = progress["last_checkpoint"]
    run_stage("adapted_pilot_audio", [python, "-u", "tools/cloud/evaluate.py", "--cases", "training/data/minnan/pilot_cases.json", "--adapter", checkpoint, "--output", "outputs/overfit"])
    # 使用真实检查点继续一个优化步，验证 optimizer／scheduler／RNG 能恢复。
    run_stage("resume_check", [python, "-u", "tools/cloud/train.py", "--config", "training/configs/overfit.json", "--resume", checkpoint, "--steps", "201"])
    state("pilot_verified", overfit_gate=gate)
    run_stage("full_extraction", [python, "-u", "tools/cloud/extract.py"])
    # 约 4.7 万条训练录音，15000*8 次采样覆盖两个完整轮次以上，
    # 让同一辞典音频的汉字／台罗视图都实际参与，而非只学习随机一半视图。
    formal = {"output_dir": "training/runs/minnan-lora", "steps": 15000, "batch_size": 4, "grad_accum": 2,
              "validate_every": 100, "save_every": 100, "max_valid": 64, "warmup_steps": 100,
              "lr": 0.0001, "row_lr": 0.00003, "rank": 32, "alpha": 64,
              "dropout": 0.05, "seed": 20261105, "gradient_checkpointing": True, "sdpa": True}
    write_json(Path("training/configs/formal.json"), formal)
    run_stage("formal_training", [python, "-u", "tools/cloud/train.py", "--config", "training/configs/formal.json"])
    final = json.loads(Path("training/runs/minnan-lora/progress.json").read_text(encoding="utf-8"))
    best = json.loads(Path("training/runs/minnan-lora/best.json").read_text(encoding="utf-8"))
    heldout = load_rows("training/data/minnan/test.cloud.jsonl")
    evaluation = select_paired_cases(heldout, utterances=4)
    write_json(Path("training/data/minnan/heldout_cases.json"), evaluation)
    run_stage("heldout_audio", [python, "-u", "tools/cloud/evaluate.py", "--cases", "training/data/minnan/heldout_cases.json", "--adapter", best["path"], "--output", "outputs/heldout"])
    package = Path("training/runs/minnan-delivery.tar.gz")
    with tarfile.open(package, "w:gz") as archive:
        checkpoint = Path(best["path"])
        archive.add(checkpoint / "adapter", arcname="adapter")
        archive.add(checkpoint / "extra_state.pt", arcname="extra_state.pt")
        archive.add("training/configs/formal.json", arcname="formal.json")
        archive.add("training/base_model_manifest.json", arcname="base_model_manifest.json")
        archive.add("training/data/minnan/data_report.json", arcname="data_report.json")
        archive.add("training/runs/minnan-lora/metrics.jsonl", arcname="training_metrics.jsonl")
        archive.add("outputs/heldout", arcname="heldout_audio")
    with package.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    training_metrics = load_rows("training/runs/minnan-lora/metrics.jsonl")
    initial_validation = next((row["val_loss"] for row in training_metrics if row.get("event") == "validation_initial"), None)
    state("complete", checkpoint=final["last_checkpoint"], best=best, package=package.as_posix(), sha256=sha,
          baseline_validation_loss=initial_validation,
          validation_improved_over_initial=initial_validation is not None and best["val_loss"] < initial_validation,
          native_listening_passed=False, delivery_scope="trained_adapter_and_engineering_checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default="cloud_regressions",
                        help="修复失败阶段后继续；已完成阶段必须保留验证回执")
    args = parser.parse_args()
    try:
        main(args.from_stage)
    except Exception as error:
        state("failed", error_type=type(error).__name__, error=str(error))
        raise
