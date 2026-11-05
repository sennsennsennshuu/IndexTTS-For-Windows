"""依据真实检查点、数据指纹和逐步日志，重放核对训练音频覆盖与文字视图轮换。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    from runtime import configure
    configure()
    # 本审计仅在 CPU 读取已有状态；不会给当前训练创建额外 CUDA 上下文。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path("training/runs/minnan-lora/checkpoints")
    checkpoint = args.checkpoint if args.checkpoint is not None else sorted(root.glob("step_*"))[-1]
    checkpoint.resolve().relative_to(root.resolve())
    # 提前打开文件，允许训练器随后正常清理旧目录而不打断已经开始的只读审计。
    with (checkpoint / "extra_state.pt").open("rb") as extra_stream, (checkpoint / "trainer_state.pt").open("rb") as state_stream:
        import torch
        extra = torch.load(extra_stream, map_location="cpu", weights_only=True)
        state = torch.load(state_stream, map_location="cpu", weights_only=False)
    from train import ViewSampler, _group_views, _load_jsonl, _manifest_path, _data_fingerprint
    from tensorboard_bridge import read_complete_records
    config = extra["config"]
    rows = _load_jsonl(_manifest_path(config, "train"), "train")
    valid = _load_jsonl(_manifest_path(config, "valid"), "valid")
    if _data_fingerprint(rows + valid) != extra["data_fingerprint"]:
        raise ValueError("当前数据与检查点指纹不一致，不能用新数据推断旧训练覆盖")
    groups = _group_views(rows)
    sampler = ViewSampler(groups, int(config["seed"]))
    effective = int(config["batch_size"]) * int(config["grad_accum"])
    if int(state["cursor"]) != int(state["step"]) * effective:
        raise ValueError("检查点实际采样游标与优化步数／有效 batch 不匹配")
    seen = defaultdict(set)
    # 音频覆盖与文字视图覆盖分别计数，同一音频的汉字／台罗不能合并成一个输入。
    manifest_views = {(row["audio_sha256"], row["text_type"], row["text"]) for row in rows}
    seen_views = set()
    counts = defaultdict(Counter)
    previous = {}
    rotations = Counter()
    expected_last = {}
    for cursor in range(int(state["cursor"])):
        row, next_cursor, view_index = sampler.next(cursor)
        if next_cursor != cursor + 1:
            raise ValueError("采样游标没有逐条递增")
        epoch = cursor // len(groups)
        key = row["audio_sha256"]
        if key in seen[epoch]:
            raise ValueError("同一轮内重复抽取同一音频，不能宣称完整覆盖")
        seen[epoch].add(key)
        seen_views.add((key, row["text_type"], row["text"]))
        counts[epoch][row["text_type"]] += 1
        if key in previous and len(groups[key]) == 2 and {value["text_type"] for value in groups[key]} == {"hanzi", "tailo"}:
            old_epoch, old_kind = previous[key]
            if epoch != old_epoch + 1 or old_kind == row["text_type"]:
                raise ValueError("同一音频在相邻轮次没有切换成对汉字／台罗视图")
            rotations[epoch + 1] += 1
        previous[key] = (epoch, row["text_type"])
        if next_cursor % effective == 0:
            expected_last[next_cursor // effective] = (key, row["text_type"])
    completed = int(state["cursor"]) // len(groups)
    for epoch in range(completed):
        if len(seen[epoch]) != len(groups):
            raise ValueError("已经完成的轮次没有覆盖全部训练音频")
    if completed >= 2 and seen_views != manifest_views:
        raise ValueError("两轮音频覆盖后仍有文字视图未参与，不能通过双输入覆盖验收")
    matched = set()
    metric_rows = 0
    for _, row in read_complete_records(Path("training/runs/minnan-lora/metrics.jsonl"), 0):
        step = int(row["step"])
        if "loss_per_token" not in row or step > int(state["step"]):
            continue
        if (row["sample_audio_sha256"], row["text_type"]) != expected_last[step] or row["samples"] != effective:
            raise ValueError(f"第 {step} 步日志中的真实末样本与采样器重放不一致")
        matched.add(step)
        metric_rows += 1
    if matched != set(range(1, int(state["step"]) + 1)):
        raise ValueError("逐步训练记录不完整，不能通过日志交叉核对")
    result = {"checked_unix": time.time(), "checkpoint": checkpoint.as_posix(), "optimizer_step": state["step"],
              "sampler_cursor": state["cursor"], "samples_per_update": effective,
              "unique_train_audio": len(groups), "manifest_text_views": len(rows), "completed_epochs": completed,
              "unique_manifest_text_views": len(manifest_views), "unique_text_views_seen": len(seen_views),
              "unseen_text_views": len(manifest_views - seen_views), "all_text_views_seen": seen_views == manifest_views,
              "epoch_numbering": "one_based", "epoch_coverage": [
                  {"epoch": epoch + 1, "unique_audio": len(seen[epoch]), "text_types": dict(counts[epoch]),
                   "complete": epoch < completed} for epoch in sorted(seen)],
              "paired_view_rotations_by_epoch": dict(rotations), "matched_last_sample_steps": len(matched),
              "metric_rows": metric_rows, "duplicate_log_rows": metric_rows - len(matched),
              "data_fingerprint_matches": True, "data_fingerprint": extra["data_fingerprint"],
              "trainer_source_sha256": hashlib.sha256(Path("tools/cloud/train.py").read_bytes()).hexdigest(),
              "coverage_method": "checkpoint cursor plus deterministic sampler replay and per-step last-sample log verification"}
    output = args.output or Path(f"logs/sampling_audit_step_{state['step']:06d}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
