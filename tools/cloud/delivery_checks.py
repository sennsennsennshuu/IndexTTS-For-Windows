"""正式交付的纯校验逻辑：训练终态、成对测试覆盖和逐文件完整性。"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import wave


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def delivery_ready(status: dict) -> bool:
    """父流水线完成后仍须等待补充验收，不能把旧四条汉字样例当作最终交付。"""
    return (status.get("stage") == "complete" and status.get("paired_heldout_verified") is True
            and status.get("paired_cases") == 8 and status.get("sampling_audit_steps") == 15000)


def validate_cases(cases: list[dict]) -> None:
    if len(cases) != 8 or len({row["name"] for row in cases}) != 8:
        raise ValueError("正式验收必须有八个不同案例")
    groups = defaultdict(list)
    for case in cases:
        if not re.fullmatch(r"heldout_[0-3]_(hanzi|tailo)", case["name"]):
            raise ValueError("案例名称不符合固定验收规则")
        if case.get("split") != "test" or case.get("evaluation_scope") != "heldout_text_paired_views":
            raise ValueError("最终验收只能使用真实 test 成对输入")
        if case["target_audio_sha256"] == case["reference_audio_sha256"]:
            raise ValueError("目标与参考音频不能相同")
        groups[case["target_audio_sha256"]].append(case)
    if len(groups) != 4:
        raise ValueError("正式验收必须覆盖四个目标录音")
    for rows in groups.values():
        if len(rows) != 2 or {row["text_type"] for row in rows} != {"hanzi", "tailo"}:
            raise ValueError("每个目标录音必须包含汉字和台罗")
        if len({row["reference_audio_sha256"] for row in rows}) != 1:
            raise ValueError("同组成对输入必须共享参考音频")


def validate_report(cases: list[dict], report: dict, adapter: str | None = None) -> None:
    validate_cases(cases)
    rows = report.get("cases", [])
    if len(rows) != 8 or len({row.get("name") for row in rows}) != 8:
        raise ValueError("真实合成报告没有完整覆盖八条输入")
    if adapter is not None and report.get("adapter") != adapter:
        raise ValueError("合成报告没有使用要求的最佳检查点")
    indexed = {row["name"]: row for row in rows}
    for case in cases:
        row = indexed[case["name"]]
        for key in ("text", "text_type", "target_audio_sha256", "evaluation_scope"):
            if row.get(key) != case[key]:
                raise ValueError(f"合成报告与输入不对应：{case['name']} {key}")
        if row.get("reference") != case["reference_audio"] or row.get("finite") is not True:
            raise ValueError("参考音频不符或音频非有限")
        if not all(math.isfinite(float(row[key])) for key in ("seconds", "rms", "peak")):
            raise ValueError("报告含非有限音频指标")
        if row["seconds"] < 0.2 or row["rms"] < 1e-5 or row["peak"] > 1.0001 or row["sample_rate"] != 22050:
            raise ValueError("音频长度、音量、峰值或采样率未通过")


def validate_package(root: Path) -> dict:
    """解包后重新核对实际文件，不仅相信远端发布的完成标记。"""
    manifest = json.loads((root / "delivery_manifest.json").read_text(encoding="utf-8"))
    if not delivery_ready({"stage": "complete", **manifest}):
        raise ValueError("交付清单未通过训练和成对验收门禁")
    entries = manifest["files"]
    if len({row["path"] for row in entries}) != len(entries):
        raise ValueError("交付清单路径重复")
    for row in entries:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
            raise ValueError("交付清单含越界路径")
        path = root / row["path"]
        path.resolve().relative_to(root.resolve())
        if path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"交付文件大小或摘要不符：{row['path']}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    expected_files = {row["path"] for row in entries} | {"delivery_manifest.json"}
    # 下载回执由本机在首次验证后追加；它不是归档源文件，不影响随后再次核验模型。
    if actual not in (expected_files, expected_files | {"delivery_receipt.json"}):
        raise ValueError("实际交付文件集合与清单不一致")
    for required in ("adapter/adapter_model.safetensors", "extra_state.pt", "sampling_audit.json",
                     "heldout_audio/cases.json", "heldout_audio/report.json"):
        if required not in {row["path"] for row in entries}:
            raise ValueError(f"交付清单缺少必要文件：{required}")
    audit = json.loads((root / "sampling_audit.json").read_text(encoding="utf-8"))
    if (audit["optimizer_step"] != 15000 or audit["sampler_cursor"] != 120000
            or audit["completed_epochs"] != 2 or audit["all_text_views_seen"] is not True
            or audit["unique_text_views_seen"] != 74741 or audit["matched_last_sample_steps"] != 15000
            or audit.get("unseen_text_views") != 0 or audit.get("data_fingerprint_matches") is not True):
        raise ValueError("最终实际采样审计未通过")
    cases = json.loads((root / "heldout_audio/cases.json").read_text(encoding="utf-8"))
    report = json.loads((root / "heldout_audio/report.json").read_text(encoding="utf-8"))
    validate_report(cases, report)
    for index, case in enumerate(cases):
        audio = root / "heldout_audio" / f"{index:02d}_{case['name']}.wav"
        audio.resolve().relative_to((root / "heldout_audio").resolve())
        with wave.open(str(audio), "rb") as source:
            if source.getframerate() != 22050 or source.getnframes() < 4410 or source.getsampwidth() != 2:
                raise ValueError("实际 WAV 头与合成报告不符")
    return manifest
