"""为未见文本验收选择同一录音对应的汉字／带调台罗，避免只覆盖第一种视图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def select_paired_cases(rows: list[dict], utterances: int = 4, seed: int = 20261105, split: str = "test") -> list[dict]:
    """固定选择指定验收 split 的成对例句；训练期试听只允许 valid，不混入最终 test。"""
    if split not in ("test", "valid"):
        raise ValueError("成对验收只允许 test 或 valid，禁止从 train 伪造泛化样例")
    if utterances < 1:
        raise ValueError("验收录音数必须大于零")
    groups = {}
    for row in rows:
        if row.get("split") != split or row.get("source") != "leku" or not 2 <= float(row["duration"]) <= 8:
            continue
        kind = row.get("text_type")
        if kind not in ("hanzi", "tailo"):
            continue
        views = groups.setdefault(row["audio_sha256"], {})
        if kind in views and views[kind]["text"] != row["text"]:
            raise ValueError("同一录音、同一书写视图存在冲突，不能任意选择验收文本")
        views[kind] = row
    eligible = {digest: views for digest, views in groups.items() if set(views) == {"hanzi", "tailo"}}
    order = sorted(eligible, key=lambda digest: hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest())
    if len(order) < max(utterances, 2):
        raise ValueError(f"{split} 中没有足够的成对例句及不同参考录音")
    cases = []
    for index, digest in enumerate(order[:utterances]):
        target = eligible[digest]["hanzi"]
        candidates = [eligible[key]["hanzi"] for key in order if key != digest
                      and eligible[key]["hanzi"]["text"] != target["text"]
                      and (not target.get("word_id") or eligible[key]["hanzi"].get("word_id") != target["word_id"])]
        if not candidates:
            raise ValueError(f"找不到不同音频、不同文本及不同词目的 {split} 参考")
        same_speaker = [row for row in candidates if target.get("speaker_id") is not None
                        and row.get("speaker_id") == target["speaker_id"]]
        reference = (same_speaker or candidates)[0]
        for kind in ("hanzi", "tailo"):
            row = eligible[digest][kind]
            cases.append({"name": f"heldout_{index}_{kind}", "text": row["text"], "text_type": kind,
                          "target_audio": row["audio"], "target_audio_sha256": digest,
                          "reference_audio": reference["audio"], "reference_audio_sha256": reference["audio_sha256"],
                          "split": split, "evaluation_scope": "heldout_text_paired_views" if split == "test" else "validation_preview_paired_views",
                          "reference_same_known_speaker": bool(same_speaker) if target.get("speaker_id") is not None else None})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("training/data/minnan/test.cloud.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training/data/minnan/heldout_paired_cases.json"))
    parser.add_argument("--utterances", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = select_paired_cases(rows, args.utterances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "types": sorted({row["text_type"] for row in cases}),
                      "output": args.output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
