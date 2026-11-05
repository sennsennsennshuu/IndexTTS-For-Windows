"""校验并复用 tmp 盘已有的同源音频，避免重复上传与重复占用磁盘。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path.cwd().resolve()
    if root != Path("/root/autodl-tmp/IndexTTS25_Minnan"):
        raise RuntimeError("音频映射只允许在指定 tmp 训练项目执行")
    data = Path("training/data/minnan")
    manifests = {}
    unique = {}
    for split in ("train", "valid", "test"):
        rows = [json.loads(line) for line in (data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        manifests[split] = rows
        for row in rows:
            unique.setdefault(row["audio_sha256"], row)
    old = Path("../IndexTTS/data")
    sutiau = old / "full_label/sutiau/wavs"
    leku = old / "full_label/leku/wavs"
    cv_root = old / "weak_label/cv-corpus-27.0-2026-09-11"
    provenance = cv_root / "preparation_20261015/conversion_manifest.json"
    cv_conversions = {}
    if provenance.exists():
        for row in json.loads(provenance.read_text(encoding="utf-8")):
            cv_conversions[(row["source_name"], row["source_sha256"])] = row
    cv_files = {p.name: p for p in cv_root.rglob("*.wav") if p.is_file()}

    def resolve(item):
        source_hash, row = item
        # 源清单来自 Windows，先规范分隔符，禁止把本地绝对路径照搬到 Linux。
        name = row["audio"].replace("\\", "/").rsplit("/", 1)[-1]
        uploaded = Path("training/audio") / (source_hash + Path(name).suffix.lower())
        if uploaded.is_file() and file_hash(uploaded) == source_hash:
            return source_hash, {"audio": uploaded.as_posix(), "cloud_audio_sha256": source_hash, "reuse": "uploaded_original"}
        candidate = sutiau / name
        if candidate.is_file() and file_hash(candidate) == source_hash:
            return source_hash, {"audio": candidate.as_posix(), "cloud_audio_sha256": source_hash, "reuse": "identical_original"}
        candidate = leku / name
        if candidate.is_file() and file_hash(candidate) == source_hash:
            return source_hash, {"audio": candidate.as_posix(), "cloud_audio_sha256": source_hash, "reuse": "identical_original"}
        conversion = cv_conversions.get((name, source_hash))
        if conversion:
            candidate = cv_files.get(conversion["wav_name"])
            if candidate and file_hash(candidate) == conversion["wav_sha256"]:
                return source_hash, {"audio": candidate.as_posix(), "cloud_audio_sha256": conversion["wav_sha256"],
                                     "reuse": "source_and_conversion_hash_verified",
                                     "conversion_max_error": conversion.get("roundtrip_max_error")}
        return source_hash, {"audio": row["audio"], "reuse": "missing", "source_audio": row["audio"]}

    with ThreadPoolExecutor(max_workers=8) as executor:
        mapping = dict(executor.map(resolve, unique.items()))
    missing = [{"audio_sha256": key, **value} for key, value in mapping.items() if value["reuse"] == "missing"]
    report = {"unique_audio": len(unique), "matched": len(unique) - len(missing), "missing": missing,
              "reuse_counts": {kind: sum(item["reuse"] == kind for item in mapping.values()) for kind in sorted({item["reuse"] for item in mapping.values()})}}
    Path("logs/data_mapping.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 只有全部样本完成校验才提交云端清单，不能把部分数据默认为全量训练集。
    if missing:
        print(json.dumps({"matched": report["matched"], "missing": len(missing)}, ensure_ascii=False), flush=True)
        raise SystemExit(3)
    for split, rows in manifests.items():
        destination = data / f"{split}.cloud.jsonl"
        temporary = destination.with_suffix(".jsonl.part")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                row = dict(row)
                row["source_audio"] = row["audio"]
                row.update(mapping[row["audio_sha256"]])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
    print(json.dumps({"matched": len(unique), "missing": 0, "reuse_counts": report["reuse_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
