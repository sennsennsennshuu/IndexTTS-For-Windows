"""构建 IndexTTS 2.5 台语 LoRA 的可审计训练清单。

设计边界（务必保持）：
1. 原始音频、source 和 dictionaries 全部只读；本脚本只写 output 目录中的 JSONL/JSON。
2. 音频时长不裁剪，只接受 0.6~18.0 秒；越界进入 review。
3. CV 只用 validated.tsv 的汉字视图，去掉括号内罗马字，不把 CV 的括号内容当带调监督。
4. sutiau/leku 的文本以 source/kautian.ods 原始 content.xml 为权威：
   - sutiau: 詞目['羅馬字音檔檔名'] 与 WAV stem 精确匹配；
   - leku:   例句['音檔檔名'] 与 WAV stem 精确匹配，并核对詞目id/例句順序。
   原始带调罗马字原样保留；斜线、括号可选片段等歧义进入 review，不猜。
5. 不使用 DiffSinger 的 928 音节/57 音素字典作为音频准入条件。
   字典只做规范/歧义审计，不删除来源合法但当前歌声字典未覆盖的音节。
6. CV 按 client_id 稳定哈希分 split；sutiau/leku 按 word_id 稳定哈希。
   再按规范汉字、speaker、word、view_group、audio_sha 做连通分量隔离，
   任何连到 valid/test 的组件都不回 train；held-out speaker 不会被移回 train。
7. 不分配 reference。features 由训练侧按 audio_sha256 提取。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
NS = {"t": TABLE_NS, "p": TEXT_NS}
CJK_RE = re.compile(r"[\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U000323AF]")
PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
EDITORIAL_RE = re.compile(r"【[^】]*】")
TAILO_PUNCT_RE = re.compile(r"[，,。.!！？?；;：:─―—“”「」『』‘’…]+")
SPACE_RE = re.compile(r"\s+")
SLASHES = ("/", "／")
OPTIONAL_PAREN = ("(", ")", "（", "）")
VALID_TEXT_TYPES = {"hanzi", "tailo"}
SPLITS = ("train", "valid", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_bucket(value: str, salt: str) -> int:
    return int(sha256_text(f"{salt}\0{value}")[:16], 16) % 100


def initial_split(value: str, salt: str) -> str:
    bucket = stable_bucket(value, salt)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "valid"
    return "test"


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value or "").strip()


def clean_hanzi(value: str) -> str:
    return SPACE_RE.sub(" ", nfc(EDITORIAL_RE.sub("", value))).strip()


def canonical_hanzi(value: str) -> str:
    normalized = nfc(EDITORIAL_RE.sub("", value))
    hanzi = "".join(CJK_RE.findall(normalized))
    if hanzi:
        return hanzi
    # 辞典「漢字」字段中也有尚无通用汉字写法的台罗借词，不能全部映射为空串。
    # 保留 Unicode 声调，仅统一大小写与词间分隔；加前缀避免与汉字键混淆。
    romanized = "".join(normalized.casefold().replace("-", "").split())
    return "__romanized__:" + romanized if romanized else ""


def normalize_tailo(raw: str) -> str:
    value = nfc(EDITORIAL_RE.sub("", raw))
    value = TAILO_PUNCT_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip()


def has_ambiguous_reading(raw: str) -> bool:
    return any(token in raw for token in SLASHES + OPTIONAL_PAREN + ("|", "｜"))


def parse_cv_hanzi(sentence: str) -> tuple[str | None, str]:
    raw = nfc(sentence)
    if raw.count("（") != raw.count("）") or raw.count("(") != raw.count(")"):
        return None, "unbalanced_parentheses"
    text = clean_hanzi(PAREN_RE.sub("", raw))
    if not text or not CJK_RE.search(text):
        return None, "no_hanzi"
    return text, ""


def read_ods_content_xml(path: Path) -> dict[str, list[list[str]]]:
    """只解析 ODS content.xml；复现重复单元格/行语义，不执行原脚本。"""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    sheets: dict[str, list[list[str]]] = {}
    for table in root.findall(".//t:table", NS):
        name = table.get(f"{{{TABLE_NS}}}name", "")
        rows: list[list[str]] = []
        for row in table.findall("t:table-row", NS):
            values: list[str] = []
            for cell in row:
                if cell.tag not in (f"{{{TABLE_NS}}}table-cell", f"{{{TABLE_NS}}}covered-table-cell"):
                    continue
                paragraphs = cell.findall("p:p", NS)
                value = "\n".join("".join(p.itertext()) for p in paragraphs)
                if not value:
                    value = cell.get(f"{{{OFFICE_NS}}}value", "")
                repeat = int(cell.get(f"{{{TABLE_NS}}}number-columns-repeated", "1"))
                values.extend([value] * min(repeat, 64))
            while values and not values[-1]:
                values.pop()
            if not values:
                continue
            row_repeat = int(row.get(f"{{{TABLE_NS}}}number-rows-repeated", "1"))
            if row_repeat > 10000:
                raise ValueError("ODS 非空行重复次数异常")
            rows.extend([values] * row_repeat)
        sheets[name] = rows
    return sheets


def rows_as_dicts(sheet: Sequence[Sequence[str]]) -> Iterator[tuple[int, dict[str, str]]]:
    if not sheet:
        return
    header = [str(value).strip() for value in sheet[0]]
    for row_number, values in enumerate(sheet[1:], 2):
        row = {header[index]: (str(values[index]) if index < len(values) else "") for index in range(len(header))}
        row["_ods_row"] = str(row_number)
        yield row_number, row


def build_ods_index(rows: Iterable[Mapping[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field, "")).strip()
        if key:
            index[key].append(dict(row))
    return dict(index)


def relative_source_path(path: Path, source_root: Path) -> str:
    return path.resolve().relative_to(source_root.resolve()).as_posix()


def audio_info(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(str(path))
        return {"duration": float(info.duration), "sample_rate": int(info.samplerate), "channels": int(info.channels), "frames": int(info.frames)}
    except Exception as soundfile_error:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(f"无法读取音频元数据: {path}: {soundfile_error}") from soundfile_error
        import wave
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if frames <= 0 or rate <= 0 or handle.getcomptype() != "NONE":
                raise ValueError(f"空音频或非 PCM WAV: {path}")
            return {"duration": frames / rate, "sample_rate": rate, "channels": handle.getnchannels(), "frames": frames}


def duration_allowed(duration: float) -> bool:
    return math.isfinite(duration) and 0.6 <= duration <= 18.0


def make_review(*, source: str, audio: str | None, audio_sha256: str | None, duration: float | None, hanzi: str | None, tailo_tone: str | None, reason: str, raw_fields: Mapping[str, Any] | None = None, sample_id: str | None = None, split: str | None = None, mapping_assumption: str = "", ods_file: str | None = None, ods_sheet: str | None = None, ods_row: int | None = None) -> dict[str, Any]:
    return {"sample_id": sample_id, "audio": audio, "audio_sha256": audio_sha256, "duration": duration, "text": None, "text_type": None, "source": source, "speaker_id": None, "split": split, "hanzi": hanzi, "tailo_tone": tailo_tone, "reject_reason": reason, "mapping_assumption": mapping_assumption, "ods_file": ods_file, "ods_sheet": ods_sheet, "ods_row": ods_row, "raw_fields": dict(raw_fields or {})}


def make_row(*, source: str, audio: str, audio_sha256: str, duration: float, text: str, text_type: str, speaker_id: str | None, split: str, sample_id: str, hanzi: str, tailo_tone: str | None, view_group: str, mapping_assumption: str, license_name: str, raw_fields: Mapping[str, Any], ods_file: str | None = None, ods_sheet: str | None = None, ods_row: int | None = None, word_id: str | None = None, audio_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if text_type not in VALID_TEXT_TYPES:
        raise ValueError(f"非法 text_type: {text_type}")
    return {"sample_id": sample_id, "audio": audio, "audio_sha256": audio_sha256, "text": text, "text_type": text_type, "source": source, "speaker_id": speaker_id, "duration": round(float(duration), 6), "split": split, "hanzi": hanzi, "tailo_tone": tailo_tone, "view_group": view_group, "source_group": source, "mapping_assumption": mapping_assumption, "license": license_name, "word_id": word_id, "ods_file": ods_file, "ods_sheet": ods_sheet, "ods_row": ods_row, "audio_meta": dict(audio_meta or {}), "raw_fields": dict(raw_fields)}
@dataclass
class RowSet:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)
    input_counts: Counter = field(default_factory=Counter)
    duration_values: list[float] = field(default_factory=list)


def add_cv_rows(*, source_root: Path, cv_root: Path, duration_map: Mapping[str, float], split_salt: str, result: RowSet) -> None:
    validated = cv_root / "validated.tsv"
    if not validated.is_file():
        raise FileNotFoundError(validated)
    with validated.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            result.input_counts["cv_rows"] += 1
            path_name = str(row.get("path", "")).strip()
            sentence = str(row.get("sentence", ""))
            speaker = str(row.get("client_id", "")).strip() or None
            audio_path = cv_root / "clips" / path_name
            audio_rel = relative_source_path(audio_path, source_root) if audio_path.exists() else f"Common Voice Scripted Speech 27.0 - Taiwanese (Minnan)/cv-corpus-27.0-2026-09-11/nan-tw/clips/{path_name}"
            if not path_name:
                result.review.append(make_review(source="cv", audio=None, audio_sha256=None, duration=None, hanzi=None, tailo_tone=None, reason="missing_audio_path", raw_fields=row, sample_id=f"cv:{len(result.input_counts)}"))
                continue
            if not speaker:
                result.review.append(make_review(source="cv", audio=audio_rel, audio_sha256=None, duration=None, hanzi=None, tailo_tone=None, reason="missing_speaker", raw_fields=row, sample_id=f"cv:{Path(path_name).stem}"))
                continue
            hanzi, reason = parse_cv_hanzi(sentence)
            if reason:
                result.review.append(make_review(source="cv", audio=audio_rel, audio_sha256=None, duration=duration_map.get(path_name), hanzi=hanzi, tailo_tone=None, reason=reason, raw_fields=row, sample_id=f"cv:{Path(path_name).stem}"))
                continue
            duration = duration_map.get(path_name)
            if duration is None:
                result.review.append(make_review(source="cv", audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi, tailo_tone=None, reason="missing_duration", raw_fields=row, sample_id=f"cv:{Path(path_name).stem}"))
                continue
            if not duration_allowed(float(duration)):
                result.review.append(make_review(source="cv", audio=audio_rel, audio_sha256=None, duration=float(duration), hanzi=hanzi, tailo_tone=None, reason="duration_out_of_range", raw_fields=row, sample_id=f"cv:{Path(path_name).stem}"))
                continue
            result.accepted.append({"source": "cv", "audio_abs": audio_path, "audio": audio_rel, "duration": float(duration), "hanzi": hanzi, "tailo_tone": None, "speaker_id": speaker, "word_id": None, "view_group": f"cv:{Path(path_name).stem}", "mapping_assumption": "validated_hanzi_only", "license": "CC0-1.0", "raw_fields": dict(row), "ods_file": None, "ods_sheet": None, "ods_row": None, "audio_meta": {}, "split_seed_key": f"cv-speaker:{speaker}", "text_views": ("hanzi",)})
            result.duration_values.append(float(duration))


def add_ods_rows(*, source_root: Path, source_name: str, audio_root: Path, rows: Sequence[Mapping[str, str]], audio_field: str, ods_rel: str, sheet_name: str, split_salt: str, result: RowSet, require_example_order: bool) -> None:
    """按 ODS 完整音檔字段精确连接 sutiau 或 leku。"""
    index = build_ods_index(rows, audio_field)
    files = sorted(path for path in audio_root.rglob("*.wav") if path.is_file()) if audio_root.is_dir() else []
    stems: Counter = Counter(path.stem for path in files)
    for path in files:
        result.input_counts[f"{source_name}_audio"] += 1
        audio_rel = relative_source_path(path, source_root)
        stem = path.stem
        if stems[stem] > 1:
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=None, tailo_tone=None, reason="duplicate_audio_stem", sample_id=f"{source_name}:{stem}"))
            continue
        candidates = index.get(stem, [])
        if not candidates:
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=None, tailo_tone=None, reason="missing_ods_audio_field", sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name))
            continue
        if len(candidates) != 1:
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=None, tailo_tone=None, reason="ambiguous_ods_audio_field", sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name))
            continue
        row = candidates[0]
        ods_row = int(row.get("_ods_row", "0") or 0)
        hanzi = clean_hanzi(str(row.get("漢字", "")))
        raw_tailo = str(row.get("羅馬字", ""))
        tailo = normalize_tailo(raw_tailo)
        word_id = str(row.get("詞目id", "")).strip() or None
        if source_name == "leku":
            parts = stem.split("-")
            if not re.fullmatch(r"\d+-\d+-\d+", stem):
                result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi or None, tailo_tone=tailo or None, reason="unexpected_audio_name", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
                continue
            if str(row.get("詞目id", "")).strip() != parts[0] or (require_example_order and str(row.get("例句順序", "")).strip() != parts[2]):
                result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi or None, tailo_tone=tailo or None, reason="ods_filename_id_or_order_conflict", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
                continue
        else:
            match = re.fullmatch(r"(\d+)\((\d+)\)", stem)
            if not match:
                result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi or None, tailo_tone=tailo or None, reason="unexpected_audio_name", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
                continue
            if str(row.get("詞目id", "")).strip() != match.group(1):
                result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi or None, tailo_tone=tailo or None, reason="ods_filename_id_conflict", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
                continue
        if not hanzi:
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=None, tailo_tone=tailo or None, reason="empty_hanzi", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
            continue
        if not tailo:
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi, tailo_tone=None, reason="empty_tailo", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
            continue
        if has_ambiguous_reading(raw_tailo):
            result.review.append(make_review(source=source_name, audio=audio_rel, audio_sha256=None, duration=None, hanzi=hanzi, tailo_tone=tailo, reason="ambiguous_romanization", raw_fields=row, sample_id=f"{source_name}:{stem}", ods_file=ods_rel, ods_sheet=sheet_name, ods_row=ods_row))
            continue
        result.accepted.append({"source": source_name, "audio_abs": path, "audio": audio_rel, "duration": None, "hanzi": hanzi, "tailo_tone": tailo, "speaker_id": None, "word_id": word_id, "view_group": f"{source_name}:{stem}", "mapping_assumption": "exact_ods_audio_filename", "license": "unknown", "raw_fields": dict(row), "ods_file": ods_rel, "ods_sheet": sheet_name, "ods_row": ods_row, "audio_meta": {}, "split_seed_key": f"word:{word_id or stem}", "text_views": ("hanzi", "tailo")})
def compute_audio_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    """只读原始音频：并发调用，不复制、不改写。"""
    path = Path(candidate["audio_abs"])
    if not path.is_file():
        candidate["reject_reason"] = "missing_audio_file"
        return candidate
    try:
        meta = audio_info(path)
        candidate["audio_meta"] = meta
        candidate["duration"] = float(meta["duration"])
        candidate["audio_sha256"] = sha256_file(path)
    except Exception as error:
        candidate["reject_reason"] = f"audio_read_error:{type(error).__name__}"
        candidate["audio_error"] = str(error)
    return candidate


def materialize_rows(candidate: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text_type in candidate["text_views"]:
        text = candidate["hanzi"] if text_type == "hanzi" else candidate["tailo_tone"]
        rows.append(make_row(source=candidate["source"], audio=candidate["audio"], audio_sha256=candidate["audio_sha256"], duration=candidate["duration"], text=text, text_type=text_type, speaker_id=candidate["speaker_id"], split=split, sample_id=f"{candidate['view_group']}:{text_type}", hanzi=candidate["hanzi"], tailo_tone=candidate["tailo_tone"], view_group=candidate["view_group"], mapping_assumption=candidate["mapping_assumption"], license_name=candidate["license"], raw_fields=candidate["raw_fields"], ods_file=candidate.get("ods_file"), ods_sheet=candidate.get("ods_sheet"), ods_row=candidate.get("ods_row"), word_id=candidate.get("word_id"), audio_meta=candidate.get("audio_meta")))
    return rows


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item
    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def apply_text_conflicts(candidates: list[dict[str, Any]], initial: Sequence[str]) -> tuple[dict[int, str], list[int], list[dict[str, Any]], dict[str, Any]]:
    """保留 speaker/word 初分，只把与 held-out 文本重叠的训练侧 clip 移入 review。

    不再用并查集把 speaker 和文本连成巨型组件。这样同一 speaker 的其他句子
    不会因为一个常用词出现在 test 而被整体搬到 test；只有真正冲突的句子进入
    review。valid/test 同时出现时保留 test，valid 侧进入 review。
    """
    if len(candidates) != len(initial):
        raise ValueError("initial split 数量与候选数不一致")
    text_indices: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        key = canonical_hanzi(candidate["hanzi"])
        if key:
            text_indices[key].append(index)
    keep = [True] * len(candidates)
    reviews: list[dict[str, Any]] = []
    bridge_groups = 0
    reviewed_train = 0
    reviewed_valid = 0
    for key, indices in text_indices.items():
        splits = {initial[index] for index in indices}
        heldout = splits & {"valid", "test"}
        if not heldout:
            continue
        bridge_groups += 1
        keep_split = "test" if "test" in splits else "valid"
        for index in indices:
            if initial[index] == keep_split:
                continue
            keep[index] = False
            candidate = candidates[index]
            if initial[index] == "train":
                reviewed_train += 1
            elif initial[index] == "valid":
                reviewed_valid += 1
            reviews.append(make_review(
                source=candidate["source"],
                audio=candidate["audio"],
                audio_sha256=candidate.get("audio_sha256"),
                duration=candidate.get("duration"),
                hanzi=candidate["hanzi"],
                tailo_tone=candidate["tailo_tone"],
                reason="text_overlap_with_heldout",
                raw_fields=candidate["raw_fields"],
                sample_id=f"{candidate['view_group']}:hanzi",
                split=initial[index],
                mapping_assumption=candidate["mapping_assumption"],
                ods_file=candidate.get("ods_file"),
                ods_sheet=candidate.get("ods_sheet"),
                ods_row=candidate.get("ods_row"),
            ))
    kept_indices = [index for index, value in enumerate(keep) if value]
    final = {index: initial[index] for index in kept_indices}
    stats = {
        "text_groups": len(text_indices),
        "bridge_text_groups": bridge_groups,
        "reviewed_train_conflicts": reviewed_train,
        "reviewed_valid_conflicts": reviewed_valid,
        "kept_candidates": len(kept_indices),
        "reviewed_conflict_candidates": len(reviews),
    }
    return final, kept_indices, reviews, stats


def assign_stable_splits_with_text_conflicts(candidates: list[dict[str, Any]], salt: str) -> tuple[dict[int, str], list[int], list[dict[str, Any]], dict[str, Any]]:
    """先按 speaker/word 稳定初分，再执行保守文本冲突过滤。"""
    initial = [initial_split(candidate["split_seed_key"], salt) for candidate in candidates]
    return apply_text_conflicts(candidates, initial)


def compute_split_isolation(split_rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """计算跨 split 的 speaker/word/text/audio 交集，并在写盘前强制为 0。"""
    def values(split: str, field: str) -> set[str]:
        result: set[str] = set()
        for row in split_rows.get(split, []):
            value = row.get(field)
            if value:
                result.add(str(value))
        return result

    def pairwise(field: str) -> dict[str, int]:
        sets = {split: values(split, field) for split in SPLITS}
        pairs = {}
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1:]:
                pairs[f"{left}__{right}"] = len(sets[left] & sets[right])
        return pairs

    speaker_pairs = pairwise("speaker_id")
    word_pairs = pairwise("word_id")
    audio_pairs = pairwise("audio_sha256")
    text_pairs: dict[str, int] = {}
    text_sets = {split: {canonical_hanzi(row.get("hanzi", "")) for row in split_rows.get(split, []) if canonical_hanzi(row.get("hanzi", ""))} for split in SPLITS}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            text_pairs[f"{left}__{right}"] = len(text_sets[left] & text_sets[right])
    overlap = {
        "speaker": sum(speaker_pairs.values()),
        "word": sum(word_pairs.values()),
        "text": sum(text_pairs.values()),
        "audio": sum(audio_pairs.values()),
    }
    if any(overlap.values()):
        raise ValueError("split 泄漏检查失败: " + json.dumps(overlap, ensure_ascii=False))
    return {
        "nonempty": {split: bool(split_rows.get(split)) for split in SPLITS},
        "speaker_intersection": overlap["speaker"],
        "word_intersection": overlap["word"],
        "text_intersection": overlap["text"],
        "audio_intersection": overlap["audio"],
        "speaker_pairs": speaker_pairs,
        "word_pairs": word_pairs,
        "text_pairs": text_pairs,
        "audio_pairs": audio_pairs,
    }


def deduplicate_candidates(candidates: list[dict[str, Any]], result: RowSet) -> list[dict[str, Any]]:
    """按音频字节 SHA 去重；同 view_group 的 hanzi/tailo 两视图保留。"""
    priority = {"sutiau": 0, "leku": 1, "cv": 2}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate["audio_sha256"]].append(candidate)
    kept: list[dict[str, Any]] = []
    duplicate_hashes = 0
    duplicate_candidates = 0
    for audio_sha, items in groups.items():
        by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_view[item["view_group"]].append(item)
        canonical_group = sorted(by_view, key=lambda group: (priority.get(by_view[group][0]["source"], 99), group))[0]
        if len(by_view) > 1:
            duplicate_hashes += 1
        for group, group_items in by_view.items():
            if group != canonical_group:
                duplicate_candidates += len(group_items)
                for item in group_items:
                    result.review.append(make_review(source=item["source"], audio=item["audio"], audio_sha256=audio_sha, duration=item["duration"], hanzi=item["hanzi"], tailo_tone=item["tailo_tone"], reason="duplicate_audio_sha256", raw_fields=item["raw_fields"], sample_id=f"{item['view_group']}:hanzi", mapping_assumption=item["mapping_assumption"], ods_file=item.get("ods_file"), ods_sheet=item.get("ods_sheet"), ods_row=item.get("ods_row")))
                continue
            seen_text: set[tuple[str, str]] = set()
            for item in sorted(group_items, key=lambda value: value["audio"]):
                marker = (item["hanzi"], item["tailo_tone"] or "")
                if marker in seen_text:
                    duplicate_candidates += 1
                    result.review.append(make_review(source=item["source"], audio=item["audio"], audio_sha256=audio_sha, duration=item["duration"], hanzi=item["hanzi"], tailo_tone=item["tailo_tone"], reason="duplicate_audio_content", raw_fields=item["raw_fields"], sample_id=f"{item['view_group']}:hanzi", mapping_assumption=item["mapping_assumption"], ods_file=item.get("ods_file"), ods_sheet=item.get("ods_sheet"), ods_row=item.get("ods_row")))
                    continue
                seen_text.add(marker)
                kept.append(item)
    result.input_counts["duplicate_audio_sha256"] = duplicate_hashes
    result.input_counts["duplicate_candidates"] = duplicate_candidates
    return kept


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
    temporary.replace(path)


def build_report(*, args: argparse.Namespace, result: RowSet, split_rows: Mapping[str, list[dict[str, Any]]], accepted_candidates: int, split_stats: Mapping[str, Any], ods_sha: str, cv_readme_sha: str | None, cv_validated_sha: str | None) -> dict[str, Any]:
    reasons = Counter(row.get("reject_reason", "unknown") for row in result.review)
    all_rows = [row for rows in split_rows.values() for row in rows]
    source_counts: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source"] for row in result.review} | {row["source"] for row in all_rows}):
        source_rows = [row for row in all_rows if row["source"] == source]
        source_review = [row for row in result.review if row["source"] == source]
        source_counts[source] = {"input_rows": result.input_counts.get(f"{source}_rows", 0) if source == "cv" else result.input_counts.get(f"{source}_audio", 0), "accepted_view_rows": len(source_rows), "review_rows": len(source_review), "unique_audio_sha256": len({row["audio_sha256"] for row in source_rows}), "split_counts": dict(Counter(row["split"] for row in source_rows)), "text_type_counts": dict(Counter(row["text_type"] for row in source_rows))}
    durations = result.duration_values
    return {"schema_version": "minnan-cloud-manifest-v1", "generated_at": utc_now(), "source_root": args.source_root, "dictionary_root": args.dictionary_root, "output_root": args.output, "ods_file": args.ods_file, "ods_sha256": ods_sha, "cv_readme_sha256": cv_readme_sha, "cv_validated_sha256": cv_validated_sha, "duration_seconds": {"min_allowed": 0.6, "max_allowed": 18.0, "observed_min": min(durations) if durations else None, "observed_max": max(durations) if durations else None}, "input_counts": dict(result.input_counts), "accepted_audio_candidates": accepted_candidates, "accepted_view_rows": len(all_rows), "review_rows": len(result.review), "split_counts": {split: len(rows) for split, rows in split_rows.items()}, "source_counts": source_counts, "reject_reason_counts": dict(sorted(reasons.items())), "deduplication": {"duplicate_audio_sha256": result.input_counts.get("duplicate_audio_sha256", 0), "duplicate_candidates": result.input_counts.get("duplicate_candidates", 0)}, "split_isolation": dict(split_stats), "mapping": {"cv": "validated.tsv sentence minus parenthesized romanization; text_type=hanzi", "sutiau": "kautian.ods 詞目['羅馬字音檔檔名'] exact == wav.stem; hanzi=漢字; tailo_tone=羅馬字", "leku": "kautian.ods 例句['音檔檔名'] exact == wav.stem; verify 詞目id/例句順序; hanzi=漢字; tailo_tone=羅馬字", "dictionary_used_as_gate": False, "dictionary_role": "normalization/ambiguity audit only; no 928-syllable/57-phoneme exclusion"}, "licenses": {"cv": "CC0-1.0", "sutiau": "unknown", "leku": "unknown", "annesia": "review_only"}, "notes": ["音频未复制、未裁剪；features 由训练侧按 audio_sha256 提取。", "CV 不生成 tailo 视图；带调监督只来自 ODS 精确配对。", "review 保留被排除项和原因，不丢失输入计数。"]}
def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    dictionary_root = Path(args.dictionary_root)
    ods_path = Path(args.ods_file)
    cv_root = Path(args.cv_root)
    output_root = Path(args.output)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if not ods_path.is_file():
        raise FileNotFoundError(ods_path)
    project_root = Path(args.output_root).resolve()
    if not output_root.resolve().is_relative_to(project_root):
        raise ValueError("output 必须位于 output-root 内")
    sheets = read_ods_content_xml(ods_path)
    if "詞目" not in sheets or "例句" not in sheets:
        raise ValueError("ODS 缺少詞目/例句工作表")
    result = RowSet()
    duration_map: dict[str, float] = {}
    clip_durations = cv_root / "clip_durations.tsv"
    if clip_durations.is_file():
        with clip_durations.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    duration_map[row["clip"]] = float(row["duration[ms]"]) / 1000.0
                except (KeyError, TypeError, ValueError):
                    continue
    add_cv_rows(source_root=source_root, cv_root=cv_root, duration_map=duration_map, split_salt=args.split_salt, result=result)
    sutiau_rows = [row for _, row in rows_as_dicts(sheets["詞目"])]
    leku_rows = [row for _, row in rows_as_dicts(sheets["例句"])]
    add_ods_rows(source_root=source_root, source_name="sutiau", audio_root=source_root / "sutiau-wav", rows=sutiau_rows, audio_field="羅馬字音檔檔名", ods_rel=args.ods_file, sheet_name="詞目", split_salt=args.split_salt, result=result, require_example_order=False)
    add_ods_rows(source_root=source_root, source_name="leku", audio_root=source_root / "leku-wav", rows=leku_rows, audio_field="音檔檔名", ods_rel=args.ods_file, sheet_name="例句", split_salt=args.split_salt, result=result, require_example_order=True)
    candidates = result.accepted
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(compute_audio_fields, candidate) for candidate in candidates]
        for future in as_completed(futures):
            candidate = future.result()
            if candidate.get("reject_reason"):
                result.review.append(make_review(source=candidate["source"], audio=candidate["audio"], audio_sha256=candidate.get("audio_sha256"), duration=candidate.get("duration"), hanzi=candidate["hanzi"], tailo_tone=candidate["tailo_tone"], reason=candidate["reject_reason"], raw_fields=candidate["raw_fields"], sample_id=f"{candidate['view_group']}:hanzi", mapping_assumption=candidate["mapping_assumption"], ods_file=candidate.get("ods_file"), ods_sheet=candidate.get("ods_sheet"), ods_row=candidate.get("ods_row")))
    good = [candidate for candidate in candidates if not candidate.get("reject_reason")]
    duration_good: list[dict[str, Any]] = []
    for candidate in good:
        if not duration_allowed(float(candidate["duration"])):
            result.review.append(make_review(source=candidate["source"], audio=candidate["audio"], audio_sha256=candidate["audio_sha256"], duration=candidate["duration"], hanzi=candidate["hanzi"], tailo_tone=candidate["tailo_tone"], reason="duration_out_of_range", raw_fields=candidate["raw_fields"], sample_id=f"{candidate['view_group']}:hanzi", mapping_assumption=candidate["mapping_assumption"], ods_file=candidate.get("ods_file"), ods_sheet=candidate.get("ods_sheet"), ods_row=candidate.get("ods_row")))
        else:
            duration_good.append(candidate)
            result.duration_values.append(float(candidate["duration"]))
    unique_candidates = deduplicate_candidates(duration_good, result)
    final_splits, kept_indices, conflict_reviews, split_stats = assign_stable_splits_with_text_conflicts(unique_candidates, args.split_salt)
    result.review.extend(conflict_reviews)
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for index in kept_indices:
        split = final_splits[index]
        split_rows[split].extend(materialize_rows(unique_candidates[index], split))
    for split in SPLITS:
        split_rows[split].sort(key=lambda row: row["sample_id"])
    if not args.allow_empty_splits and any(not split_rows[split] for split in SPLITS):
        raise ValueError("train/valid/test 存在空集；拒绝写出不完整 manifest: " + str({split: len(split_rows[split]) for split in SPLITS}))
    split_stats.update(compute_split_isolation(split_rows))
    result.review.sort(key=lambda row: (str(row.get("source")), str(row.get("sample_id")), str(row.get("reject_reason"))))
    for split in SPLITS:
        write_jsonl(output_root / f"{split}.jsonl", split_rows[split])
    write_jsonl(output_root / "review.jsonl", result.review)
    cv_readme = cv_root / "README.md"
    cv_validated = cv_root / "validated.tsv"
    report = build_report(args=args, result=result, split_rows=split_rows, accepted_candidates=len(unique_candidates), split_stats=split_stats, ods_sha=sha256_file(ods_path), cv_readme_sha=sha256_file(cv_readme) if cv_readme.is_file() else None, cv_validated_sha=sha256_file(cv_validated) if cv_validated.is_file() else None)
    (output_root / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 IndexTTS 2.5 台语 LoRA 可审计 manifest")
    parser.add_argument("--source-root", default="../../minan-tw/source", help="原始 source 相对路径")
    parser.add_argument("--dictionary-root", default="../../minan-tw/dictionaries", help="字典相对路径（只读审计用）")
    parser.add_argument("--ods-file", default="../../minan-tw/source/kautian.ods", help="教育部 ODS 相对路径")
    parser.add_argument("--cv-root", default="../../minan-tw/source/Common Voice Scripted Speech 27.0 - Taiwanese (Minnan)/cv-corpus-27.0-2026-09-11/nan-tw", help="CV nan-tw 目录相对路径")
    parser.add_argument("--output", default="training/data/minnan", help="manifest 输出目录")
    parser.add_argument("--output-root", default=".", help="输出边界根目录；默认当前项目")
    parser.add_argument("--split-salt", default="indextts25-minnan-v1", help="稳定 split 盐；变更会产生新 split")
    parser.add_argument("--workers", type=int, default=8, help="并发读取音频元数据/SHA 的线程数")
    parser.add_argument("--allow-empty-splits", action="store_true", help="仅测试小样本时允许空 split；正式全量构建不要使用")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({"accepted_view_rows": report["accepted_view_rows"], "review_rows": report["review_rows"], "split_counts": report["split_counts"], "reject_reason_counts": report["reject_reason_counts"], "output": report["output_root"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
