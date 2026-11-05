"""构建可审计、可重复、可断点的台湾闽南语离线词库。

设计边界：
1. 原始响应按字节保存到 models/extra_language/zh-nan-tw/words/sources/，
   网络中断时已完成的文件会保留，下一次运行直接复用。
2. entries.jsonl 每行严格保存九个字段；同一来源的多个台罗读音拆成多条，
   不擅自替用户选择一个读音。
3. 只有明确对译字段才进入 mandarin。长释义、例句、带“例：/表示/指”的
   说明不会被截断后冒充可直接对译。
4. THUOCL 是普通话词语候选，只生成 tailo 为空、verified=false 的记录，
   不参与台语汉字索引，也不参与可信读音判断。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORDS_DIR = PROJECT_ROOT / "models" / "extra_language" / "zh-nan-tw" / "words"
SOURCES_DIR = WORDS_DIR / "sources"
ENTRIES_PATH = WORDS_DIR / "entries.jsonl"
MANIFEST_PATH = WORDS_DIR / "manifest.json"
SQLITE_PATH = WORDS_DIR / "lexicon.sqlite3"
USER_AGENT = "IndexTTS-MinnanLexiconBuilder/1.0 (offline research cache)"
CHUNK_SIZE = 1024 * 1024
HTTP_TIMEOUT = 90

# 统一字段顺序便于逐行审核；不要随意调整，测试会检查键集合。
ENTRY_FIELDS = (
    "id", "hanzi", "tailo", "mandarin", "source", "source_id",
    "license", "kind", "verified",
)

CHHOE_BASE = "https://raw.githubusercontent.com/ChhoeTaigi/ChhoeTaigiDatabase/master/ChhoeTaigiDatabase"
CHHOE_README_URL = "https://raw.githubusercontent.com/ChhoeTaigi/ChhoeTaigiDatabase/master/README.md"
CHHOE_FILES: dict[str, dict[str, str]] = {
    "chhoe_itaigi": {
        "filename": "ChhoeTaigi_iTaigiHoataiTuichiautian.csv",
        "license": "CC0-1.0",
        "title": "iTaigi 華台對照典",
    },
    "chhoe_taihoa": {
        "filename": "ChhoeTaigi_TaihoaSoanntengTuichiautian.csv",
        "license": "CC-BY-SA-4.0",
        "title": "台文華文線頂對照典",
    },
    "chhoe_kauiokpoo": {
        "filename": "ChhoeTaigi_KauiokpooTaigiSutian.csv",
        "license": "CC-BY-ND-3.0-TW",
        "title": "教育部臺灣閩南語常用詞辭典（經 ChhoeTaigi 整理）",
    },
    "chhoe_pehoe": {
        "filename": "ChhoeTaigi_TaioanPehoeKichhooGiku.csv",
        "license": "CC-BY-SA-4.0",
        "title": "台灣白話基礎語句",
    },
}

THUOCL_TREE_URL = "https://api.github.com/repos/thunlp/THUOCL/git/trees/master?recursive=1"
THUOCL_RAW_BASE = "https://raw.githubusercontent.com/thunlp/THUOCL/master"
THUOCL_FILES = (
    "data/THUOCL_IT.txt",
    "data/THUOCL_animal.txt",
    "data/THUOCL_caijing.txt",
    "data/THUOCL_car.txt",
    "data/THUOCL_chengyu.txt",
    "data/THUOCL_diming.txt",
    "data/THUOCL_food.txt",
    "data/THUOCL_law.txt",
    "data/THUOCL_lishimingren.txt",
    "data/THUOCL_medical.txt",
    "data/THUOCL_poem.txt",
)

ICORPUS_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=Bohanlu%2FiCorpus-100&config=default&split=test&offset=0&length=100"
)
ICORPUS_CARD_URL = "https://huggingface.co/datasets/Bohanlu/iCorpus-100/raw/main/README.md"

WIKI_PAGE_URL = (
    "https://zh.wikiversity.org/w/index.php?title=Taiwan_Tongues/"
    "%E8%87%BA%E7%81%A3%E8%87%BA%E8%AA%9E%E9%96%8B%E6%BA%90%E8%AA%9E%E6%96%99%E5%BA%AB&action=raw"
)
TGGL_URL = "https://tggl.naer.edu.tw/practical_taiwanese_dictionaries"
TGGL_QUERY_URL = TGGL_URL + "?type=word&text=%E9%A3%9F&n1=100&n2=100&page1=1&page2=1"
PTS_BASE_URL = "https://taigi.acv-vision.com/taigi-words"
TAIWAN_LINGU_URL = "https://taiwan-lingu.ist/corpus/"

PTS_LICENSE = (
    "Public website; no explicit reuse license found "
    "(private-use only; obtain permission before redistribution)"
)
TGGL_LICENSE = (
    "Copyright held by original author; public web lookup only; "
    "not bulk-incorporated"
)
TAIWAN_LINGU_LICENSE = (
    "Site/login terms apply; corpus download restricted; "
    "audit sample only, not bulk-incorporated"
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
DEFINITION_STARTS = ("例：", "例:", "表示", "比喻", "指", "用於", "用于", "用來", "用来", "意思", "即")
SENTENCE_PUNCT_RE = re.compile(r"[。！？!?；;，,]")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_path(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def compact_space(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def strip_tags(value: str) -> str:
    return compact_space(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def extract_hanzi(value: str) -> str:
    """保留原始汉罗词目及标点；只有完全没有汉字时才返回空字符串。

    历史上的实现只 join 汉字，会造成 ``食 pháng`` 被压成 ``食``，但仍与完整
    读音关联，破坏词形与读音的一一对应。现在保留来源字段的汉罗混排和标点，
    再由独立的 ``tailo`` 字段承载台罗。
    """
    value = (value or "").replace("\u00a0", " ").strip()
    return value if CJK_RE.search(value) else ""


def split_tailo_readings(*values: str) -> list[str]:
    """拆分一个来源行中的多个读音；不解析、不重写声调。"""
    readings: list[str] = []
    for value in values:
        if not value:
            continue
        for part in re.split(r"\s*/\s*", compact_space(str(value))):
            part = part.strip(" ,，;；")
            if not part or not re.search(r"[A-Za-zÀ-ž]", part):
                continue
            if part not in readings:
                readings.append(part)
    return readings


def _candidate_allowed(candidate: str, max_length: int) -> bool:
    if not candidate or len(candidate) > max_length:
        return False
    if not CJK_RE.search(candidate):
        return False
    if re.search(r"[\n\r\t]", candidate):
        return False
    if any(marker in candidate for marker in DEFINITION_STARTS):
        return False
    if re.search(r"[()（）\[\]【】<>《》]", candidate):
        return False
    if re.search(r"[。！？!?：:]", candidate):
        return False
    if "----" in candidate or "…" in candidate:
        return False
    return True


def extract_mandarin_candidates(
    value: str,
    max_length: int = 24,
    *,
    preserve_whole: bool = False,
) -> list[str]:
    """按来源语义提取普通话对译，避免把句子拆成假同义词。

    ``preserve_whole=True`` 用于来源已明确声明整个字段是 HoaBun 的记录
    （台湾白话基础语句）。其他来源只在没有逗号/句号等句子标点时，才把
    ``、``/``；`` 视为明确的并列分隔符。带逗号的普通话保持为完整一项，
    避免多个片段各自错误关联同一条整词台罗。
    """
    text = compact_space(value or "")
    if not text or not CJK_RE.search(text):
        return []
    if "例：" in text or "例:" in text:
        return []
    if preserve_whole:
        return [text]
    # 定义性字段不进入直接对译；HoaBun 的正常句子没有这些前缀。
    if text.startswith(DEFINITION_STARTS):
        return []
    # 逗号/句号说明它是完整句子或谚语，不能拆成若干“同义词”。
    if re.search(r"[，,。]", text):
        if "。、" in text:  # 上游错误拼接的枚举串
            text = text.replace("。、", "、")
        else:
            return [text]
    pieces = re.split(r"\s*(?:、|；|;|/)\s*", text)
    if len(pieces) == 1:
        piece = pieces[0].strip(" .。·-—")
        return [piece] if _candidate_allowed(piece, max(max_length, len(piece))) else []
    # 只要有一个分片像完整句子，就保留整体；否则按明确枚举拆分。
    if any(len(piece) > max_length for piece in pieces):
        return [text]
    candidates: list[str] = []
    for piece in pieces:
        piece = piece.strip(" .。·-—")
        if _candidate_allowed(piece, max_length) and piece not in candidates:
            candidates.append(piece)
    return candidates


def classify_kind(hanzi: str, mandarin: Iterable[str]) -> str:
    # 同时检查汉罗词目和普通话字段：有时台语词目不带句号，而 HoaBun 是完整句。
    mandarin_text = " ".join(mandarin)
    joined = f"{hanzi} {mandarin_text}".strip()
    if SENTENCE_PUNCT_RE.search(joined) or len(CJK_RE.findall(hanzi or mandarin_text)) > 10:
        return "sentence"
    return "word"


def make_entry(
    *,
    source: str,
    source_id: str,
    hanzi: str,
    tailo: str,
    mandarin: list[str],
    license: str,
    kind: str | None = None,
    verified: bool,
) -> dict[str, Any]:
    # 稳定 ID 只依赖来源与来源内编号，重建时不会因文件顺序变化而漂移。
    digest = hashlib.sha256(f"{source}:{source_id}".encode("utf-8")).hexdigest()[:20]
    return {
        "id": f"{source}-{digest}",
        # 汉罗词目保留来源内部的空格与标点，不能与完整台罗错位。
        "hanzi": (hanzi or "").replace("\u00a0", " ").strip(),
        "tailo": compact_space(tailo),
        "mandarin": list(dict.fromkeys(item for item in mandarin if item)),
        "source": source,
        "source_id": str(source_id),
        "license": license,
        "kind": kind or classify_kind(hanzi, mandarin),
        "verified": bool(verified),
    }


class Fetcher:
    """项目内可断点下载器；绝不写入系统临时目录。"""

    def __init__(self, refresh: bool = False, offline: bool = False):
        self.refresh = refresh
        self.offline = offline
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.5",
        })
        self.records: list[dict[str, Any]] = []

    def fetch(self, url: str, relative_path: str, *, license: str, note: str = "") -> Path:
        destination = SOURCES_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        status = "cached"
        if self.refresh or not destination.is_file() or destination.stat().st_size == 0:
            if self.offline:
                raise FileNotFoundError(f"离线模式缺少原始文件：{relative_path}")
            status = self._download(url, destination)
        stat = destination.stat()
        fetched_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat()
        self.records.append({
            "url": url,
            "path": destination.relative_to(WORDS_DIR).as_posix(),
            "status": status,
            "fetched_at": fetched_at,
            "bytes": stat.st_size,
            "sha256": digest_path(destination),
            "license": license,
            "note": note,
        })
        return destination

    def _download(self, url: str, destination: Path) -> str:
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = self.session.get(url, timeout=HTTP_TIMEOUT, stream=True)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                temporary = destination.with_name(destination.name + ".part")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(CHUNK_SIZE):
                        if block:
                            stream.write(block)
                if temporary.stat().st_size == 0:
                    raise IOError("服务器返回空文件")
                temporary.replace(destination)
                return "downloaded"
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(attempt * 2)
        raise RuntimeError(f"下载失败：{url}：{last_error}")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_bytes().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def build_chhoe(fetcher: Fetcher) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetcher.fetch(CHHOE_README_URL, "chhoetaigi/README.md", license="Multiple; see per-file sections")
    entries: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for source, spec in CHHOE_FILES.items():
        path = fetcher.fetch(
            f"{CHHOE_BASE}/{spec['filename']}",
            f"chhoetaigi/{spec['filename']}",
            license=spec["license"],
            note=spec["title"],
        )
        rows = load_csv_rows(path)
        included = 0
        rejected = Counter()
        for row_number, row in enumerate(rows, start=2):
            source_id = (row.get("DictWordID") or f"line-{row_number}").strip()
            han_raw = row.get("HanLoTaibunKip") or row.get("HanLoTaibunPoj") or ""
            hanzi = extract_hanzi(han_raw)
            readings = split_tailo_readings(
                row.get("KipUnicode") or "", row.get("KipUnicodeOthers") or ""
            )
            # Kauiokpoo 的释义混在 Kaisoeh 字段；HoaBun 才是明确对译。
            # Pehoe 的 HoaBun 是完整普通话字段，必须保持单值，不能按逗号拆开。
            max_len = 16 if source == "chhoe_kauiokpoo" else 24
            mandarin = extract_mandarin_candidates(
                row.get("HoaBun") or "",
                max_length=max_len,
                preserve_whole=source == "chhoe_pehoe",
            )
            if not readings:
                rejected["no_tailo"] += 1
                continue
            if not hanzi and not mandarin:
                rejected["no_hanzi_or_mandarin"] += 1
                continue
            if not mandarin and not hanzi:
                rejected["no_indexable_value"] += 1
                continue
            for reading_index, reading in enumerate(readings, start=1):
                sid = source_id if len(readings) == 1 else f"{source_id}:reading-{reading_index}"
                entries.append(make_entry(
                    source=source,
                    source_id=sid,
                    hanzi=hanzi,
                    tailo=reading,
                    mandarin=mandarin,
                    license=spec["license"],
                    verified=True,
                ))
                included += 1
        details[source] = {
            "title": spec["title"],
            "input_rows": len(rows),
            "entries": included,
            "rejected": dict(rejected),
            "limitations": (
                "hanzi 保留 HanLoTaibunKip/HanLoTaibunPoj 的原始汉罗词目和标点；"
                "完全没有汉字时才为空。台罗读音来自 KipUnicode/KipUnicodeOthers，"
                "多个候选拆成多条；Pehoe HoaBun 始终保持为单个完整字段。"
            ),
        }
    return entries, details


def parse_pts_page(text: str, page_number: int) -> list[dict[str, Any]]:
    """解析公视台语新词辞库；页面字段为“词目 / 音读 / 对应华语”。"""
    entries: list[dict[str, Any]] = []
    chunks = re.split(r'(?=<div class="col-lg-4 mb-20">)', text)
    for chunk in chunks:
        if 'class="col-lg-4 mb-20"' not in chunk:
            continue
        title_match = re.search(
            r'<div class="h3">\s*<a href="([^"]+)" title="([^"]*)">(.*?)</a>',
            chunk, re.S,
        )
        if not title_match:
            continue
        detail_url = html.unescape(title_match.group(1))
        hanzi = strip_tags(title_match.group(3))
        if not hanzi or not CJK_RE.search(hanzi):
            continue
        reading_block_match = re.search(
            r"音讀：.*?<div class=\"col-8 pl-0\">(.*?)</div>\s*</div>",
            chunk, re.S,
        )
        mandarin_match = re.search(
            r"對應華語：.*?<div class=\"col-8 pl-0\">\s*<span class=\"\">(.*?)</span>",
            chunk, re.S,
        )
        if not reading_block_match or not mandarin_match:
            continue
        readings = [
            strip_tags(value)
            for value in re.findall(r'<span class="">(.*?)</span>', reading_block_match.group(1), re.S)
        ]
        mandarin = extract_mandarin_candidates(strip_tags(mandarin_match.group(1)), max_length=30)
        if not readings or not mandarin:
            continue
        item_match = re.search(r"/taigi-words/(\d+)", detail_url)
        item_id = item_match.group(1) if item_match else f"page-{page_number}"
        for reading_index, reading in enumerate(dict.fromkeys(readings), start=1):
            if not re.search(r"[A-Za-zÀ-ž]", reading):
                continue
            entries.append(make_entry(
                source="pts_taigi_words",
                source_id=f"{item_id}:reading-{reading_index}",
                hanzi=hanzi,
                tailo=reading,
                mandarin=mandarin,
                license=PTS_LICENSE,
                verified=True,
            ))
    return entries


def build_pts(fetcher: Fetcher) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_path = fetcher.fetch(
        PTS_BASE_URL + "?page=1",
        "pts_taigi_words/page-001.html",
        license=PTS_LICENSE,
        note="公视台语新词辞库，第 1 页；页面未发现明确再利用许可",
    )
    first_text = first_path.read_text(encoding="utf-8")
    page_numbers = [int(value) for value in re.findall(r"page=(\d+)", first_text)]
    max_page = max(page_numbers or [1])
    # 避免上游异常把分页写成超大数字时无限抓取。
    max_page = min(max_page, 200)
    entries = parse_pts_page(first_text, 1)
    page_counts = {1: len(entries)}
    for page in range(2, max_page + 1):
        path = fetcher.fetch(
            f"{PTS_BASE_URL}?page={page}",
            f"pts_taigi_words/page-{page:03d}.html",
            license=PTS_LICENSE,
            note=f"公视台语新词辞库，第 {page} 页",
        )
        page_entries = parse_pts_page(path.read_text(encoding="utf-8"), page)
        entries.extend(page_entries)
        page_counts[page] = len(page_entries)
    return entries, {
        "title": "公视台语新词辞库",
        "pages": max_page,
        "page_counts": page_counts,
        "entries": len(entries),
        "limitations": (
            "页面明确给出词目、音读、对应华语，但未找到可复用的开放许可；"
            "按用户私人使用范围收录，商业使用或再分发前必须取得授权。"
        ),
    }


def build_icorpus(fetcher: Fetcher) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_path = fetcher.fetch(
        ICORPUS_ROWS_URL,
        "taiwan_tongues/icorpus-100.datasets-server.json",
        license="CC-BY-NC-4.0",
        note="维基学院 Taiwan Tongues 页面链接的 iCorpus-100 句对",
    )
    fetcher.fetch(
        ICORPUS_CARD_URL,
        "taiwan_tongues/icorpus-100.README.md",
        license="CC-BY-NC-4.0",
        note="Hugging Face 数据集卡与许可说明",
    )
    fetcher.fetch(
        WIKI_PAGE_URL,
        "taiwan_tongues/wikiversity_page.wiki",
        license="CC-BY-SA-4.0 (page text); linked datasets have their own licenses",
        note="用户指定页面原始 wikitext，用于来源链审计",
    )
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    entries: list[dict[str, Any]] = []
    for item in rows:
        row = item.get("row", item)
        mandarin_text = compact_space(str(row.get("ZH") or ""))
        hanzi = compact_space(str(row.get("HAN") or ""))
        tailo = compact_space(str(row.get("TL") or ""))
        if not mandarin_text or not hanzi or not tailo:
            continue
        entries.append(make_entry(
            source="icorpus_100",
            source_id=str(row.get("ID", item.get("row_idx", len(entries)))),
            hanzi=hanzi,
            tailo=tailo,
            mandarin=[mandarin_text],
            license="CC-BY-NC-4.0",
            kind="sentence",
            verified=True,
        ))
    return entries, {
        "title": "iCorpus-100 台华平行句对",
        "input_rows": len(rows),
        "entries": len(entries),
        "limitations": (
            "100 条人工校正汉字的新闻句对；英译由 GPT 生成但本词库不使用英文。"
            "该数据为 CC BY-NC 4.0，不能据此宣称全部词库可商业使用。"
        ),
    }


def build_thuocl(fetcher: Fetcher) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetcher.fetch(
        THUOCL_RAW_BASE + "/LICENSE",
        "thuocl/LICENSE",
        license="MIT",
        note="THUOCL 仓库许可证",
    )
    fetcher.fetch(
        THUOCL_RAW_BASE + "/README.md",
        "thuocl/README.md",
        license="MIT",
        note="THUOCL 分类与数据说明",
    )
    entries: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for relative in THUOCL_FILES:
        path = fetcher.fetch(
            f"{THUOCL_RAW_BASE}/{relative}",
            "thuocl/" + relative.split("/", 1)[1],
            license="MIT",
            note="仅作为普通话词语候选，不提供台语读音或对译",
        )
        rows = 0
        included = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            word = re.split(r"\t+", line.strip(), maxsplit=1)[0].strip()
            if not word or not CJK_RE.search(word) or len(word) > 30:
                continue
            # 关键边界：THUOCL 不是台语词典。hanzi 必须为空，避免主程序
            # 把普通话词形当作台语汉字；tailo 为空且 verified=false。
            entries.append(make_entry(
                source="thuocl",
                source_id=f"{relative}:{line_number}",
                hanzi="",
                tailo="",
                mandarin=[word],
                license="MIT",
                kind=classify_kind("", [word]),
                verified=False,
            ))
            included += 1
        details[relative] = {"rows": rows, "entries": included}
    return entries, {
        "title": "THU Open Chinese Lexicon",
        "files": details,
        "entries": len(entries),
        "limitations": "只提供普通话词候选；不含台湾闽南语读音，不得用于判定台罗。",
    }


def build_audits(fetcher: Fetcher) -> dict[str, Any]:
    """保存用户指定站点的实际可访问证据；受限页面不批量镜像。"""
    tggl_path = fetcher.fetch(
        TGGL_URL,
        "audit/tggl_practical_taiwanese_dictionaries.html",
        license=TGGL_LICENSE,
        note="卢广诚《实用台语词典》公开查询页；页面说明版权归原作者",
    )
    tggl_query_path = fetcher.fetch(
        TGGL_QUERY_URL,
        "audit/tggl_query_food.html",
        license=TGGL_LICENSE,
        note="实测查询“食”返回完全符合 2 笔、部分符合 96 笔",
    )
    taiwan_lingu_path = fetcher.fetch(
        TAIWAN_LINGU_URL,
        "audit/taiwan_lingu_corpus.html",
        license=TAIWAN_LINGU_LICENSE,
        note="清大台湾语言多样性语料库入口；下载全部结果需站点流程/授权",
    )
    # 清大检索接口在匿名状态下对教育部辞典样本可用；保存响应用于审计，
    # 不把受限站点批量转成离线词库。
    session = fetcher.session
    landing = taiwan_lingu_path.read_text(encoding="utf-8")
    csrf_match = re.search(r"csrfmiddlewaretoken:\s*['\"]([^'\"]+)", landing)
    sample_record: dict[str, Any] = {"status": "unavailable", "reason": "csrf token not found"}
    cached_sample = SOURCES_DIR / "audit" / "taiwan_lingu_search_food.json"
    if fetcher.offline:
        if cached_sample.is_file():
            sample_payload = json.loads(cached_sample.read_text(encoding="utf-8"))
            sample_record = {
                "status": "ok_cached",
                "path": cached_sample.relative_to(WORDS_DIR).as_posix(),
                "sha256": digest_path(cached_sample),
                "bytes": cached_sample.stat().st_size,
                "corpus": sample_payload.get("corpus"),
                "rows": len(sample_payload.get("result", [])),
                "note": "离线复核使用项目内缓存的匿名检索样本，未重新联网。",
            }
        else:
            sample_record = {"status": "failed", "reason": "离线模式缺少已缓存检索样本"}
    elif csrf_match:
        body = {
            "nameid": "96864fbd458f475b2aa7caf799fe9bcdde2eab6e24f51e6253c6d87d209a0206",
            "title_list": ["詞目"],
            "type": "lexicon",
            "text": "食",
        }
        try:
            response = session.post(
                "https://taiwan-lingu.ist/corpus/search/",
                json=body,
                headers={
                    "X-CSRFToken": csrf_match.group(1),
                    "Referer": TAIWAN_LINGU_URL,
                },
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            sample_path = SOURCES_DIR / "audit" / "taiwan_lingu_search_food.json"
            atomic_write_bytes(sample_path, response.content)
            sample_payload = response.json()
            sample_record = {
                "status": "ok",
                "path": sample_path.relative_to(WORDS_DIR).as_posix(),
                "sha256": digest_path(sample_path),
                "bytes": sample_path.stat().st_size,
                "corpus": sample_payload.get("corpus"),
                "rows": len(sample_payload.get("result", [])),
                "note": "匿名检索样本；未批量下载，未将站点内容作为独立许可来源。",
            }
        except Exception as exc:  # 审计失败必须记录，不伪造数据。
            sample_record = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "tggl": {
            "url": TGGL_URL,
            "status": "accessible_public_query_only",
            "query_probe": "完全符合 2 笔；部分符合 96 笔",
            "license": TGGL_LICENSE,
            "limitation": "页面说明非专属授权、可在网站公开查询，版权仍为原作者所有；未批量复制。",
            "raw": tggl_path.relative_to(WORDS_DIR).as_posix(),
            "query_raw": tggl_query_path.relative_to(WORDS_DIR).as_posix(),
        },
        "taiwan_lingu": {
            "url": TAIWAN_LINGU_URL,
            "status": "accessible_but_download_restricted",
            "license": TAIWAN_LINGU_LICENSE,
            "limitation": "语料库整合教育部辞典等来源；“下载全部结果”依赖站点流程，未批量镜像。",
            "raw": taiwan_lingu_path.relative_to(WORDS_DIR).as_posix(),
            "anonymous_search_sample": sample_record,
        },
    }


def deduplicate(entries: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    conflicts = 0
    for entry in entries:
        existing = unique.get(entry["id"])
        if existing is None:
            unique[entry["id"]] = entry
            continue
        duplicates += 1
        if any(existing[field] != entry[field] for field in ENTRY_FIELDS):
            conflicts += 1
    ordered = sorted(unique.values(), key=lambda item: (item["source"], item["source_id"], item["id"]))
    return ordered, {"duplicate_ids": duplicates, "conflicting_duplicate_ids": conflicts}


def write_entries(entries: list[dict[str, Any]]) -> str:
    lines = [json.dumps(entry, ensure_ascii=False, separators=(",", ":")) for entry in entries]
    atomic_write_text(ENTRIES_PATH, "\n".join(lines) + ("\n" if lines else ""))
    return digest_path(ENTRIES_PATH)


def write_sqlite(entries: list[dict[str, Any]]) -> dict[str, Any]:
    temporary = SQLITE_PATH.with_name(SQLITE_PATH.name + ".part")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE entries (
                id TEXT PRIMARY KEY,
                hanzi TEXT NOT NULL,
                tailo TEXT NOT NULL,
                mandarin_json TEXT NOT NULL,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                license TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('word','sentence')),
                verified INTEGER NOT NULL CHECK(verified IN (0,1))
            );
            CREATE INDEX idx_entries_hanzi ON entries(hanzi);
            CREATE INDEX idx_entries_source ON entries(source, source_id);
            """
        )
        connection.executemany(
            "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["id"], row["hanzi"], row["tailo"],
                    json.dumps(row["mandarin"], ensure_ascii=False),
                    row["source"], row["source_id"], row["license"],
                    row["kind"], int(row["verified"]),
                )
                for row in entries
            ],
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(SQLITE_PATH)
    with sqlite3.connect(SQLITE_PATH) as check:
        count = check.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    return {"path": SQLITE_PATH.relative_to(WORDS_DIR).as_posix(), "bytes": SQLITE_PATH.stat().st_size,
            "sha256": digest_path(SQLITE_PATH), "rows": count}


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(row["source"] for row in entries)
    license_counts = Counter(row["license"] for row in entries)
    return {
        "entries": len(entries),
        "word": sum(row["kind"] == "word" for row in entries),
        "sentence": sum(row["kind"] == "sentence" for row in entries),
        "with_hanzi": sum(bool(row["hanzi"]) for row in entries),
        "with_tailo": sum(bool(row["tailo"]) for row in entries),
        "with_mandarin": sum(bool(row["mandarin"]) for row in entries),
        "verified": sum(bool(row["verified"]) for row in entries),
        "by_source": dict(sorted(source_counts.items())),
        "by_license": dict(sorted(license_counts.items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    WORDS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(refresh=args.refresh, offline=args.offline)
    started_at = utc_now()
    chhoe_entries, chhoe_details = build_chhoe(fetcher)
    pts_entries, pts_details = build_pts(fetcher)
    icorpus_entries, icorpus_details = build_icorpus(fetcher)
    thuocl_entries, thuocl_details = build_thuocl(fetcher)
    audits = build_audits(fetcher)
    entries, dedup = deduplicate(
        [*chhoe_entries, *pts_entries, *icorpus_entries, *thuocl_entries]
    )
    entries_sha = write_entries(entries)
    sqlite_record = write_sqlite(entries) if not args.skip_sqlite else None
    summary = summarize(entries)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "started_at": started_at,
        "builder": "tools/build_minnan_words.py",
        "entry_schema": list(ENTRY_FIELDS),
        "field_semantics": {
            "hanzi": "台语词目；保留汉罗混排来源的原始汉字、拉丁拼写和标点；仅完全无汉字时为空，THUOCL 固定为空",
            "tailo": "带调台罗；多个候选按来源行拆分，不由程序替用户消歧",
            "mandarin": "明确对译候选；不是整段释义，长释义/例句已过滤",
            "verified": "来源字段对齐且可用于唯一读音判断；THUOCL 恒为 false",
        },
        "outputs": {
            "entries": {
                "path": ENTRIES_PATH.relative_to(WORDS_DIR).as_posix(),
                "bytes": ENTRIES_PATH.stat().st_size,
                "sha256": entries_sha,
            },
            "sqlite": sqlite_record,
        },
        "summary": summary,
        "deduplication": dedup,
        "sources": {
            "chhoetaigi": {
                "url": "https://github.com/ChhoeTaigi/ChhoeTaigiDatabase",
                "status": "downloaded_and_parsed",
                "license_summary": "per-file: iTaigi CC0-1.0; Taihoa CC-BY-SA-4.0; Kauiokpoo CC-BY-ND-3.0-TW; Pehoe CC-BY-SA-4.0",
                "details": chhoe_details,
            },
            "pts_taigi_words": {
                "url": PTS_BASE_URL,
                "status": "downloaded_and_parsed",
                "license": PTS_LICENSE,
                "details": pts_details,
            },
            "taiwan_tongues": {
                "url": "https://zh.wikiversity.org/zh-my/Taiwan_Tongues/臺灣臺語開源語料庫",
                "status": "downloaded_and_parsed_linked_dataset",
                "license_summary": "page CC BY-SA; linked iCorpus-100 CC BY-NC-4.0",
                "details": icorpus_details,
            },
            "thuocl": {
                "url": "https://github.com/thunlp/THUOCL",
                "status": "downloaded_mandarin_candidates_only",
                "license": "MIT",
                "details": thuocl_details,
            },
            "audits": audits,
        },
        "raw_files": sorted(fetcher.records, key=lambda item: item["path"]),
        "limitations": [
            "词库混合多个许可；用户私人使用不代表全部内容可商业使用或可再分发。",
            "CC BY-ND 来源按原字段收录，不改写释义；发布衍生数据前需单独评估。",
            "公视页面未发现明确再利用许可，仅按用户私人使用范围收录，再分发需授权。",
            "国家教育研究院实用台语词典与清大语料库未批量镜像；只保存审计证据。",
            "THUOCL 只用于普通话候选覆盖，不能作为台湾闽南语读音或对译证据。",
            "自动提取不能替代语言专家复核；verified 表示来源字段对齐，不表示所有读音在任意语境都唯一。",
        ],
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建台湾闽南语离线词库")
    parser.add_argument("--refresh", action="store_true", help="重新下载所有原始来源")
    parser.add_argument("--offline", action="store_true", help="只使用项目内已下载来源")
    parser.add_argument("--skip-sqlite", action="store_true", help="不生成 SQLite 查询副本")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.refresh and args.offline:
        raise SystemExit("--refresh 与 --offline 不能同时使用")
    manifest = build(args)
    summary = manifest["summary"]
    print(
        "词库构建完成："
        f"{summary['entries']} 条，"
        f"台罗 {summary['with_tailo']} 条，"
        f"普通话对译 {summary['with_mandarin']} 条，"
        f"可信 {summary['verified']} 条"
    )
    print(f"输出：{ENTRIES_PATH}")
    print(f"清单：{MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
