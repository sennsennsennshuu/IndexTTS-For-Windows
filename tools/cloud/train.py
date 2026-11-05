"""IndexTTS 2.5 台语 GPT LoRA 训练器（只读基础模型，不加载 S2Mel/BigVGAN）。

设计边界：
1. 训练阶段只加载 GPT 与 tokenizer；codes、speaker、emotion 均来自预提取缓存。
2. 文本统一使用已注册的 ``<|zh|> `` 前缀，语言身份使用 minnan 的第 100 行。
3. teacher forcing 使用单 EOS：
   text_in=[start_text, text_tokens, stop_text]
   audio_in=[start_mel, codes]
   target=[codes, stop_mel]
4. 参考条件由同 speaker、同 split 的其他 utterance 提供；未知 speaker 只允许
   使用目标自身的 192 维 style，emotion 使用固定的 neutral 向量，避免内容泄漏。
5. 保存时只覆盖本训练输出目录中的 checkpoint，不删除原项目或其他 agent 的产物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

# 训练脚本从项目根目录启动；这里只调整 import 路径，不改变调用方工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.cloud.runtime import configure, encode_text, load_gpt
except ImportError:  # 兼容直接从 tools/cloud 目录启动
    from runtime import configure, encode_text, load_gpt

import torch
import torch.nn.functional as F

MINNAN_INDEX = 100
ZH_INDEX = 1
FEATURES_TAG = "w2vbert_layer17_normalized_codec25hz_campplus192_emovec1280"
TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]

DEFAULTS: dict[str, Any] = {
    "model_dir": "models/IndexTTS-2.5",
    "feature_dir": "training/features",
    "base_manifest": "training/base_model_manifest.json",
    "train_manifest": "training/data/minnan/train.cloud.jsonl",
    "valid_manifest": "training/data/minnan/valid.cloud.jsonl",
    "output_dir": "training/runs/minnan-lora",
    "steps": 5000,
    "grad_accum": 8,
    "batch_size": 1,
    "validate_every": 100,
    "save_every": 100,
    "max_valid": 64,
    "warmup_steps": 100,
    "min_lr_ratio": 0.10,
    "rank": 32,
    "alpha": 64,
    "dropout": 0.05,
    "lr": 1.0e-4,
    "row_lr": 3.0e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 1234,
    "device": "cuda",
    "precision": "bf16",
    "sdpa": False,
    "gradient_checkpointing": True,
    "allow_cross_split_refs": False,
    "neutral_emotion_path": "models/IndexTTS-2.5/feat2.pt",
    "smoke_rows": 8,
    "smoke_valid": 4,
}


def _json_default(value: Any) -> Any:
    """把少量非 JSON 类型转为可记录形式，避免日志写出失败。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stable_index(seed: int, key: str, size: int) -> int:
    if size <= 0:
        raise ValueError("stable_index size 必须大于 0")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % size


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_metric(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {split} manifest：{path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
        required = ("audio", "audio_sha256", "text", "text_type", "duration")
        for key in required:
            if key not in row or row[key] in (None, ""):
                raise ValueError(f"{path}:{line_number} 缺少字段 {key}")
        if row.get("split") not in (None, split):
            raise ValueError(f"{path}:{line_number} split={row.get('split')} 与文件 split={split} 不一致")
        source_hash = str(row["audio_sha256"]).lower()
        cloud_hash = str(row.get("cloud_audio_sha256") or source_hash).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError(f"{path}:{line_number} audio_sha256 不是 SHA256")
        if not re.fullmatch(r"[0-9a-f]{64}", cloud_hash):
            raise ValueError(f"{path}:{line_number} cloud_audio_sha256 不是 SHA256")
        speaker = row.get("speaker_id")
        rows.append({
            "audio": str(row["audio"]),
            "audio_sha256": source_hash,
            "cloud_audio_sha256": cloud_hash,
            "text": str(row["text"]),
            "text_type": str(row["text_type"]),
            "source": str(row.get("source", "")),
            "speaker_id": None if speaker in (None, "") else str(speaker),
            "duration": float(row["duration"]),
            "split": split,
        })
    if not rows:
        raise ValueError(f"{split} manifest 为空：{path}")
    return rows


def _manifest_path(config: dict[str, Any], split: str) -> Path:
    key = f"{split}_manifest"
    configured = Path(str(config[key]))
    if configured.is_file():
        return configured
    # 兼容较早的非 cloud 命名；cloud 文件是首选。
    legacy = configured.with_name(configured.name.replace(".cloud.jsonl", ".jsonl"))
    if legacy.is_file():
        return legacy
    return configured


def _base_identity(config: dict[str, Any]) -> dict[str, Any]:
    """从固定基础模型 manifest 读取 GPT/codec 摘要，并校验摘要完整。"""
    path = Path(str(config["base_manifest"]))
    if not path.is_file():
        raise FileNotFoundError(f"缺少基础模型 manifest：{path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    files = payload.get("files")
    if files is None:
        raise ValueError("base_model_manifest.json 缺少 files")

    def find_entry(name: str) -> dict[str, Any]:
        if isinstance(files, dict):
            value = files.get(name)
            return value if isinstance(value, dict) else {}
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict) and (item.get("path") == name or item.get("name") == name):
                    return item
        return {}

    gpt = find_entry("gpt.pth")
    codec = find_entry("codec.pth")
    gpt_hash = str(gpt.get("sha256", "")).lower()
    codec_hash = str(codec.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", gpt_hash) or not re.fullmatch(r"[0-9a-f]{64}", codec_hash):
        raise ValueError("base_model_manifest.json 的 gpt.pth/codec.pth 摘要不完整")
    return {
        "schema": 1,
        "model_fingerprint": _sha256_bytes(raw),
        "gpt_sha256": gpt_hash,
        "codec_sha256": codec_hash,
        "features": FEATURES_TAG,
    }


def _expected_feature_signature(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """与 tools/cloud/extract.py 完全一致：基础签名加本条音频映射摘要。"""
    return {
        **base,
        "source_audio_sha256": row["audio_sha256"],
        "cloud_audio_sha256": row.get("cloud_audio_sha256", row["audio_sha256"]),
    }


def _data_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    payload = [
        {
            "audio_sha256": row["audio_sha256"],
            "cloud_audio_sha256": row.get("cloud_audio_sha256", row["audio_sha256"]),
            "text": row["text"],
            "text_type": row["text_type"],
            "speaker_id": row.get("speaker_id"),
            "duration": round(float(row["duration"]), 6),
            "split": row.get("split"),
        }
        for row in rows
    ]
    payload.sort(key=lambda item: (item["audio_sha256"], item["text_type"], item["text"]))
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _feature_fingerprint(rows: Iterable[dict[str, Any]], base: dict[str, Any]) -> str:
    signatures = [_expected_feature_signature(base, row) for row in rows]
    signatures.sort(key=_canonical_json)
    return _sha256_bytes(_canonical_json(signatures).encode("utf-8"))


class FeatureStore:
    """按源音频 SHA256 读取并严格校验预提取特征。"""

    def __init__(self, root: Path, base: dict[str, Any]):
        self.root = root
        self.base = base
        self._cache: dict[str, dict[str, Any]] = {}

    def expected_signature(self, row: dict[str, Any]) -> dict[str, Any]:
        return _expected_feature_signature(self.base, row)

    def get(self, row: dict[str, Any]) -> dict[str, Any]:
        key = row["audio_sha256"]
        if key in self._cache:
            return self._cache[key]
        path = self.root / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"缺少特征缓存：{path}")
        item = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(item, dict):
            raise ValueError(f"特征缓存不是 dict：{path}")
        expected = self.expected_signature(row)
        if item.get("signature") != expected:
            raise RuntimeError(
                f"特征 signature 不一致，拒绝训练：{path}\n"
                f"expected={_canonical_json(expected)}\nactual={_canonical_json(item.get('signature'))}"
            )
        if int(item.get("schema", -1)) != 1:
            raise ValueError(f"特征 schema 不是 1：{path}")
        codes = item.get("codes")
        speaker = item.get("speaker")
        emotion = item.get("emotion")
        if not isinstance(codes, torch.Tensor) or codes.numel() < 1:
            raise ValueError(f"codes 缺失或为空：{path}")
        codes = codes.reshape(-1).to(dtype=torch.long)
        if int(codes.min()) < 0 or int(codes.max()) >= 8192:
            raise ValueError(f"codes 超出 [0,8192)：{path}")
        if not isinstance(speaker, torch.Tensor) or speaker.numel() != 192:
            raise ValueError(f"speaker 必须是 192 维：{path}")
        if not isinstance(emotion, torch.Tensor) or emotion.numel() != 1280:
            raise ValueError(f"emotion 必须是 1280 维：{path}")
        duration = float(item.get("duration", 0.0))
        if not math.isfinite(duration) or abs(duration - float(row["duration"])) > 0.15:
            raise ValueError(f"缓存与 manifest 时长不一致：{path}")
        result = {
            "codes": codes,
            "speaker": speaker.reshape(-1).float(),
            "emotion": emotion.reshape(-1).float(),
            "duration": duration,
            "signature": expected,
        }
        self._cache[key] = result
        return result

    def validate_all(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.get(row)


def _group_views(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["audio_sha256"], []).append(row)
    for values in groups.values():
        values.sort(key=lambda item: (item["text_type"], item["text"]))
    return groups


def select_validation_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """按来源和输入书写方式轮流抽取，避免排序后仅验证 Common Voice 汉字。

    每个分层内部按固定哈希排序，不依赖源目录的枚举顺序。同一套配置重跑或
    断点恢复时验证案例保持一致；这只是固定验证子集，不改动数据 split。
    """
    from collections import defaultdict

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("source", "unknown")), row["text_type"])].append(row)
    for values in strata.values():
        values.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['audio_sha256']}:{row['text_type']}".encode()).hexdigest())
    selected = []
    cursor = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for key in sorted(strata):
            if cursor < len(strata[key]):
                selected.append(strata[key][cursor])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        cursor += 1
    return selected


class ViewSampler:
    """同一 audio 只计一次；每个 epoch 轮换 Hanzi/TaiLo 视图。"""

    def __init__(self, groups: dict[str, list[dict[str, Any]]], seed: int):
        if not groups:
            raise ValueError("训练视图为空")
        self.groups = groups
        self.hashes = sorted(groups)
        self.seed = seed

    def next(self, cursor: int) -> tuple[dict[str, Any], int, int]:
        count = len(self.hashes)
        group_index = cursor % count
        epoch = cursor // count
        key = self.hashes[group_index]
        views = self.groups[key]
        view_index = (epoch + _stable_index(self.seed, key, len(views))) % len(views)
        return views[view_index], cursor + 1, view_index


class ReferenceIndex:
    """从同 speaker、同 split 的不同音频选择 speaker/emotion 参考。"""

    def __init__(self, rows: list[dict[str, Any]], seed: int, allow_cross_split: bool = False):
        self.seed = seed
        self.allow_cross_split = allow_cross_split
        self.by_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.all_by_speaker: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            speaker = row.get("speaker_id")
            if speaker is None:
                continue
            self.by_split.setdefault((row["split"], speaker), []).append(row)
            self.all_by_speaker.setdefault(speaker, []).append(row)
        for values in list(self.by_split.values()) + list(self.all_by_speaker.values()):
            values.sort(key=lambda item: (item["audio_sha256"], item["text_type"], item["text"]))

    def _candidates(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        speaker = row.get("speaker_id")
        if speaker is None:
            return []
        key = (row["split"], speaker)
        values = self.by_split.get(key, [])
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in values:
            digest = item["audio_sha256"]
            if digest == row["audio_sha256"] or digest in seen:
                continue
            seen.add(digest)
            result.append(item)
        return result

    def refs(self, row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if row.get("speaker_id") is None:
            return None, None
        candidates = self._candidates(row)
        if not candidates:
            return None, None
        if len(candidates) == 1:
            # 保留真实 speaker_id，只有 speaker style 使用另一句，情感采用固定平静均值。
            return candidates[0], None
        start = _stable_index(self.seed, f"{row['split']}:{row['audio_sha256']}:{row['text_type']}", len(candidates))
        speaker_ref = candidates[start]
        emotion_ref = candidates[(start + 1) % len(candidates)]
        if speaker_ref["audio_sha256"] == emotion_ref["audio_sha256"]:
            emotion_ref = candidates[(start + 2) % len(candidates)]
        if speaker_ref["audio_sha256"] == emotion_ref["audio_sha256"]:
            raise ValueError("无法选择两个不同参考 utterance")
        return speaker_ref, emotion_ref


class MinnanLanguageAdapter(torch.nn.Module):
    """冻结整张语言表，只训练 minnan row 的 FP32 delta。"""

    def __init__(self, embedding: torch.nn.Embedding, index: int = MINNAN_INDEX, source_index: int = ZH_INDEX):
        super().__init__()
        if not isinstance(embedding, torch.nn.Embedding):
            raise TypeError("lang_embedding 必须是 nn.Embedding")
        if embedding.num_embeddings <= max(index, source_index):
            raise ValueError("lang_embedding 行数不足")
        self.embedding = embedding
        self.index = int(index)
        self.source_index = int(source_index)
        with torch.no_grad():
            embedding.weight[self.index].copy_(embedding.weight[self.source_index])
        embedding.weight.requires_grad_(False)
        self.delta = torch.nn.Parameter(torch.zeros(embedding.embedding_dim, dtype=torch.float32, device=embedding.weight.device))
        self.register_buffer("base_row", embedding.weight[self.index].detach().float().cpu().clone(), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(dtype=torch.long)
        base = F.embedding(ids, self.embedding.weight)
        mask = (ids == self.index).to(dtype=base.dtype).unsqueeze(-1)
        return base + mask * self.delta.to(device=base.device, dtype=base.dtype)

    def export_state(self) -> dict[str, torch.Tensor | int]:
        base = self.embedding.weight[self.index].detach().float().cpu()
        delta = self.delta.detach().float().cpu()
        return {
            "index": self.index,
            "source_index": self.source_index,
            "base_row": base,
            "delta": delta,
            "actual_row": base + delta,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if int(state.get("index", -1)) != self.index:
            raise ValueError("minnan row index 不一致")
        base = torch.as_tensor(state["base_row"], dtype=torch.float32, device=self.embedding.weight.device)
        delta = torch.as_tensor(state["delta"], dtype=torch.float32, device=self.delta.device)
        actual = torch.as_tensor(state["actual_row"], dtype=torch.float32, device=self.embedding.weight.device)
        if base.numel() != self.embedding.embedding_dim or delta.numel() != self.embedding.embedding_dim:
            raise ValueError("minnan row 维度不一致")
        with torch.no_grad():
            self.embedding.weight[self.index].copy_(base)
            self.delta.copy_(delta)
            self.base_row.copy_(base.detach().float().cpu())
        if not torch.allclose(actual.float(), base.float() + delta.float(), atol=2.0e-2, rtol=2.0e-2):
            raise ValueError("minnan row actual/base+delta 不一致")


def _load_neutral_emotion(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    path = Path(str(config["neutral_emotion_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"缺少 neutral emotion 矩阵：{path}")
    matrix = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.shape[0] < 24:
        raise ValueError("feat2.pt 必须是至少 24 行的二维 tensor")
    return matrix[-24:].mean(dim=0).to(device=device, dtype=torch.float32)


def _load_tokenizer(config: dict[str, Any]):
    from indextts.utils.tokenizer import LANGUAGE_DICT, get_tokenizer
    if int(LANGUAGE_DICT.get("minnan", -1)) != MINNAN_INDEX:
        raise ValueError("当前 tokenizer 的 minnan row 不是 100，拒绝混用")
    return get_tokenizer(multilingual=True, model_dir=str(config["model_dir"]))


def _configure_attention(model: torch.nn.Module, use_sdpa: bool) -> None:
    """将 runtime 创建的 eager GPT2 注意力替换为同权重 SDPA 注意力。"""
    if not use_sdpa:
        return
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("GPT 缺少 config，无法启用 SDPA")
    if getattr(config, "_attn_implementation", "eager") == "sdpa":
        return
    try:
        from indextts.gpt.transformers_gpt2 import GPT2SdpaAttention
        for index, block in enumerate(model.h):
            old = block.attn
            new = GPT2SdpaAttention(config, layer_idx=index)
            new.load_state_dict(old.state_dict(), strict=False)
            block.attn = new
        model._attn_implementation = "sdpa"
        config._attn_implementation = "sdpa"
    except Exception as exc:
        raise RuntimeError("无法启用 SDPA；请保持 sdpa=false 或适配 runtime.load_gpt 的 attention 构建") from exc


def _inject_lora(model, config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config["dropout"]),
        bias="none",
        target_modules=list(TARGET_MODULES),
        fan_in_fan_out=True,
    )
    wrapped = get_peft_model(model, lora)
    wrapped.train()
    return wrapped


def _enable_gradient_checkpointing(model, enabled: bool) -> None:
    if not enabled:
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError as exc:
        raise RuntimeError("当前 transformers/PEFT 不支持 use_reentrant=False") from exc
    # 非重入 checkpoint 不需要额外的 embedding 输出梯度 hook；PEFT 已处理初始化。



def make_teacher_forcing_sequences(
    codes: torch.Tensor | list[int], text_tokens: torch.Tensor | list[int], g
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造单 EOS、无 padding 的 teacher forcing 序列，并保持 batch 维为 1。

    真实 LearnedPositionEmbeddings.forward 使用 ``x.shape[1]`` 作为时间长度，
    因此这里不能返回 1D 张量；必须返回 [1,T]，否则 text/mel position
    embedding 会把特征维误当成长度。
    """
    codes_t = torch.as_tensor(codes, dtype=torch.long).reshape(-1)
    text_t = torch.as_tensor(text_tokens, dtype=torch.long).reshape(-1)
    if codes_t.numel() == 0 or text_t.numel() == 0:
        raise ValueError("codes/text_tokens 必须是非空一维序列")
    device = codes_t.device
    start_text = torch.tensor([int(g.start_text_token)], dtype=torch.long, device=device)
    stop_text = torch.tensor([int(g.stop_text_token)], dtype=torch.long, device=device)
    start_mel = torch.tensor([int(g.start_mel_token)], dtype=torch.long, device=device)
    stop_mel = torch.tensor([int(g.stop_mel_token)], dtype=torch.long, device=device)
    text_in = torch.cat([start_text, text_t.to(device), stop_text]).unsqueeze(0)
    audio_in = torch.cat([start_mel, codes_t.to(device)]).unsqueeze(0)
    target = torch.cat([codes_t.to(device), stop_mel]).unsqueeze(0)
    if target.shape[1] != audio_in.shape[1]:
        raise AssertionError("teacher forcing 的 audio_in/target 长度不一致")
    return text_in, audio_in, target


def build_tensors(
    g,
    text_tokens: torch.Tensor,
    codes: torch.Tensor,
    lang_id: torch.Tensor,
    speaker_raw: torch.Tensor,
    emotion: torch.Tensor,
) -> dict[str, Any]:
    """生成 GPT 输入 embedding 和单样本 CE 目标。"""
    device = next(g.text_embedding.parameters()).device
    text_in, audio_in, target = make_teacher_forcing_sequences(codes, text_tokens, g)
    text_in = text_in.to(device)
    audio_in = audio_in.to(device)
    target = target.to(device)
    lang_id = lang_id.to(device=device, dtype=torch.long).reshape(-1)
    if lang_id.numel() != 1 or int(lang_id.item()) != MINNAN_INDEX:
        raise ValueError(f"训练语言 ID 必须是 {MINNAN_INDEX}，实际为 {lang_id.tolist()}")

    embed_dtype = g.text_embedding.weight.dtype
    speaker = speaker_raw.to(device=device, dtype=g.spk_emb_proj.weight.dtype).reshape(1, -1)
    spk = g.spk_emb_proj(speaker).unsqueeze(1)  # [1,1,D]
    emo = emotion.to(device=device, dtype=embed_dtype).reshape(1, -1)
    zeros = torch.zeros((1, 2, spk.shape[-1]), device=device, dtype=embed_dtype)
    cond = torch.cat([spk + emo.unsqueeze(1), zeros], dim=1)
    # text_in/audio_in 已经是 [1,T]；position embedding 与 token embedding 共用该 batch 维。
    text_emb = g.text_embedding(text_in) + g.text_pos_embedding(text_in)
    text_emb = text_emb + g.lang_embedding(lang_id).unsqueeze(1)
    mel_emb = g.mel_embedding(audio_in) + g.mel_pos_embedding(audio_in)
    inputs = torch.cat([cond, text_emb, mel_emb], dim=1)
    total = int(inputs.shape[1])
    n_positions = int(getattr(g.gpt.config, "n_positions", 0))
    if n_positions <= 0:
        raise ValueError("GPT config.n_positions 无效")
    if total > n_positions:
        raise ValueError(f"序列长度越界：{total} > {n_positions}")
    if text_in.shape[1] > g.text_pos_embedding.emb.num_embeddings:
        raise ValueError("text_in 超出 text position embedding 范围")
    if audio_in.shape[1] > g.mel_pos_embedding.emb.num_embeddings:
        raise ValueError("audio_in 超出 mel position embedding 范围")
    return {
        "inputs": inputs,
        "text_in": text_in,
        "audio_in": audio_in,
        "target": target,
        "cond_len": int(cond.shape[1]),
        "text_len": int(text_emb.shape[1]),
        "mel_len": int(mel_emb.shape[1]),
        "total_len": total,
    }


def forward_loss(g, tensors: dict[str, Any]) -> dict[str, Any]:
    """只对 mel 段做 CE；不调用 UnifiedVoice.forward()。"""
    output = g.gpt(inputs_embeds=tensors["inputs"], use_cache=False, return_dict=True)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0]
    start = tensors["cond_len"] + tensors["text_len"]
    mel_hidden = hidden[:, start:, :]
    target = tensors["target"].reshape(1, -1).to(device=mel_hidden.device, dtype=torch.long)
    if mel_hidden.shape[1] != target.shape[1]:
        raise RuntimeError(f"mel hidden/target 长度不一致：{mel_hidden.shape[1]} vs {target.shape[1]}")
    logits = g.mel_head(g.final_norm(mel_hidden)).transpose(1, 2)
    loss_sum = F.cross_entropy(logits.float(), target, reduction="sum")
    tokens = int(target.numel())
    loss_per_token = loss_sum / tokens
    return {
        "loss_sum": loss_sum,
        "loss_per_token": loss_per_token,
        "tokens": tokens,
        "logits": logits,
        "target": target,
    }



def _validate_batch_settings(config: dict[str, Any]) -> tuple[int, int]:
    """校验并返回 (batch_size, grad_accum)，防止 0 或负数造成死循环/除零。"""
    try:
        batch_size = int(config.get("batch_size", 1))
        grad_accum = int(config.get("grad_accum", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size/grad_accum 必须是整数") from exc
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，实际为 {batch_size}")
    if grad_accum < 1:
        raise ValueError(f"grad_accum 必须 >= 1，实际为 {grad_accum}")
    return batch_size, grad_accum


def forward_loss_batch(g, tensors_list: list[dict[str, Any]]) -> dict[str, Any]:
    """ragged batch：右侧 pad、单次 GPT forward，只对有效 mel 位置计算 CE。

    - 每个样本的 inputs 为 [1, Ni, D]，右侧补到 batch 内最大长度；有效 token
      始终从位置 0 连续排列，因此不会发生左侧 pad 导致的 mel offset 错位。
    - attention_mask 逐样本标记有效长度，pad 行不参与任何有效位置的注意力。
    - 按每个样本自己的 cond_len+text_len 切 mel hidden，再合并所有有效 mel
      位置，只做一次 final_norm + mel_head + CE。
    - 目标函数保持“每 utterance 先平均、batch 内再平均”：loss_mean 用于
      backward；batch_size=1 时它等于原 forward_loss 的 loss_per_token。
      loss_sum/tokens 保留给 token 加权日志，不能与 loss_mean 混用。
    """
    if not tensors_list:
        raise ValueError("forward_loss_batch 至少需要一个样本")
    device = tensors_list[0]["inputs"].device
    max_len = max(int(item["total_len"]) for item in tensors_list)
    if max_len <= 0:
        raise ValueError("batch 序列长度无效")

    padded_inputs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for index, item in enumerate(tensors_list):
        inputs = item["inputs"]
        if inputs.ndim != 3 or inputs.shape[0] != 1 or inputs.shape[1] != int(item["total_len"]):
            raise ValueError(f"batch 样本 {index} 的 inputs 必须是 [1,total_len,D]")
        pad = max_len - int(item["total_len"])
        if pad:
            # 只补右侧；零值仅占位，不进入 mel 切片，也不会成为 CE target。
            inputs = F.pad(inputs, (0, 0, 0, pad))
        padded_inputs.append(inputs)
        mask = torch.zeros(max_len, dtype=torch.long, device=device)
        mask[: int(item["total_len"])] = 1
        masks.append(mask)

    inputs_batch = torch.cat(padded_inputs, dim=0)
    attention_mask = torch.stack(masks, dim=0)
    output = g.gpt(
        inputs_embeds=inputs_batch,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0]

    mel_pieces: list[torch.Tensor] = []
    target_pieces: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, item in enumerate(tensors_list):
        start = int(item["cond_len"]) + int(item["text_len"])
        mel_len = int(item["mel_len"])
        end = start + mel_len
        if start < 0 or mel_len <= 0 or end != int(item["total_len"]):
            raise ValueError(f"batch 样本 {index} 的 mel 切片边界不一致")
        target = item["target"].reshape(-1)
        if target.numel() != mel_len:
            raise ValueError(f"batch 样本 {index} 的 target 长度与 mel_len 不一致")
        mel_pieces.append(hidden[index:index + 1, start:end, :])
        target_pieces.append(target.to(device=hidden.device, dtype=torch.long))
        spans.append((cursor, cursor + mel_len))
        cursor += mel_len

    # 所有有效 mel 位置合并后只做一次头部计算，pad 位置完全不进入 loss。
    mel_hidden = torch.cat(mel_pieces, dim=1)
    target = torch.cat(target_pieces, dim=0)
    logits = g.mel_head(g.final_norm(mel_hidden)).float()
    if logits.shape[1] != target.numel():
        raise AssertionError("batch logits/target token 数不一致")
    token_losses = F.cross_entropy(
        logits.transpose(1, 2), target.reshape(1, -1), reduction="none"
    ).reshape(-1)
    loss_sum = token_losses.sum()
    tokens = int(target.numel())
    per_utterance = torch.stack([token_losses[begin:end].mean() for begin, end in spans])
    loss_mean = per_utterance.mean()
    return {
        "loss_sum": loss_sum,
        "loss_per_token": loss_sum / tokens,
        "loss_mean": loss_mean,
        "per_utterance_mean": per_utterance,
        "tokens": tokens,
        "utterances": len(tensors_list),
        "logits": logits,
        "target": target,
        "mel_spans": spans,
        "attention_mask": attention_mask,
    }


def take_batch(sampler, cursor: int, batch_size: int) -> tuple[list[dict[str, Any]], int]:
    """逐次更新游标，确保一个 batch 消费不同录音，而非重复同一游标。"""
    rows = []
    for _ in range(batch_size):
        row, cursor, _ = sampler.next(cursor)
        rows.append(row)
    return rows, cursor


def _prepare_sample(
    g,
    store: FeatureStore,
    refs: ReferenceIndex,
    tokenizer,
    neutral_emotion: torch.Tensor,
    row: dict[str, Any],
    max_text_tokens: int,
) -> dict[str, Any]:
    target_feature = store.get(row)
    speaker_ref, emotion_ref = refs.refs(row)
    if speaker_ref is None:
        speaker_feature = target_feature
        emotion = neutral_emotion
        ref_mode = "self_style+neutral"
    elif emotion_ref is None:
        speaker_feature = store.get(speaker_ref)
        emotion = neutral_emotion
        ref_mode = "same_speaker_style+neutral"
    else:
        speaker_feature = store.get(speaker_ref)
        emotion_feature = store.get(emotion_ref)
        emotion = emotion_feature["emotion"]
        ref_mode = "same_speaker_refs"
    tokens = encode_text(tokenizer, row["text"], maximum=max_text_tokens)
    tensors = build_tensors(
        g,
        torch.tensor(tokens, dtype=torch.long),
        target_feature["codes"],
        torch.tensor([MINNAN_INDEX], dtype=torch.long),
        speaker_feature["speaker"],
        emotion,
    )
    return {
        "tensors": tensors,
        "row": row,
        "ref_mode": ref_mode,
        "speaker_ref": None if speaker_ref is None else speaker_ref["audio_sha256"],
        "emotion_ref": None if emotion_ref is None else emotion_ref["audio_sha256"],
    }


def _autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _gpu_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"gpu_allocated_mb": 0.0, "gpu_reserved_mb": 0.0}
    return {
        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
        "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
    }


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_adapter(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True, save_embedding_layers=False)


def _load_adapter(base_model, path: Path):
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(path), is_trainable=True)


def _checkpoint_dirs(output_dir: Path) -> list[Path]:
    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(
        [item for item in root.iterdir() if item.is_dir() and re.fullmatch(r"step_\d+", item.name)],
        key=lambda item: int(item.name.split("_")[1]),
    )


def _prune_checkpoints(output_dir: Path, best_step: int | None) -> None:
    """只清理本训练器创建的 step_* 目录，保留最后 3 个与 best。"""
    dirs = _checkpoint_dirs(output_dir)
    keep = {item.name for item in dirs[-3:]}
    if best_step is not None:
        keep.add(f"step_{best_step:06d}")
    best_record = output_dir / "best.json"
    if best_record.is_file():
        recorded_step = int(json.loads(best_record.read_text(encoding="utf-8"))["step"])
        keep.add(f"step_{recorded_step:06d}")
    root = (output_dir / "checkpoints").resolve()
    for item in dirs:
        if item.name in keep:
            continue
        resolved = item.resolve()
        if resolved.parent != root or not item.name.startswith("step_"):
            raise RuntimeError(f"拒绝清理非训练器 checkpoint：{resolved}")
        shutil.rmtree(item)


def save_checkpoint(
    output_dir: Path,
    step: int,
    gpt_model,
    lang_adapter: MinnanLanguageAdapter,
    optimizer,
    scheduler,
    config: dict[str, Any],
    base_identity: dict[str, Any],
    data_fingerprint: str,
    feature_fingerprint: str,
    cursor: int,
    best_val_loss: float | None,
    val_loss: float | None = None,
    is_best: bool = False,
) -> Path:
    """先写临时目录，再原子替换为本训练器的 step_* checkpoint。"""
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".tmp-step-{step:06d}-{uuid.uuid4().hex}"
    final = root / f"step_{step:06d}"
    try:
        temporary.mkdir(parents=True, exist_ok=False)
        _save_adapter(gpt_model, temporary / "adapter")
        extra_state = {
            "format": 1,
            "step": int(step),
            "base_identity": base_identity,
            "data_fingerprint": data_fingerprint,
            "feature_fingerprint": feature_fingerprint,
            "lang": lang_adapter.export_state(),
            "config": config,
            "batch_size": int(config.get("batch_size", 1)),
            "grad_accum": int(config.get("grad_accum", 1)),
            "effective_batch_size": int(config.get("batch_size", 1)) * int(config.get("grad_accum", 1)),
        }
        trainer_state = {
            "step": int(step),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "rng": _capture_rng(),
            "cursor": int(cursor),
            "best_val_loss": best_val_loss,
            "accum_count": 0,
            "samples_per_optimizer_step": int(config.get("batch_size", 1)) * int(config.get("grad_accum", 1)),
        }
        _atomic_torch_save(extra_state, temporary / "extra_state.pt")
        _atomic_torch_save(trainer_state, temporary / "trainer_state.pt")
        _atomic_write_json(temporary / "checkpoint.json", {
            "step": int(step), "val_loss": val_loss, "best_val_loss": best_val_loss, "is_best": bool(is_best),
        })
        os.replace(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    if is_best:
        _atomic_write_json(output_dir / "best.json", {
            "step": int(step), "path": str(final), "val_loss": val_loss, "best_val_loss": best_val_loss,
        })
    _prune_checkpoints(output_dir, step if is_best else None)
    return final


def read_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if path.is_file() and path.name == "best.json":
        path = Path(json.loads(path.read_text(encoding="utf-8"))["path"])
    if not (path / "extra_state.pt").is_file():
        raise FileNotFoundError(f"不是完整 checkpoint：{path}")
    extra = torch.load(path / "extra_state.pt", map_location="cpu", weights_only=True)
    trainer = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    meta = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    return extra, trainer, meta


def resolve_resume(path: str, output_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_file() and candidate.name == "best.json":
        return Path(json.loads(candidate.read_text(encoding="utf-8"))["path"])
    if (candidate / "adapter").is_dir() and (candidate / "extra_state.pt").is_file():
        return candidate
    if (candidate / "checkpoints").is_dir():
        dirs = _checkpoint_dirs(candidate)
        if dirs:
            return dirs[-1]
    if candidate.is_dir() and candidate == output_dir:
        dirs = _checkpoint_dirs(output_dir)
        if dirs:
            return dirs[-1]
    raise FileNotFoundError(f"找不到可恢复 checkpoint：{path}")


def _compatible_config(old: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("model_dir", "feature_dir", "base_manifest", "rank", "alpha", "dropout", "train_manifest", "valid_manifest"):
        if old.get(key) != current.get(key):
            raise ValueError(f"resume 配置不兼容：{key}: {old.get(key)} != {current.get(key)}")
    if list(old.get("target_modules", [])) != list(current.get("target_modules", TARGET_MODULES)):
        raise ValueError("resume 配置不兼容：target_modules")
    # 旧 checkpoint 没有 batch_size 字段，按 1 处理；grad_accum 同样做显式比较。
    for key, fallback in (("batch_size", 1), ("grad_accum", 1)):
        old_value = int(old.get(key, fallback))
        current_value = int(current.get(key, fallback))
        if old_value != current_value:
            raise ValueError(f"resume 配置不兼容：{key}: {old_value} != {current_value}")


def _build_optimizer(g, lang_adapter: MinnanLanguageAdapter, config: dict[str, Any]):
    lora_params = [parameter for parameter in g.gpt.parameters() if parameter.requires_grad]
    if not lora_params:
        raise RuntimeError("PEFT 没有创建可训练 LoRA 参数")
    groups = [
        {"params": lora_params, "lr": float(config["lr"]), "weight_decay": float(config["weight_decay"])},
        {"params": [lang_adapter.delta], "lr": float(config["row_lr"]), "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1.0e-8)
    return optimizer, lora_params


def _build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup = max(1, min(int(warmup_steps), max(1, total_steps // 10)))
    def factor(current_step: int) -> float:
        if current_step < warmup:
            return max(0.05, float(current_step + 1) / warmup)
        progress = min(1.0, max(0.0, (current_step - warmup) / max(1, total_steps - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate(
    g,
    rows: list[dict[str, Any]],
    store: FeatureStore,
    refs: ReferenceIndex,
    tokenizer,
    neutral_emotion: torch.Tensor,
    device: torch.device,
    precision: str,
    max_text_tokens: int,
) -> float:
    """每次验证都用当前 minnan delta 重建输入，禁止复用旧 embedding 图。"""
    if not rows:
        raise ValueError("验证集为空")
    g.gpt.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for row in rows:
            entry = _prepare_sample(g, store, refs, tokenizer, neutral_emotion, row, max_text_tokens)
            with _autocast_context(device, precision):
                result = forward_loss(g, entry["tensors"])
            total_loss += float(result["loss_sum"].detach().float().cpu())
            total_tokens += int(result["tokens"])
    g.gpt.train()
    if total_tokens <= 0:
        raise ValueError("验证 token 数为 0")
    return total_loss / total_tokens


def _load_config(value: str) -> dict[str, Any]:
    if Path(value).is_file():
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--config 必须是 JSON object 或 JSON 文件路径")
    config = dict(DEFAULTS)
    config.update(payload)
    config["target_modules"] = list(TARGET_MODULES)
    return config


def _apply_smoke(config: dict[str, Any], steps_override: int | None) -> None:
    config["steps"] = int(steps_override if steps_override is not None else min(20, int(config["steps"])))
    config["grad_accum"] = 1
    config["validate_every"] = min(5, max(1, int(config["steps"])))
    config["save_every"] = min(10, max(1, int(config["steps"])))
    config["max_valid"] = min(4, int(config["max_valid"]))
    config["warmup_steps"] = min(2, int(config["steps"]))
    config["smoke_rows"] = min(8, int(config["smoke_rows"]))


def train(config: dict[str, Any], resume: str | None = None, steps_override: int | None = None) -> dict[str, Any]:
    """执行训练；所有写操作都在 config['output_dir'] 下。"""
    config = dict(config)
    if steps_override is not None:
        config["steps"] = int(steps_override)
    if config.get("smoke"):
        _apply_smoke(config, steps_override)
    configure()
    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境没有 CUDA")
    if device.type != "cuda":
        raise RuntimeError("正式训练要求 CUDA；mock 测试请直接调用 build_tensors/forward_loss")
    if str(config.get("precision", "bf16")) != "bf16":
        raise ValueError("当前只支持 bf16 训练")
    if config.get("allow_cross_split_refs"):
        raise ValueError("禁止跨 split 取参考；参考不足时使用已记录的平静情感策略")
    batch_size, grad_accum = _validate_batch_settings(config)
    torch.set_float32_matmul_precision("high")
    seed = int(config.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_jsonl(_manifest_path(config, "train"), "train")
    valid_rows = _load_jsonl(_manifest_path(config, "valid"), "valid")
    if not valid_rows:
        raise ValueError("valid manifest 不能为空")
    train_hashes = {row["audio_sha256"] for row in train_rows}
    valid_hashes = {row["audio_sha256"] for row in valid_rows}
    overlap = train_hashes & valid_hashes
    if overlap:
        raise ValueError(f"train/valid 音频重叠，拒绝泄漏：{sorted(overlap)[:3]}")

    base = _base_identity(config)
    store = FeatureStore(Path(str(config["feature_dir"])), base)
    all_rows = train_rows + valid_rows
    store.validate_all(all_rows)
    data_fp = _data_fingerprint(all_rows)
    feature_fp = _feature_fingerprint(all_rows, base)

    groups = _group_views(train_rows)
    if config.get("smoke"):
        selected = set(sorted(groups)[: int(config.get("smoke_rows", 8))])
        groups = {key: value for key, value in groups.items() if key in selected}
    if not groups:
        raise ValueError("smoke 训练视图为空")
    refs = ReferenceIndex(train_rows + valid_rows, seed=seed, allow_cross_split=bool(config.get("allow_cross_split_refs", False)))
    tokenizer = _load_tokenizer(config)
    neutral = _load_neutral_emotion(config, device)
    g, model_config = load_gpt(str(config["model_dir"]), str(device))
    _configure_attention(g.gpt, bool(config.get("sdpa", False)))
    g.requires_grad_(False)
    lang_adapter = MinnanLanguageAdapter(g.lang_embedding, MINNAN_INDEX, ZH_INDEX)
    g.lang_embedding = lang_adapter
    # 官方推理 post_init 同样将实际 mel embedding 注册为 GPT 的 wte。
    # 此处提供真实的冻结别名，使 PEFT 的梯度检查点接口可用；输入仍使用
    # inputs_embeds，不创建随机 embedding，也不训练基础词表。
    g.gpt.wte = g.mel_embedding
    if resume:
        g.gpt = _load_adapter(g.gpt, resolve_resume(resume, output_dir) / "adapter")
    else:
        g.gpt = _inject_lora(g.gpt, config)
    _enable_gradient_checkpointing(g.gpt, bool(config.get("gradient_checkpointing", True)))
    optimizer, lora_params = _build_optimizer(g, lang_adapter, config)
    scheduler = _build_scheduler(optimizer, int(config["steps"]), int(config["warmup_steps"]), float(config["min_lr_ratio"]))

    max_text_tokens = int(getattr(g, "max_text_tokens", 600)) - 2
    if max_text_tokens < 1:
        raise ValueError("GPT max_text_tokens 无效")
    # 只保存 row；tensors 必须每次验证重建，否则会复用旧 lang_embedding 图。
    valid_limit = min(int(config["max_valid"]), len(valid_rows))
    valid_rows_for_eval = select_validation_rows(valid_rows, valid_limit, seed)

    start_step = 0
    cursor = 0
    best_val_loss = float("inf")
    if resume:
        checkpoint = resolve_resume(resume, output_dir)
        extra, trainer_state, _ = read_checkpoint(checkpoint)
        if int(extra.get("format", -1)) != 1:
            raise ValueError("checkpoint extra_state 格式不支持")
        if extra.get("base_identity") != base:
            raise ValueError("checkpoint 的基础模型标识与当前模型不一致")
        if extra.get("feature_fingerprint") != feature_fp:
            raise ValueError("checkpoint 的特征 fingerprint 与当前数据不一致")
        _compatible_config(extra.get("config", {}), config)
        lang_adapter.load_state(extra["lang"])
        # 上方已在干净基础 transformer 上恢复 adapter，禁止在 PEFT 上再次套 PEFT。
        g.gpt.train()
        optimizer, lora_params = _build_optimizer(g, lang_adapter, config)
        scheduler = _build_scheduler(optimizer, int(config["steps"]), int(config["warmup_steps"]), float(config["min_lr_ratio"]))
        if trainer_state.get("optimizer") is not None:
            optimizer.load_state_dict(trainer_state["optimizer"])
        if trainer_state.get("scheduler") is not None:
            scheduler.load_state_dict(trainer_state["scheduler"])
        start_step = int(trainer_state.get("step", 0))
        cursor = int(trainer_state.get("cursor", 0))
        best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
        _restore_rng(trainer_state.get("rng", {}))

    sampler = ViewSampler(groups, seed)
    metrics_path = output_dir / "metrics.jsonl"
    progress_path = output_dir / "progress.json"
    total_steps = int(config["steps"])
    if start_step >= total_steps:
        return {"step": start_step, "total": total_steps, "output": str(output_dir), "already_complete": True}

    if not resume:
        # 先测未微调起点，避免把一个比基础模型更差的 checkpoint 误叫做改进。
        initial_val = evaluate(g, valid_rows_for_eval, store, refs, tokenizer, neutral, device,
                               str(config["precision"]), max_text_tokens)
        if not math.isfinite(initial_val):
            raise RuntimeError("基础模型验证 loss 非有限值")
        best_val_loss = initial_val
        _append_metric(metrics_path, {"event": "validation_initial", "step": 0, "val_loss": initial_val})
        save_checkpoint(output_dir, 0, g.gpt, lang_adapter, optimizer, scheduler, config, base,
                        data_fp, feature_fp, 0, initial_val, val_loss=initial_val, is_best=True)

    _atomic_write_json(progress_path, {
        "status": "running", "step": start_step, "total": total_steps, "epoch": 0,
        "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss,
        "output": str(output_dir), "device": str(device), "gpu": _gpu_memory(),
        "lora_trainable": sum(parameter.numel() for parameter in lora_params),
        "batch_size": batch_size, "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "samples_per_optimizer_step": batch_size * grad_accum,
        "data_fingerprint": data_fp, "feature_fingerprint": feature_fp,
    })

    def log_step(
        step: int,
        loss_sum: float,
        tokens: int,
        opt_loss_mean: float,
        samples: int,
        grad_norm: float,
        elapsed: float,
        sample: dict[str, Any],
    ) -> None:
        # loss_per_token/tokenweighted 是 token 加权 CE；optloss_mean 是 backward
        # 实际优化的“每 utterance 平均 CE 的 micro 平均”，变长样本下两者不同。
        payload = {
            "step": step,
            "loss": loss_sum / max(1, tokens),
            "loss_per_token": loss_sum / max(1, tokens),
            "tokenweighted": loss_sum / max(1, tokens),
            "optloss_mean": float(opt_loss_mean),
            "tokens": tokens,
            "samples": int(samples),
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "grad_norm": float(grad_norm),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_row": float(optimizer.param_groups[1]["lr"]),
            "elapsed": round(elapsed, 3),
            "sample_audio_sha256": sample["row"]["audio_sha256"],
            "text_type": sample["row"]["text_type"],
            "ref_mode": sample["ref_mode"],
            "speaker_ref": sample["speaker_ref"],
            "emotion_ref": sample["emotion_ref"],
            **_gpu_memory(),
        }
        _append_metric(metrics_path, payload)
        _atomic_write_json(progress_path, {
            "status": "running", "step": step, "total": total_steps,
            "epoch": (cursor // max(1, len(groups))) + 1,
            "loss": payload["loss"], "loss_per_token": payload["loss_per_token"],
            "optloss_mean": payload["optloss_mean"], "samples": payload["samples"],
            "batch_size": batch_size, "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "grad_norm": payload["grad_norm"], "lr_lora": payload["lr_lora"], "lr_row": payload["lr_row"],
            "elapsed": round(elapsed, 3), "eta": round(elapsed / max(1, step) * max(0, total_steps - step), 3),
            "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss,
            "gpu": _gpu_memory(), "last_checkpoint": None,
        })

    step = start_step
    cursor = cursor
    accum_count = 0
    accum_loss_sum = 0.0
    accum_tokens = 0
    accum_opt_loss = 0.0
    accum_samples = 0
    started = time.monotonic()
    last_sample: dict[str, Any] | None = None
    saved_steps: set[int] = set()
    while step < total_steps:
        # 每个 micro 取 batch_size 个真实 ViewSampler 样本；cursor 顺序前进，
        # 不复制样本伪造 batch。grad_accum 仍表示 micro 数。
        picked, cursor = take_batch(sampler, cursor, batch_size)
        samples = [
            _prepare_sample(g, store, refs, tokenizer, neutral, item, max_text_tokens)
            for item in picked
        ]
        with _autocast_context(device, str(config["precision"])):
            result = forward_loss_batch(g, [sample["tensors"] for sample in samples])
            if not bool(torch.isfinite(result["loss_sum"]).all()) or not bool(torch.isfinite(result["loss_mean"]).all()):
                raise RuntimeError(f"非有限 loss，step={step + 1}，sample={samples[-1]['row']['audio_sha256']}")
            # batch 内先按 utterance 平均，再按 micro 数平均；batch=1 与原目标一致。
            scaled = result["loss_mean"] / grad_accum
        scaled.backward()
        accum_count += 1
        accum_loss_sum += float(result["loss_sum"].detach().float().cpu())
        accum_tokens += int(result["tokens"])
        accum_opt_loss += float(result["loss_mean"].detach().float().cpu())
        accum_samples += int(result["utterances"])
        last_sample = samples[-1]
        if accum_count < grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in g.gpt.parameters() if parameter.requires_grad] + [lang_adapter.delta],
            float(config["grad_clip"]),
        )
        if not bool(torch.isfinite(grad_norm).all()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"非有限梯度 norm，step={step + 1}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        elapsed = time.monotonic() - started
        log_step(
            step, accum_loss_sum, accum_tokens, accum_opt_loss / max(1, accum_count),
            accum_samples, float(grad_norm.detach().float().cpu()), elapsed, last_sample,
        )
        accum_count = 0
        accum_loss_sum = 0.0
        accum_tokens = 0
        accum_opt_loss = 0.0
        accum_samples = 0

        should_validate = step % int(config["validate_every"]) == 0 or step == total_steps
        val_loss = evaluate(g, valid_rows_for_eval, store, refs, tokenizer, neutral, device, str(config["precision"]), max_text_tokens) if should_validate else None
        if val_loss is not None:
            _append_metric(metrics_path, {"event": "validation", "step": step, "val_loss": float(val_loss)})
        is_best = val_loss is not None and val_loss < best_val_loss
        if is_best:
            best_val_loss = float(val_loss)
        should_save = step % int(config["save_every"]) == 0 or step == total_steps or is_best
        if should_save:
            ckpt = save_checkpoint(
                output_dir, step, g.gpt, lang_adapter, optimizer, scheduler, config, base,
                data_fp, feature_fp, cursor, None if not math.isfinite(best_val_loss) else best_val_loss,
                val_loss=val_loss, is_best=is_best,
            )
            saved_steps.add(step)
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            progress.update({"last_checkpoint": str(ckpt), "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss})
            _atomic_write_json(progress_path, progress)

    # 常规情况下 step 已到达 total；保留此分支以处理未来加入 early stop 的情况。
    if accum_count > 0 and step < total_steps:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in g.gpt.parameters() if parameter.requires_grad] + [lang_adapter.delta],
            float(config["grad_clip"]),
        )
        if not bool(torch.isfinite(grad_norm).all()):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"非有限梯度 norm，step={step + 1}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        log_step(
            step, accum_loss_sum, accum_tokens, accum_opt_loss / max(1, accum_count),
            accum_samples, float(grad_norm.detach().float().cpu()), time.monotonic() - started, last_sample,
        )

    final_val = evaluate(g, valid_rows_for_eval, store, refs, tokenizer, neutral, device, str(config["precision"]), max_text_tokens)
    is_best = final_val < best_val_loss
    if is_best:
        best_val_loss = final_val
    if step in saved_steps:
        ckpt = output_dir / "checkpoints" / f"step_{step:06d}"
        if is_best:
            _atomic_write_json(output_dir / "best.json", {
                "step": step, "path": str(ckpt), "val_loss": final_val, "best_val_loss": best_val_loss,
            })
    else:
        ckpt = save_checkpoint(
            output_dir, step, g.gpt, lang_adapter, optimizer, scheduler, config, base,
            data_fp, feature_fp, cursor, best_val_loss, val_loss=final_val, is_best=is_best,
        )
    _atomic_write_json(progress_path, {
        "status": "complete", "step": step, "total": total_steps, "epoch": (cursor // max(1, len(groups))) + 1,
        "loss": None, "loss_per_token": None, "best_val_loss": best_val_loss, "last_val_loss": final_val,
        "last_checkpoint": str(ckpt), "output": str(output_dir), "data_fingerprint": data_fp,
        "feature_fingerprint": feature_fp, "gpu": _gpu_memory(),
        "batch_size": batch_size, "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "samples_per_optimizer_step": batch_size * grad_accum,
    })
    return {"step": step, "total": total_steps, "output": str(output_dir), "checkpoint": str(ckpt), "best_val_loss": best_val_loss}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON 文件路径或 JSON object")
    parser.add_argument("--resume", default=None, help="checkpoint 目录、best.json 或输出目录")
    parser.add_argument("--steps", type=int, default=None, help="覆盖配置中的总 optimizer steps")
    parser.add_argument("--smoke", action="store_true", help="小样本过拟合模式：少量 step、accum=1")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.smoke or bool(config.get("smoke", False)):
        config["smoke"] = True
        _apply_smoke(config, args.steps)
    elif args.steps is not None:
        config["steps"] = int(args.steps)
    summary = train(config, resume=args.resume, steps_override=args.steps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


