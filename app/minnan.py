"""台语语言包：校验实际权重、恢复独立语言行，并管理可撤回的 LoRA 推理上下文。"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .paths import MINNAN_MODEL_DIR, MODEL_DIR


class MinnanTextFrontend:
    """保留原始台罗声调和台语汉字，不套用普通话 FST 或中英文 G2P。"""

    char_rep_map = {}
    clean_pattern = re.compile(r"(?!)")

    def load(self):
        return self

    def normalize(self, text):
        return " ".join(unicodedata.normalize("NFC", text).split())


class MinnanTokenizer:
    """保持训练时既有 zh 文本前缀；minnan 语言身份由第 100 行表达。"""

    def __init__(self, tokenizer):
        self.base = tokenizer

    def encode(self, text, *args, **kwargs):
        if text.startswith("<|minnan|> "):
            text = "<|zh|> " + text[len("<|minnan|> "):]
        return self.base.encode(unicodedata.normalize("NFC", text), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


@lru_cache(maxsize=128)
def _checked_digest(path: str, size: int, modified: int) -> str:
    """缓存仅绑定文件的路径、大小及纳秒修改时间；内容变化后重新核对。"""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest(path: Path) -> str:
    stat = path.stat()
    return _checked_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def validate_minnan_model(base_dir=MODEL_DIR, model_dir=MINNAN_MODEL_DIR) -> dict:
    """加载前核对适配器、语言状态及其依赖的实际基础文件，拒绝混搭权重。"""
    base_dir, model_dir = Path(base_dir), Path(model_dir)
    try:
        manifest = json.loads((model_dir / "delivery_manifest.json").read_text(encoding="utf-8"))
        base = json.loads((model_dir / "base_model_manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError("缺少台语语言包，请将正式模型放入 models/extra_language/zh-nan-tw/model/") from error
    required = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json", "extra_state.pt", "base_model_manifest.json"}
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for relative in required:
        if relative not in entries:
            raise ValueError(f"台语交付清单缺少 {relative}")
        path = model_dir / relative
        if path.stat().st_size != entries[relative]["bytes"] or digest(path) != entries[relative]["sha256"]:
            raise ValueError(f"台语模型文件摘要不符：{relative}")
    # 文本情感模型在训练时冻结且不参与普通合成，不让可选权重阻塞语言切换。
    for relative, entry in base["files"].items():
        if relative.startswith("qwen0.6bemo4-merge/"):
            continue
        path = base_dir / relative
        path.resolve().relative_to(base_dir.resolve())
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"基础模型与台语训练版本不匹配：{relative}")
    return manifest


class MinnanAdapter:
    """只给同一实例挂载一次适配器；非台语请求在上下文中完整禁用 LoRA。"""

    def __init__(self, tts, base_dir=MODEL_DIR):
        self.tts = tts
        self.base_dir = Path(base_dir)
        self.original_tokenizer = tts.tokenizer
        self.original_frontend = tts.text_process
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        validate_minnan_model(self.base_dir)
        import torch
        from peft import PeftModel
        from indextts.utils.tokenizer import LANGUAGE_DICT
        extra = torch.load(MINNAN_MODEL_DIR / "extra_state.pt", map_location="cpu", weights_only=True)
        fingerprint = digest(MINNAN_MODEL_DIR / "base_model_manifest.json")
        if (extra.get("format") != 1 or extra["lang"]["index"] != 100 or LANGUAGE_DICT.get("minnan") != 100
                or extra["base_identity"]["model_fingerprint"] != fingerprint):
            raise ValueError("台语语言行或基础模型指纹不匹配")
        row = extra["lang"]["actual_row"]
        if row.shape != self.tts.gpt.lang_embedding.weight[100].shape or not torch.isfinite(row).all():
            raise ValueError("台语语言行的形状或数值无效")
        wrapped = PeftModel.from_pretrained(self.tts.gpt.gpt, str(MINNAN_MODEL_DIR / "adapter"), is_trainable=False)
        with torch.no_grad():
            self.tts.gpt.lang_embedding.weight[100].copy_(row.to(self.tts.device))
        self.tts.gpt.gpt = wrapped
        wrapped.eval()
        # 推理包装器必须引用已挂载的 transformer；原包装器不能继续使用旧对象。
        self.tts.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=self.tts.use_bf16)
        self.loaded = True

    @contextmanager
    def activate(self, enabled: bool):
        """异常退出也恢复前端；不同语言共享参考条件时不会污染文本和适配器状态。"""
        if enabled:
            self.load()
        context = self.tts.gpt.gpt.disable_adapter() if self.loaded and not enabled else nullcontext()
        previous = (self.tts.tokenizer, self.tts.text_process)
        try:
            self.tts.tokenizer = MinnanTokenizer(self.original_tokenizer) if enabled else self.original_tokenizer
            self.tts.text_process = MinnanTextFrontend() if enabled else self.original_frontend
            with context:
                yield
        finally:
            self.tts.tokenizer, self.tts.text_process = previous
