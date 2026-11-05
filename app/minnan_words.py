"""台语词库与本地整句翻译：对译来源可追踪，未知内容不伪装成词库命中。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import re
import threading
import unicodedata

from .paths import MINNAN_DIR, MINNAN_WORDS_DIR


@dataclass
class PreparedMinnan:
    """界面与引擎共用的转换结果；text 是实际交给语音模型的文本。"""
    text: str
    hanzi: str = ""
    tailo: str = ""
    matches: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    method: str = ""
    warnings: list = field(default_factory=list)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    return OpenCC("t2s")


def lookup_key(text: str) -> str:
    """仅为匹配建立简繁兼容键，实际台语输出保留来源字形和声调。"""
    return _converter().convert(normalize(text)).strip(" 。.!！?？,，;；:：\"“”")


class MinnanLexicon:
    """惰性加载可信对译与读音索引，保留一词多义，不随意选择第一条读音。"""

    def __init__(self, folder=MINNAN_WORDS_DIR):
        self.folder = Path(folder)
        self.mandarin = {}
        self.hanzi = {}
        self.loaded = False
        self._load_lock = threading.RLock()

    def load(self):
        with self._load_lock:
            if not self.loaded:
                self._load()

    def _load(self):
        if self.loaded:
            return
        source = self.folder / "entries.jsonl"
        if not source.is_file():
            raise FileNotFoundError("缺少台语词库，请先运行 tools/build_minnan_words.py")
        manifest_path = self.folder / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("缺少台语词库清单，请重新构建词库")
        from .minnan import digest
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))["outputs"]["entries"]
        if source.stat().st_size != metadata["bytes"] or digest(source) != metadata["sha256"]:
            raise ValueError("台语词库大小或 SHA-256 不符，请重新构建，不使用损坏内容")
        # 自定义项追加而不修改来源文件，方便后续升级和复核。
        for path in (source, self.folder / "user_entries.jsonl"):
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if path.name == "user_entries.jsonl":
                        # 用户明确维护的读法优先于来源候选，且不会被写回原始资料。
                        row = {**row, "_user_override": True}
                    if row.get("hanzi"):
                        self.hanzi.setdefault(lookup_key(row["hanzi"]), []).append(row)
                    for alias in row.get("mandarin", []):
                        if alias:
                            self.mandarin.setdefault(lookup_key(alias), []).append(row)
        self.loaded = True

    def find(self, text: str, language="mandarin") -> list[dict]:
        self.load()
        return (self.mandarin if language == "mandarin" else self.hanzi).get(lookup_key(text), [])

    @staticmethod
    def unique_reading(rows: list[dict]) -> str | None:
        preferred = [row for row in rows if row.get("_user_override")]
        rows = preferred or rows
        values = {normalize(row["tailo"]) for row in rows if row.get("tailo") and row.get("verified")}
        return next(iter(values)) if len(values) == 1 else None


@lru_cache(maxsize=8)
def _shared_lexicon(folder: str, modified: int, overrides_modified: int, manifest_modified: int) -> MinnanLexicon:
    """后台预览频繁创建处理器，共享同一版本索引，避免重复解析几十万词条。"""
    return MinnanLexicon(folder)


class LocalMinnanTranslator:
    """固定本地 Marian 翻译模型；在 CPU 串行执行，避免抢占语音合成显存。"""
    _lock = threading.RLock()
    _model = None
    _tokenizer = None

    @classmethod
    def unload(cls):
        """切换档位时释放不再需要的 CPU 权重，不要求用户重启窗口。"""
        with cls._lock:
            cls._model, cls._tokenizer = None, None
            gc.collect()

    @classmethod
    def load(cls):
        if cls._model is not None:
            return
        folder = MINNAN_DIR / "translation"
        manifest_path = folder / "download_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("缺少本地普通话转台语模型，请运行 tools/download_minnan_translation.py")
        from .minnan import digest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = folder / row["path"]
            path.resolve().relative_to(folder.resolve())
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"翻译模型文件损坏：{row['path']}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # 上游由 transformers 5 导出；4.52 中此字段要求 dict，显式兼容且不改原始文件。
        cls._tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, extra_special_tokens={})
        cls._model = AutoModelForSeq2SeqLM.from_pretrained(str(folder), local_files_only=True).eval()

    @classmethod
    def translate(cls, text: str) -> str:
        import torch
        with cls._lock:
            cls.load()
            # 模型训练以短句为主，按真实标点切分；任何片段超长都明确提示而不是截断原文。
            pieces = [part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text) if part.strip()]
            outputs = []
            for piece in pieces:
                batch = cls._tokenizer([">>nan<< " + piece], return_tensors="pt")
                if batch["input_ids"].shape[1] > 64:
                    raise ValueError("普通话单句过长，请增加句号或分号后转换，避免长句翻译失真。")
                with torch.inference_mode():
                    tokens = cls._model.generate(**batch, num_beams=4, max_new_tokens=192,
                                                 early_stopping=True, use_cache=True)
                output = normalize(cls._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
                if not output or "<unk>" in output or len(tokens[0]) >= 193:
                    raise ValueError("本地翻译未生成完整可靠的结果，请缩短句子或手动输入台语。")
                outputs.append(output)
            return " ".join(outputs)


class TaigiMinnanTranslator:
    """专用 Taigi-Llama-2 翻译器，采用作者规定的 HAN 提示格式，输出台语汉字。"""
    _lock = threading.RLock()
    _model = None

    @classmethod
    def unload(cls):
        """llama.cpp 使用显式 close 释放 mmap 与 KV 缓存，精简档不常驻完整模型。"""
        with cls._lock:
            if cls._model is not None:
                cls._model.close()
                cls._model = None

    @classmethod
    def load(cls):
        if cls._model is not None:
            return
        folder = MINNAN_DIR / "translation/taigi"
        manifest_path = folder / "download_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("缺少台语专用翻译模型，请运行 tools/download_minnan_translation.py --backend taigi")
        from .minnan import digest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = folder / row["path"]
            path.resolve().relative_to(folder.resolve())
            if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"台语专用翻译模型损坏：{row['path']}")
        weights = [row["path"] for row in manifest["files"] if row["path"].endswith(".gguf")]
        if len(weights) != 1:
            raise ValueError("翻译模型清单必须包含唯一 GGUF 权重")
        from llama_cpp import Llama
        cls._model = Llama(model_path=str(folder / weights[0]), n_ctx=1024, n_gpu_layers=0,
                           n_threads=min(6, os.cpu_count() or 4), n_threads_batch=min(8, os.cpu_count() or 4),
                           n_batch=128, use_mmap=True, seed=20261105, verbose=False)

    @classmethod
    def translate(cls, text: str) -> str:
        from opencc import OpenCC
        with cls._lock:
            cls.load()
            source = OpenCC("s2t").convert(text)
            pieces = [part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", source) if part.strip()]
            if not pieces:
                raise ValueError("没有可翻译的句子")
            outputs = []
            for piece in pieces:
                # 作者模型是专项翻译微调，不使用通用聊天模板，末尾换行不可省略。
                prompt = f"[TRANS]\n{piece}\n[/TRANS]\n[HAN]\n"
                if len(cls._model.tokenize(prompt.encode("utf-8"))) > 700:
                    raise ValueError("普通话单句过长，请分句后转换；原文未被截断。")
                result = cls._model(prompt, max_tokens=256, temperature=0.0, repeat_penalty=1.1,
                                    stop=["[/HAN]", "[/TRANS]", "[TRANS]"], echo=False)
                choice = result["choices"][0]
                output = normalize(choice["text"])
                if choice.get("finish_reason") == "length" or not output or re.search(r"\[(?:HAN|TRANS|POJ|ZH)\]", output):
                    raise ValueError("台语译文不完整，请缩短句子或手动输入台语。")
                outputs.append(output)
            return " ".join(outputs)


class MinnanTextProcessor:
    """普通话先对译再合成；台语汉字及台罗输入可直接使用，互不混淆。"""

    def __init__(self, words_dir=MINNAN_WORDS_DIR):
        folder = Path(words_dir)
        source, override, manifest = folder / "entries.jsonl", folder / "user_entries.jsonl", folder / "manifest.json"
        self.lexicon = _shared_lexicon(str(folder), source.stat().st_mtime_ns if source.is_file() else 0,
                                      override.stat().st_mtime_ns if override.is_file() else 0,
                                      manifest.stat().st_mtime_ns if manifest.is_file() else 0)

    def prepare(self, text: str, mode="mandarin", translation_mode="full") -> PreparedMinnan:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入要转换的文字")
        if "<|" in text or "|>" in text:
            raise ValueError("台语输入不能包含内部语言控制标记")
        text = normalize(text)
        if mode not in {"mandarin", "hanzi", "tailo"}:
            raise ValueError("台语输入方式必须为普通话、台语汉字或带声调台罗")
        if translation_mode not in {"full", "compact", "none"}:
            raise ValueError("翻译模型必须为完整、精简或不使用")
        if mode == "mandarin" and translation_mode == "none":
            TaigiMinnanTranslator.unload()
            LocalMinnanTranslator.unload()
        if mode == "tailo":
            if not re.search(r"[A-Za-zÀ-ž]", text):
                raise ValueError("台罗输入需要包含罗马字；汉字请使用“台语汉字”模式")
            return PreparedMinnan(text=text, tailo=text, method="tailo_input")
        rows = self.lexicon.find(text, "mandarin" if mode == "mandarin" else "hanzi")
        tailo = self.lexicon.unique_reading(rows)
        if tailo:
            # 精确对译必须有唯一已核验读音；歧义交由上下文模型或原汉字模型处理。
            hanzi = rows[0].get("hanzi", "")
            return PreparedMinnan(text=tailo, hanzi=hanzi, tailo=tailo, matches=rows, method="dictionary_exact")
        if mode == "hanzi":
            # 现有 LoRA 训练过台语汉字；无可信整句对照时保留完整输入，不拼凑猜读音。
            return PreparedMinnan(text=text, hanzi=text, matches=rows,
                                  unresolved=[text] if rows else [], method="hanzi_input",
                                  warnings=["未找到唯一整句台罗，使用台语汉字模型读音"])
        if translation_mode == "none":
            raise ValueError("当前不使用翻译模型，词库未找到唯一整句对译。请编辑台语结果，或改用台语汉字／带声调台罗。")
        if translation_mode == "compact":
            TaigiMinnanTranslator.unload()
            output = LocalMinnanTranslator.translate(text)
            if not output:
                raise ValueError("精简翻译模型未返回结果，未开始合成")
            return PreparedMinnan(text=output, tailo=output, method="compact_translation",
                                  warnings=["精简模型在实测中存在词义偏差，请检查译文后生成"])
        # 默认采用完整档；释放精简档权重，避免两个翻译模型无必要地同时驻留。
        LocalMinnanTranslator.unload()
        output = TaigiMinnanTranslator.translate(text)
        if not output:
            raise ValueError("普通话转台语失败，未开始合成")
        translated_rows = self.lexicon.find(output, "hanzi")
        reading = self.lexicon.unique_reading(translated_rows)
        return PreparedMinnan(text=reading or output, hanzi=output, tailo=reading or "",
                              matches=translated_rows,
                              method="taigi_translation", warnings=["机器翻译结果，专有名词和自然度请复核"])
