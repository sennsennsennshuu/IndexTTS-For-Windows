"""IndexTTS 2.5 台语训练的最小运行时：特征提取与 GPT 分阶段占用显存。"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import unicodedata


def configure() -> None:
    """仅使用训练项目内缓存；不调用 Windows 桌面入口改变当前工作目录。"""
    for variable, relative in {
        "HF_HOME": "cache/huggingface", "XDG_CACHE_HOME": "cache",
        "TMPDIR": "cache/tmp", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/matplotlib",
    }.items():
        Path(relative).mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(Path(relative).resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(Path("vendor").resolve()))


def canonical_text(text: str) -> str:
    """保留台罗声调和大小写，仅做 Unicode NFC 与空白规范化。"""
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


class MinnanTextFrontend:
    """台语专用纯文本前端：保留声调、原字形及标点，不使用普通话规则。"""

    def __init__(self):
        self.char_rep_map = {}
        # 推理入口会执行 clean_pattern；空匹配表必须使用永不匹配模式。
        self.clean_pattern = re.compile(r"(?!)")

    def load(self):
        """本前端只有 Unicode 规范化，不需要 FST 文件或网络资源。"""
        return self

    def normalize(self, text: str) -> str:
        return canonical_text(text)

    def load_glossary_from_yaml(self, path):
        raise ValueError("台语训练不接受中英文读音词汇表，请在来源标注中明确台罗")


def encode_text(tokenizer, text: str, maximum: int = 598) -> list[int]:
    """沿用已有 zh 前缀的 token ID，台语身份由独立语言嵌入表示。

    官方 tokenizer 未把 minnan 注册为特殊词。直接增加 num_languages 会改变后续
    已有特殊词的编号，因此这里不重排词表，也不把台语文字做普通话 G2P。
    """
    normalized = canonical_text(text)
    if not normalized:
        raise ValueError("文本为空")
    if "<|" in normalized or "|>" in normalized:
        raise ValueError("语料中包含未批准的控制标记")
    tokens = tokenizer.encode("<|zh|> " + normalized, allowed_special="all")
    if len(tokens) > maximum:
        raise ValueError(f"完整文本超出窗口：{len(tokens)} > {maximum}；禁止截断文本保留完整音频")
    return tokens


def load_gpt(model_dir: str = "models/IndexTTS-2.5", device: str = "cuda"):
    """严格加载固定 2.5 GPT；缺失权重时立即失败，不使用随机层继续训练。"""
    configure()
    import torch
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice

    config = OmegaConf.load(Path(model_dir) / "config.yaml")
    if str(config.version) != "2.5":
        raise ValueError("只接受 IndexTTS 2.5 基础模型")
    model = UnifiedVoice(**config.gpt, use_accel=False, spk_cond_mode="campplus")
    checkpoint = torch.load(Path(model_dir) / config.gpt_checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # 官方 checkpoint 可能包含推理包装器的重复引用。基础 GPT 的任何缺失都必须阻断。
    if missing:
        raise RuntimeError(f"GPT 基础权重缺失：{missing[:12]}")
    if unexpected:
        raise RuntimeError(f"GPT 存在未处理权重：{unexpected[:12]}")
    model = model.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.eval().requires_grad_(False)
    return model, config


class FeatureRuntime:
    """仅加载提取所需模型；不加载 S2Mel、BigVGAN 或 Qwen。"""

    def __init__(self, model_dir: str = "models/IndexTTS-2.5", device: str = "cuda"):
        configure()
        import torch
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
        from indextts.codec.models import EnhancedCodec
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

        self.device = device
        self.gpt, config = load_gpt(model_dir, device)
        auxiliary = Path(model_dir) / "hf_cache"
        self.extractor = SeamlessM4TFeatureExtractor.from_pretrained(str(auxiliary / "w2v-bert-2.0"), local_files_only=True)
        self.semantic = Wav2Vec2BertModel.from_pretrained(str(auxiliary / "w2v-bert-2.0"), local_files_only=True).to(device).eval().requires_grad_(False)
        statistics = torch.load(Path(model_dir) / config.w2v_stat, map_location=device, weights_only=True)
        self.mean, self.std = statistics["mean"], statistics["var"].sqrt()
        self.codec = EnhancedCodec(**config.semantic_codec, cfg=config.semantic_codec)
        codec_weights = torch.load(Path(model_dir) / "codec.pth", map_location="cpu", weights_only=True)["model"]
        self.codec.load_state_dict(codec_weights, strict=True)
        self.codec = self.codec.to(device).eval().requires_grad_(False)
        self.speaker = CAMPPlus(feat_dim=80, embedding_size=192)
        self.speaker.load_state_dict(torch.load(auxiliary / "campplus_cn_common.bin", map_location="cpu", weights_only=True), strict=True)
        self.speaker = self.speaker.to(device).eval().requires_grad_(False)

    def extract(self, path: str, expected_duration: float | None = None) -> dict:
        """完整解码录音并生成离散目标及参考条件；不静默裁切长音频。"""
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(samples).all():
            raise ValueError(f"音频含无穷值或 NaN：{path}")
        duration = len(samples) / sample_rate
        if not 0.6 <= duration <= 18.0:
            raise ValueError(f"音频时长不符合当前完整句训练范围：{duration:.3f}s")
        if expected_duration is not None and abs(duration - expected_duration) > 0.15:
            raise ValueError(f"清单与音频时长不一致：{path}")
        waveform = torch.from_numpy(samples.mean(axis=1).copy()).unsqueeze(0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        with torch.inference_mode():
            inputs = self.extractor(waveform, sampling_rate=16000, return_tensors="pt")
            semantic = self.semantic(
                input_features=inputs["input_features"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                output_hidden_states=True,
            ).hidden_states[17]
            semantic = (semantic - self.mean) / self.std
            codes, _ = self.codec.quantize(semantic)
            codes = codes.reshape(1, -1)
            if not 1 <= codes.shape[1] <= 1810 or int(codes.min()) < 0 or int(codes.max()) >= 8192:
                raise ValueError(f"无效语义 token：{path}")
            features = torchaudio.compliance.kaldi.fbank(waveform.to(self.device), num_mel_bins=80, dither=0, sample_frequency=16000)
            features -= features.mean(dim=0, keepdim=True)
            speaker = self.speaker(features.unsqueeze(0))
            lengths = torch.tensor([semantic.shape[1]], device=self.device)
            emotion = self.gpt.get_emovec(semantic.to(self.gpt.text_embedding.weight.dtype), lengths)
        # 保存至 CPU，不保存与目标内容高度相关的完整语义隐藏序列作为参考条件。
        return {"codes": codes[0].to(device="cpu", dtype=torch.int16),
                "speaker": speaker[0].float().cpu(), "emotion": emotion[0].float().cpu(),
                "duration": duration, "schema": 1}
