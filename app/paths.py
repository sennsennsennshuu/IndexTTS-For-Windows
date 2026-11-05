"""统一使用项目相对路径，避免依赖原助手目录或用户级模型缓存。"""
from pathlib import Path
import os
import sys


def prepare_runtime():
    """入口先固定工作目录，之后所有读写路径均相对于本项目。"""
    os.chdir(Path(__file__).resolve().parent.parent)
    for folder in ("cache", "cache/huggingface", "cache/torch", "cache/tmp", "outputs", "logs"):
        Path(folder).mkdir(parents=True, exist_ok=True)
    # 强制隔离外部 Python 配置，防止用户的 PYTHONPATH 或模型缓存污染项目。
    os.environ.update({
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_HOME": "cache/torch",
        "MPLCONFIGDIR": "cache/matplotlib",
        "MODELSCOPE_CACHE": "cache/modelscope",
        "NUMBA_CACHE_DIR": "cache/numba",
        "TEMP": str(Path("cache/tmp").resolve()),
        "TMP": str(Path("cache/tmp").resolve()),
    })
    # vendor 是本项目复制的源码，优先级高于环境中可能存在的同名包。
    vendor = str(Path("vendor").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


MODEL_DIR = Path("models/IndexTTS-2.5")
MODEL_VERSION = "2.5"
APP_VERSION = "1.4.0"
LANGUAGES = {"ZH": "中文", "EN": "English", "JA": "日本語", "ES": "Español", "AR": "العربية"}
LANGUAGES["ZH-NAN-TW"] = "台语（台湾闽南语）"
MINNAN_DIR = Path("models/extra_language/zh-nan-tw")
MINNAN_MODEL_DIR = MINNAN_DIR / "model"
MINNAN_WORDS_DIR = MINNAN_DIR / "words"
VOICE_LIBRARY_DIR = Path("models/voice_library")
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")


def validate_models(model_dir=None, require_emotion=False):
    """按当前模式检查本地模型；文字情感权重不阻塞其他三种模式。"""
    import yaml
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    required = [
        "config.yaml", "gpt.pth", "s2mel.pth", "codec.pth",
        "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "hf_cache/w2v-bert-2.0/config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors",
        "hf_cache/w2v-bert-2.0/preprocessor_config.json",
        "hf_cache/campplus_cn_common.bin",
        "hf_cache/bigvgan/config.json", "hf_cache/bigvgan/bigvgan_generator.pt",
    ]
    if require_emotion:
        required += ["qwen0.6bemo4-merge/config.json", "qwen0.6bemo4-merge/model.safetensors",
                     "qwen0.6bemo4-merge/tokenizer_config.json", "qwen0.6bemo4-merge/tokenizer.json"]
    missing = [str(model_dir / name) for name in required
               if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("缺少项目内模型文件：\n" + "\n".join(missing) +
                                "\n请按 docs/MODEL_SETUP.md 准备 IndexTTS 2.5 权重。")
    config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or str(config.get("version")) != MODEL_VERSION:
        raise ValueError("模型配置不是 IndexTTS 2.5，请勿混用其他版本的权重。")
    # Git LFS 指针不是权重，提前提供可读错误而不是反序列化异常。
    for name in required:
        path = model_dir / name
        if path.suffix in {".pth", ".pt", ".safetensors", ".bin"} and path.stat().st_size < 256:
            raise ValueError(f"模型文件不完整或仍是 Git LFS 指针：{path}")
    return len(required)
