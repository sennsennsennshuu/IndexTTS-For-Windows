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


MODEL_DIR = Path("models/IndexTTS-2")
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")


def validate_models():
    """启动前检查所有实际使用的权重；缺少文件时直接给出本地相对路径。"""
    required = [
        "config.yaml", "gpt.pth", "s2mel.pth", "bpe.model",
        "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt",
        "qwen_0.6b_emo4_merge/config.json",
        "qwen_0.6b_emo4_merge/model.safetensors",
        "qwen_0.6b_emo4_merge/tokenizer_config.json",
        "qwen_0.6b_emo4_merge/tokenizer.json",
        "checkpoints/hf_cache/models--facebook--w2v-bert-2.0/config.json",
        "checkpoints/hf_cache/models--facebook--w2v-bert-2.0/model.safetensors",
        "checkpoints/hf_cache/models--facebook--w2v-bert-2.0/preprocessor_config.json",
        "checkpoints/hf_cache/models--amphion--MaskGCT/semantic_codec/model.safetensors",
        "checkpoints/hf_cache/models--funasr--campplus/campplus_cn_common.bin",
        "checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/config.json",
        "checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt",
    ]
    missing = [str(MODEL_DIR / name) for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少项目内模型文件：\n" + "\n".join(missing))
    return len(required)
