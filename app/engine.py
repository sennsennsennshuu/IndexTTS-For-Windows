"""独立的推理封装：复用已加载模型，输出统一保存到 outputs/。"""
from datetime import datetime
from pathlib import Path
import gc
import threading
from uuid import uuid4

from .paths import MODEL_DIR, OUTPUT_DIR, validate_models


class IndexTTSEngine:
    """同一窗口复用一个引擎，用锁阻止并发访问模型及其参考音频缓存。"""

    def __init__(self, model_dir=MODEL_DIR, use_fp16=True):
        self.model_dir = str(model_dir)
        self.use_fp16 = use_fp16
        self.tts_model = None
        self.is_loaded = False
        self._lock = threading.RLock()

    def load_model(self):
        """第一次生成时加载模型，普通 CUDA 运算无需本地编译扩展。"""
        with self._lock:
            if self.is_loaded:
                return
            validate_models()
            import torch
            from indextts.infer_v2 import IndexTTS2
            self.tts_model = IndexTTS2(
                model_dir=self.model_dir,
                cfg_path=str(Path(self.model_dir) / "config.yaml"),
                use_fp16=self.use_fp16 and torch.cuda.is_available(),
                use_deepspeed=False,
                use_cuda_kernel=False,
            )
            self.is_loaded = True

    def infer(self, text, speaker_audio, output_path=None, emo_mode=0,
              emo_audio=None, emo_weight=0.65, emo_vector=None, emo_text=None, **kwargs):
        """校验输入并转发原版四种情感模式；文件名包含微秒及随机后缀以防覆盖。"""
        if not text.strip():
            raise ValueError("请输入要合成的文本")
        if not Path(speaker_audio).is_file():
            raise FileNotFoundError(f"参考音频不存在：{speaker_audio}")
        if emo_mode not in range(4):
            raise ValueError("情感模式必须为 0、1、2 或 3")
        if emo_mode == 1 and (not emo_audio or not Path(emo_audio).is_file()):
            raise ValueError("请选择有效的情感参考音频")
        if emo_mode == 2 and (emo_vector is None or len(emo_vector) != 8):
            raise ValueError("情感向量必须包含 8 个数值")
        if not 0 <= emo_weight <= 1:
            raise ValueError("情感权重必须介于 0 和 1 之间")
        with self._lock:
            self.load_model()
            output = Path(output_path) if output_path else OUTPUT_DIR / (
                f"tts_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:6]}.wav")
            output.parent.mkdir(parents=True, exist_ok=True)
            # 保留原界面的情感偏置与总量限制；权重缩放仍由原版 infer 处理。
            if emo_mode == 2:
                emo_vector = self.tts_model.normalize_emo_vec(emo_vector, apply_bias=True)
            self.tts_model.infer(
                spk_audio_prompt=str(speaker_audio), text=text, output_path=str(output),
                emo_audio_prompt=emo_audio if emo_mode == 1 else None,
                emo_alpha=emo_weight if emo_mode else 1.0,
                emo_vector=emo_vector if emo_mode == 2 else None,
                use_emo_text=emo_mode == 3,
                emo_text=emo_text if emo_mode == 3 else None,
                verbose=False, **kwargs,
            )
            if not output.is_file() or output.stat().st_size <= 44:
                raise RuntimeError("推理结束，但没有生成有效的音频文件")
            return str(output)

    def unload_model(self):
        """仅在线程结束后释放模型及显存。"""
        with self._lock:
            # 还未生成过音频时直接关闭，避免仅为释放空模型而首次导入 PyTorch。
            if self.tts_model is None:
                self.is_loaded = False
                return
            self.tts_model = None
            self.is_loaded = False
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
