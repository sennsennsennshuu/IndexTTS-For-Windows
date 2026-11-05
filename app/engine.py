"""IndexTTS 2.5 推理封装：复用模型、按需情感加载、原子保存音频。"""
from datetime import datetime
from pathlib import Path
import gc
import math
import threading
from uuid import uuid4

from .paths import MODEL_DIR, MODEL_VERSION, LANGUAGES, OUTPUT_DIR, validate_models


class IndexTTSEngine:
    """同一窗口复用一个引擎，用锁阻止并发访问模型及其参考音频缓存。"""

    def __init__(self, model_dir=MODEL_DIR, use_bf16=True):
        self.model_dir = str(model_dir)
        self.model_version = MODEL_VERSION
        self.use_bf16 = use_bf16
        self.tts_model = None
        self.is_loaded = False
        self._lock = threading.RLock()
        self._reference_signatures = None
        self.progress_callback = None
        self.text_prepared_callback = None
        self._minnan_adapter = None
        self._minnan_processor = None

    def load_model(self):
        """第一次生成时加载模型，普通 CUDA 运算无需本地编译扩展。"""
        with self._lock:
            if self.is_loaded:
                return
            validate_models(self.model_dir)
            import torch
            from indextts.infer_v2_5 import IndexTTS2
            self.tts_model = IndexTTS2(
                model_dir=self.model_dir,
                cfg_path=str(Path(self.model_dir) / "config.yaml"),
                # 新版采用 BF16；旧显卡不支持时自动采用 FP32。
                use_bf16=self.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                use_deepspeed=False,
                use_cuda_kernel=False,
                use_accel=False, use_torch_compile=False, use_qwen_emo=False,
            )
            self.is_loaded = True

    def _ensure_emotion_model(self):
        """首次文字情感请求才在 CPU 加载，避免常驻分类器占用合成显存。"""
        if self.tts_model.qwen_emo is None:
            validate_models(self.model_dir, require_emotion=True)
            from indextts.infer_v2_5 import QwenEmotion
            self.tts_model.qwen_emo = QwenEmotion(
                str(Path(self.model_dir) / self.tts_model.cfg.qwen_emo_path), device="cpu")

    @staticmethod
    def _file_signature(path):
        """同路径参考音频被覆盖时，也必须更新条件缓存。"""
        file = Path(path)
        stat = file.stat()
        return (str(file.resolve()), stat.st_mtime_ns, stat.st_size)

    def infer(self, text, speaker_audio, output_path=None, emo_mode=0,
              emo_audio=None, emo_weight=0.65, emo_vector=None, emo_text=None,
              lang="ZH", duration_factor=1.0, minnan_input_mode="mandarin",
              minnan_translation_mode="full", **kwargs):
        """先校验输入，再转发五语言、时长与情感参数；发音标注保持原样。"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入要合成的文本")
        if not speaker_audio or not Path(speaker_audio).is_file():
            raise FileNotFoundError(f"参考音频不存在：{speaker_audio}")
        if emo_mode not in range(4):
            raise ValueError("情感模式必须为 0、1、2 或 3")
        lang = str(lang).upper()
        if lang not in LANGUAGES:
            raise ValueError("请选择界面中支持的合成语言")
        if not isinstance(duration_factor, (int, float)) or not math.isfinite(duration_factor) or not 0.5 <= duration_factor <= 2.0:
            raise ValueError("时长系数必须介于 0.5 和 2.0 之间；小于 1 更快，大于 1 更慢")
        if emo_mode == 1 and (not emo_audio or not Path(emo_audio).is_file()):
            raise ValueError("请选择有效的情感参考音频")
        if emo_mode == 2 and (emo_vector is None or len(emo_vector) != 8):
            raise ValueError("情感向量必须包含 8 个数值")
        if emo_mode == 2 and any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in emo_vector):
            raise ValueError("情感向量的每个值必须为 0 到 1 之间的有限数值")
        if not isinstance(emo_weight, (int, float)) or not math.isfinite(emo_weight) or not 0 <= emo_weight <= 1:
            raise ValueError("情感权重必须介于 0 和 1 之间")
        with self._lock:
            is_minnan = lang == "ZH-NAN-TW"
            if is_minnan:
                from .minnan_words import MinnanTextProcessor
                if self.progress_callback:
                    self.progress_callback("正在转换台语文字与读音…")
                if self._minnan_processor is None:
                    self._minnan_processor = MinnanTextProcessor()
                prepared = self._minnan_processor.prepare(text, mode=minnan_input_mode,
                                                          translation_mode=minnan_translation_mode)
                text = prepared.text
                if self.text_prepared_callback:
                    self.text_prepared_callback(text)
            if emo_mode == 3:
                validate_models(self.model_dir, require_emotion=True)
            self.load_model()
            # 只有使用台语功能时才创建语言包控制器，原五语言无需加载额外权重。
            if is_minnan and self._minnan_adapter is None:
                from .minnan import MinnanAdapter
                self._minnan_adapter = MinnanAdapter(self.tts_model, self.model_dir)
            if emo_mode == 3:
                if self.progress_callback:
                    self.progress_callback("正在准备文字情感…")
                self._ensure_emotion_model()
            if self.progress_callback:
                self.progress_callback("正在生成音频…")
            signatures = (self._file_signature(speaker_audio),
                          self._file_signature(emo_audio) if emo_mode == 1 else None)
            if self._reference_signatures is not None and signatures != self._reference_signatures:
                for attr in ("cache_spk_cond", "cache_spk_audio_prompt", "cache_emo_cond", "cache_emo_audio_prompt"):
                    setattr(self.tts_model, attr, None)
            self._reference_signatures = signatures
            output = Path(output_path) if output_path else OUTPUT_DIR / (
                f"tts_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:6]}.wav")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}_{uuid4().hex[:8]}.partial.wav")
            # 保留原界面的情感偏置与总量限制；权重缩放仍由原版 infer 处理。
            if emo_mode == 2:
                emo_vector = self.tts_model.normalize_emo_vec(emo_vector, apply_bias=True)
            try:
                from contextlib import nullcontext
                language_context = self._minnan_adapter.activate(is_minnan) if self._minnan_adapter else nullcontext()
                if is_minnan:
                    kwargs["text_normalization"] = False
                with language_context:
                    self.tts_model.infer(
                        spk_audio_prompt=str(speaker_audio), text=text, output_path=str(temporary),
                        lang="minnan" if is_minnan else lang, duration_factor=duration_factor,
                        emo_audio_prompt=str(emo_audio) if emo_mode == 1 else None,
                        emo_alpha=emo_weight if emo_mode else 1.0,
                        emo_vector=emo_vector if emo_mode == 2 else None,
                        use_emo_text=emo_mode == 3,
                        emo_text=emo_text if emo_mode == 3 else None,
                        verbose=False, **kwargs,
                    )
                if not temporary.is_file() or temporary.stat().st_size <= 44:
                    raise RuntimeError("推理结束，但没有生成有效的音频文件")
                import soundfile as sf
                if sf.info(str(temporary)).frames == 0:
                    raise RuntimeError("生成的音频没有有效采样")
                temporary.replace(output)
            finally:
                # 合成失败不留下半成品，也不破坏同名的已有结果。
                temporary.unlink(missing_ok=True)
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
            self._reference_signatures = None
            self._minnan_adapter = None
            self._minnan_processor = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
