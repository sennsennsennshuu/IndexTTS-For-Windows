"""在工作线程内执行加载与合成，并将完整异常记录到项目日志。"""
import logging
from PyQt6.QtCore import QThread, pyqtSignal


class IndexTTSWorker(QThread):
    generation_finished = pyqtSignal(str)
    generation_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, text, speaker_audio, emo_params, advanced_params, engine):
        super().__init__()
        self.text = text
        self.speaker_audio = speaker_audio
        self.emo_params = emo_params
        self.advanced_params = advanced_params
        self.engine = engine

    def run(self):
        """窗口始终持有线程及引擎；完成后才允许下一次生成或关闭。"""
        try:
            self.progress_update.emit("正在加载模型..." if not self.engine.is_loaded else "正在生成音频...")
            self.engine.load_model()
            self.progress_update.emit("正在生成音频...")
            path = self.engine.infer(self.text, self.speaker_audio,
                                     **self.emo_params, **self.advanced_params)
            self.generation_finished.emit(path)
        except Exception as exc:
            logging.getLogger(__name__).exception("语音合成失败")
            self.generation_error.emit(f"{type(exc).__name__}: {exc}")
