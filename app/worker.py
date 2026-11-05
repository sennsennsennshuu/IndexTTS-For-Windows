"""在工作线程内执行加载、合成与台语文本准备，并将完整异常记录到项目日志。"""
import logging
from PyQt6.QtCore import QThread, pyqtSignal


class IndexTTSWorker(QThread):
    generation_finished = pyqtSignal(str)
    generation_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    # 保留文本准备结果的显式信号，便于后续将引擎侧准备进度呈现到界面。
    text_prepared = pyqtSignal(str)

    def __init__(self, text, speaker_audio, emo_params, advanced_params, engine):
        super().__init__()
        # text 保存本次实际送入引擎的文本；界面完成回调不能读取已被替换的输入框。
        self.text = text
        self.speaker_audio = speaker_audio
        self.emo_params = emo_params
        self.advanced_params = advanced_params
        self.engine = engine

    def run(self):
        """窗口始终持有线程及引擎；完成后才允许下一次生成或关闭。"""
        try:
            self.engine.progress_callback = self.progress_update.emit
            # 引擎准备台语文本后通过该回调返回最终送入 TTS 的文本；
            # 信号跨线程投递到窗口线程，避免生成线程直接操作 Qt 控件。
            self.engine.text_prepared_callback = self.text_prepared.emit
            self.progress_update.emit("正在加载模型..." if not self.engine.is_loaded else "正在生成音频...")
            # infer 先验证当前模式，再加载模型；缺失权重时无需先占用显存。
            path = self.engine.infer(self.text, self.speaker_audio,
                                     **self.emo_params, **self.advanced_params)
            self.generation_finished.emit(path)
        except Exception as exc:
            logging.getLogger(__name__).exception("语音合成失败")
            self.generation_error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # 回调绑定本次线程；任务结束后必须解除，防止后续请求误用已结束线程。
            self.engine.progress_callback = None
            self.engine.text_prepared_callback = None


class MinnanPrepareWorker(QThread):
    """在独立线程准备台语预览，避免词库处理阻塞 Qt 事件循环。"""

    text_prepared = pyqtSignal(str)
    preparation_ready = pyqtSignal(object)
    preparation_failed = pyqtSignal(str)

    def __init__(self, text, mode, request_id=0, translation_mode="full"):
        super().__init__()
        self.text = text
        self.mode = mode
        self.request_id = request_id
        # 默认完整档保持旧调用兼容；每次预览都显式携带档位，避免跨档复用。
        self.translation_mode = translation_mode

    def run(self):
        """延迟导入核心 API，使未使用台语的启动路径不加载词库依赖。"""
        try:
            from .minnan_words import MinnanTextProcessor
            prepared = MinnanTextProcessor().prepare(
                self.text, mode=self.mode, translation_mode=self.translation_mode)
            result_text = str(getattr(prepared, "text", "") or "").strip()
            if not result_text:
                raise ValueError("台语转译没有返回可合成文本")
            # 先发文本再发详情，界面可立即显示结果，同时保持状态更新顺序确定。
            self.text_prepared.emit(result_text)
            self.preparation_ready.emit(prepared)
        except Exception as exc:
            logging.getLogger(__name__).exception("台语文本准备失败")
            self.preparation_failed.emit(f"{type(exc).__name__}: {exc}")
