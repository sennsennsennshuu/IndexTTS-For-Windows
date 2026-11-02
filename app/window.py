"""IndexTTS 桌面工作台：文本、声音设置与生成结果拥有清晰的操作层级。"""
from pathlib import Path
import logging
import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QDesktopServices, QPalette, QColor
from PyQt6.QtWidgets import (
    QDialog, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QTextEdit, QLineEdit, QPushButton, QComboBox, QButtonGroup, QSlider,
    QSpinBox, QDoubleSpinBox, QFileDialog, QScrollArea,
    QAbstractSpinBox, QCheckBox, QProgressBar, QSizePolicy,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from .engine import IndexTTSEngine
from .paths import MODEL_DIR, OUTPUT_DIR, validate_models
from .worker import IndexTTSWorker
from .theme import STYLESHEET
from .widgets import Card, AudioDrop, Waveform, StudioSlider, icon, label


class IndexTTSWindow(QDialog):
    """双栏深色工作台；模型与后台线程继续使用独立项目原有实现。"""
    audio_generated = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = IndexTTSEngine()
        self.worker = None
        self.player = None
        self.audio_output = None
        self._close_when_finished = False
        self._result_path = None
        self._duration_ms = 0
        self.setObjectName("studio")
        self.setWindowTitle("IndexTTS 语音合成")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.WindowCloseButtonHint)
        self.setFont(QFont("Microsoft YaHei UI", 9))
        self.setMinimumSize(1000, 700)
        self.resize(1180, 820)
        # 对输入框和系统弹窗也设置明确调色板，避免 Windows 系统主题穿透。
        palette = self.palette()
        for role, color in ((QPalette.ColorRole.Window, "#191c21"),
                            (QPalette.ColorRole.Base, "#13161a"),
                            (QPalette.ColorRole.Text, "#e5e8ed"),
                            (QPalette.ColorRole.WindowText, "#e5e8ed"),
                            (QPalette.ColorRole.PlaceholderText, "#788493")):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.setStyleSheet(STYLESHEET)
        self.init_ui()

    def init_ui(self):
        """左侧用于创作和试听，右侧集中放置声音设置。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 0)
        root.setSpacing(20)
        self._build_header(root)
        workspace = QHBoxLayout()
        workspace.setSpacing(20)
        left = QVBoxLayout()
        left.setSpacing(18)
        left.addWidget(self._build_editor(), 5)
        left.addWidget(self._build_result(), 3)
        workspace.addLayout(left, 1)
        # 右栏独立滚动，低分辨率或展开高级参数时不挤压文本与主操作。
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(352)
        scroll.setMaximumWidth(386)
        side = QWidget()
        settings = QVBoxLayout(side)
        settings.setContentsMargins(0, 0, 7, 0)
        settings.setSpacing(14)
        settings.addWidget(self._build_voice())
        settings.addWidget(self._build_emotion())
        settings.addWidget(self._build_advanced())
        settings.addStretch()
        scroll.setWidget(side)
        workspace.addWidget(scroll)
        root.addLayout(workspace, 1)
        self._build_footer(root)
        self.on_emo_method_changed(0)
        self._update_text_count()

    def _build_header(self, root):
        header = QHBoxLayout()
        header.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(icon("brand", "#adbed3", 30).pixmap(30, 30))
        header.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        name = QHBoxLayout()
        name.setSpacing(14)
        name.addWidget(label("IndexTTS", "brand"))
        name.addWidget(label("VOICE STUDIO", "brandTag"))
        name.addStretch()
        brand.addLayout(name)
        brand.addWidget(label("语音创作工作台  /  让表达自然发生", "subtitle"))
        header.addLayout(brand, 1)
        header.addWidget(label("本地运行 · 2.0", "pill"))
        root.addLayout(header)

    def _build_editor(self):
        card = Card("合成文本", "01")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        clear = QPushButton("清空")
        clear.setObjectName("ghost")
        clear.clicked.connect(lambda: self.text_input.clear())
        card.heading.addWidget(clear)
        self.text_input = QTextEdit()
        self.text_input.setAcceptRichText(False)
        self.text_input.setPlaceholderText("在这里写下想说的话。\n\n一段旁白，一句对白，或一个新的声音。")
        self.text_input.setMinimumHeight(170)
        self.text_input.textChanged.connect(self._update_text_count)
        card.body.addWidget(self.text_input, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(label("自然停顿，从标点开始。", "quiet"))
        bottom.addStretch()
        self.char_count = label("0 字", "muted")
        bottom.addWidget(self.char_count)
        card.body.addLayout(bottom)
        return card

    def _build_result(self):
        card = Card("生成结果", "02")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.result_badge = label("等待生成", "quiet")
        card.heading.addWidget(self.result_badge)
        self.waveform = Waveform()
        self.waveform.seek_requested.connect(self._seek_to)
        card.body.addWidget(self.waveform, 1)
        controls = QHBoxLayout()
        controls.setSpacing(12)
        self.play_btn = QPushButton()
        self.play_btn.setObjectName("play")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setIcon(icon("play", "#17212e", 18))
        self.play_btn.setEnabled(False)
        self.play_btn.setToolTip("播放 / 暂停")
        self.play_btn.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_btn)
        info = QVBoxLayout()
        info.setSpacing(4)
        self.result_name = label("尚未生成音频", "fileTitle")
        self.result_name.setMinimumWidth(0)
        info.addWidget(self.result_name)
        self.time_label = label("00:00 / 00:00", "timecode")
        info.addWidget(self.time_label)
        controls.addLayout(info, 1)
        self.output_btn = QPushButton("输出目录")
        self.output_btn.setObjectName("ghost")
        self.output_btn.setIcon(icon("folder", size=16))
        self.output_btn.clicked.connect(self._open_output_folder)
        controls.addWidget(self.output_btn)
        card.body.addLayout(controls)
        return card

    def _build_voice(self):
        card = Card("声音参考")
        self.speaker_drop = AudioDrop()
        self.speaker_drop.browse_requested.connect(self.browse_speaker_audio)
        self.speaker_drop.file_selected.connect(self._set_speaker_audio)
        # 保留文本路径对象作为原工作流的数据接口，完整路径通过提示显示。
        self.speaker_audio_input = QLineEdit(self)
        self.speaker_audio_input.hide()
        self.speaker_audio_input.textChanged.connect(self.speaker_drop.set_file)
        card.body.addWidget(self.speaker_drop)
        return card

    def _build_emotion(self):
        card = Card("情感表达")
        self.emo_method_combo = QComboBox(self)
        self.emo_method_combo.addItems(["跟随音色", "参考音频", "情感向量", "文字描述"])
        self.emo_method_combo.hide()
        self.emo_method_combo.currentIndexChanged.connect(self.on_emo_method_changed)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = []
        modes = QGridLayout()
        modes.setSpacing(7)
        for index, text in enumerate(("跟随音色", "参考音频", "情感向量", "文字描述")):
            button = QPushButton(text)
            button.setProperty("mode", True)
            button.setCheckable(True)
            self.mode_group.addButton(button, index)
            self.mode_buttons.append(button)
            modes.addWidget(button, index // 2, index % 2)
        self.mode_group.idClicked.connect(self.emo_method_combo.setCurrentIndex)
        card.body.addLayout(modes)
        self.emo_description = label("沿用参考音频中的语气，自然还原声音。", "muted")
        self.emo_description.setWordWrap(True)
        card.body.addWidget(self.emo_description)
        self.emo_audio_widget = QWidget()
        audio = QHBoxLayout(self.emo_audio_widget)
        audio.setContentsMargins(0, 0, 0, 0)
        audio.setSpacing(6)
        self.emo_audio_input = QLineEdit()
        self.emo_audio_input.setPlaceholderText("选择情感参考音频")
        self.emo_audio_input.setMinimumWidth(0)
        audio.addWidget(self.emo_audio_input, 1)
        choose = QPushButton("选择")
        choose.clicked.connect(self.browse_emo_audio)
        audio.addWidget(choose)
        card.body.addWidget(self.emo_audio_widget)
        self.emo_text_widget = QWidget()
        text = QVBoxLayout(self.emo_text_widget)
        text.setContentsMargins(0, 0, 0, 0)
        self.emo_text_input = QLineEdit()
        self.emo_text_input.setPlaceholderText("例如：平静地讲述，带一点期待")
        text.addWidget(self.emo_text_input)
        card.body.addWidget(self.emo_text_widget)
        self.emo_vector_widget = QWidget()
        vector = QGridLayout(self.emo_vector_widget)
        vector.setContentsMargins(0, 0, 0, 0)
        vector.setVerticalSpacing(10)
        vector.setHorizontalSpacing(12)
        self.emo_sliders = []
        for index, text in enumerate(("喜悦", "愤怒", "悲伤", "恐惧", "厌恶", "低落", "惊讶", "平静")):
            cell = QVBoxLayout()
            cell.setSpacing(4)
            heading = QHBoxLayout()
            heading.addWidget(label(text, "muted"))
            heading.addStretch()
            value = label("0.00", "timecode")
            heading.addWidget(value)
            cell.addLayout(heading)
            slider = StudioSlider()
            slider.setRange(0, 100)
            slider.setMinimumWidth(0)
            slider.setAccessibleName(text)
            slider.valueChanged.connect(lambda v, target=value: target.setText(f"{v / 100:.2f}"))
            cell.addWidget(slider)
            vector.addLayout(cell, index // 2, index % 2)
            self.emo_sliders.append((slider, value))
        card.body.addWidget(self.emo_vector_widget)
        self.emo_weight_widget = QWidget()
        weight = QVBoxLayout(self.emo_weight_widget)
        weight.setContentsMargins(0, 0, 0, 0)
        weight.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(label("情感强度", "muted"))
        heading.addStretch()
        self.emo_weight_label = label("0.65", "timecode")
        heading.addWidget(self.emo_weight_label)
        weight.addLayout(heading)
        self.emo_weight_slider = StudioSlider()
        self.emo_weight_slider.setRange(0, 100)
        self.emo_weight_slider.setValue(65)
        self.emo_weight_slider.valueChanged.connect(lambda v: self.emo_weight_label.setText(f"{v / 100:.2f}"))
        weight.addWidget(self.emo_weight_slider)
        card.body.addWidget(self.emo_weight_widget)
        return card

    def _build_advanced(self):
        frame = QFrame()
        frame.setObjectName("card")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(14)
        self.advanced_toggle = QPushButton("高级参数")
        self.advanced_toggle.setObjectName("fold")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        fold = QHBoxLayout()
        fold.addWidget(self.advanced_toggle, 1)
        self.advanced_indicator = label("+", "quiet")
        fold.addWidget(self.advanced_indicator)
        outer.addLayout(fold)
        self.advanced_content = QWidget()
        grid = QGridLayout(self.advanced_content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        specs = (("采样温度", "temp_spin", 0.1, 2.0, 0.8, 0.1),
                 ("Top P", "top_p_spin", 0.01, 1.0, 0.8, 0.01),
                 ("Top K", "top_k_spin", 0, 100, 30, 1),
                 ("分段 Token 上限", "max_tokens_spin", 20, 500, 120, 1))
        for index, (text, attr, minimum, maximum, value, step) in enumerate(specs):
            column = QVBoxLayout()
            column.setSpacing(6)
            column.addWidget(label(text, "muted"))
            spin = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setSingleStep(step)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(0)
            column.addWidget(spin)
            setattr(self, attr, spin)
            grid.addLayout(column, index // 2, index % 2)
        outer.addWidget(self.advanced_content)
        self.advanced_content.hide()
        return frame

    def _build_footer(self, root):
        footer = QFrame()
        footer.setObjectName("footer")
        box = QVBoxLayout(footer)
        box.setContentsMargins(0, 14, 0, 18)
        box.setSpacing(9)
        row = QHBoxLayout()
        status = QVBoxLayout()
        status.setSpacing(4)
        self.status_label = label("准备就绪", "status")
        status.addWidget(self.status_label)
        self.status_hint = label("添加参考音频与文本，开始一次新的创作。", "quiet")
        status.addWidget(self.status_hint)
        row.addLayout(status, 1)
        self.auto_play = QCheckBox("完成后自动播放")
        self.auto_play.setChecked(True)
        row.addWidget(self.auto_play)
        row.addSpacing(20)
        self.generate_btn = QPushButton("生成语音")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.setIcon(icon("spark", "#17212e", 18))
        self.generate_btn.setIconSize(QSize(18, 18))
        self.generate_btn.setMinimumWidth(170)
        self.generate_btn.clicked.connect(self.generate_speech)
        row.addWidget(self.generate_btn)
        box.addLayout(row)
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setTextVisible(False)
        self.busy_indicator.hide()
        box.addWidget(self.busy_indicator)
        root.addWidget(footer)

    def _update_text_count(self):
        if hasattr(self, "char_count"):
            self.char_count.setText(f"{len(self.text_input.toPlainText())} 字")

    def _toggle_advanced(self, expanded):
        self.advanced_content.setVisible(expanded)
        self.advanced_indicator.setText("−" if expanded else "+")

    def _set_speaker_audio(self, path):
        self.speaker_audio_input.setText(path)

    def get_indextts_model_path(self):
        validate_models()
        return str(MODEL_DIR)

    def browse_speaker_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音色参考音频", "", "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if path:
            self._set_speaker_audio(path)

    def browse_emo_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择情感参考音频", "", "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if path:
            self.emo_audio_input.setText(path)
            self.emo_audio_input.setToolTip(path)

    def on_emo_method_changed(self, index):
        """模式选择按钮与原参数索引同步；只显示当前模式需要的设置。"""
        self.mode_buttons[index].setChecked(True)
        self.emo_audio_widget.setVisible(index == 1)
        self.emo_vector_widget.setVisible(index == 2)
        self.emo_text_widget.setVisible(index == 3)
        self.emo_weight_widget.setVisible(index in (1, 2, 3))
        self.emo_description.setText(("沿用参考音频中的语气，自然还原声音。", "从另一段音频中借用情绪与表达。",
                                      "微调八种情绪，让声音更贴近你的想象。", "用一句话描述你想要的情绪与语气。")[index])

    def collect_emotion_params(self):
        mode = self.emo_method_combo.currentIndex()
        params = {"emo_mode": mode, "emo_weight": self.emo_weight_slider.value() / 100.0}
        if mode == 1:
            params["emo_audio"] = self.emo_audio_input.text().strip()
        elif mode == 2:
            params["emo_vector"] = [slider.value() / 100.0 for slider, _ in self.emo_sliders]
        elif mode == 3:
            params["emo_text"] = self.emo_text_input.text().strip() or None
        return params

    def generate_speech(self):
        """保留原异步推理入口，先检查所有必要输入再占用模型资源。"""
        if self.worker is not None:
            return
        text = self.text_input.toPlainText().strip()
        speaker = self.speaker_audio_input.text().strip()
        if not text:
            self._show_input_error("请先输入要合成的文本。", self.text_input)
            return
        if not speaker or not Path(speaker).is_file():
            self._show_input_error("请先选择有效的音色参考音频。")
            return
        emotion = self.collect_emotion_params()
        if emotion["emo_mode"] == 1 and not Path(emotion["emo_audio"]).is_file():
            self._show_input_error("请选择有效的情感参考音频。", self.emo_audio_input)
            return
        try:
            self.get_indextts_model_path()
        except FileNotFoundError as exc:
            self._show_input_error(str(exc))
            return
        advanced = {"temperature": self.temp_spin.value(), "top_p": self.top_p_spin.value(),
                    "top_k": self.top_k_spin.value(), "max_text_tokens_per_segment": self.max_tokens_spin.value()}
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("正在生成…")
        self.busy_indicator.show()
        self.update_status("准备生成…")
        self.status_hint.setText("首次生成需要加载模型，后续任务会复用已加载的模型。")
        self.worker = IndexTTSWorker(text, speaker, emotion, advanced, self.engine)
        self.worker.progress_update.connect(self.update_status)
        self.worker.generation_finished.connect(self.on_generation_finished)
        self.worker.generation_error.connect(self.on_generation_error)
        self.worker.finished.connect(self.cleanup_worker)
        self.worker.start()

    def _show_input_error(self, message, focus=None):
        self.update_status(message, "error")
        if focus is not None:
            focus.setFocus()

    def update_status(self, text, state="ready"):
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def on_generation_finished(self, path):
        # 保存任务文本后再处理媒体回调，避免完成信号清理 worker 后读取失效对象。
        text = self.worker.text[:100] if self.worker else self.text_input.toPlainText()[:100]
        if not self._close_when_finished:
            self.play_generated_audio(path)
        self.update_status("语音已生成", "done")
        self.status_hint.setText("音频已保存。可以继续调整文本，或试听当前结果。")
        self.audio_generated.emit(path, text)

    def on_generation_error(self, message):
        logging.getLogger(__name__).error("窗口收到合成错误：%s", message)
        self.update_status("生成未完成，请检查输入或日志。", "error")
        self.status_hint.setText(message[:100])
        self.status_hint.setToolTip(message)

    def cleanup_worker(self):
        """线程彻底结束后才恢复操作；延迟关闭时继续沿用安全释放流程。"""
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("生成语音")
        self.busy_indicator.hide()
        if self._close_when_finished:
            self.close()

    def play_generated_audio(self, path):
        """结果直接展示在工作台，提供可重复试听的播放器，避免完成弹窗打断编辑。"""
        file = Path(path)
        if not file.is_file():
            self._show_input_error(f"音频文件不存在：{path}")
            return
        self._result_path = file
        from PyQt6.QtGui import QFontMetrics
        self.result_name.setText(QFontMetrics(self.result_name.font()).elidedText(file.name, Qt.TextElideMode.ElideMiddle, 320))
        self.result_name.setToolTip(str(file.resolve()))
        try:
            self.waveform.load_audio(file)
            import soundfile as sf
            info = sf.info(str(file))
            self._duration_ms = int(info.duration * 1000)
            self.result_badge.setText(f"{info.samplerate / 1000:g} kHz · WAV")
        except Exception:
            logging.getLogger(__name__).exception("波形预览读取失败")
            self.result_badge.setText("音频已保存")
        self._update_position(0)
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self._update_position)
            self.player.durationChanged.connect(self._update_duration)
            self.player.playbackStateChanged.connect(self._update_play_icon)
            self.player.errorOccurred.connect(self._playback_error)
        self.player.setSource(QUrl.fromLocalFile(str(file.resolve())))
        self.play_btn.setEnabled(True)
        if self.auto_play.isChecked():
            self.player.play()

    @staticmethod
    def _format_time(milliseconds):
        total = max(0, milliseconds // 1000)
        return f"{total // 60:02d}:{total % 60:02d}"

    def _update_duration(self, duration):
        self._duration_ms = duration
        self._update_position(self.player.position() if self.player else 0)

    def _update_position(self, position):
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(self._duration_ms)}")
        self.waveform.set_progress(position / self._duration_ms if self._duration_ms else 0)

    def _update_play_icon(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setIcon(icon("pause" if playing else "play", "#17212e", 18))

    def _toggle_playback(self):
        if self.player:
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                self.player.play()

    def _seek_to(self, ratio):
        if self.player:
            self.player.setPosition(int(ratio * self._duration_ms))

    def _playback_error(self, error, message):
        self.update_status("音频已保存，播放器暂时无法播放。", "error")
        self.status_hint.setText(message[:100])

    def _open_output_folder(self):
        folder = self._result_path.parent if self._result_path else OUTPUT_DIR
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def showEvent(self, event):
        """Windows 原生标题栏也采用深色；不改变平台标准窗口操作。"""
        super().showEvent(event)
        if os.name == "nt":
            try:
                import ctypes
                enabled = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), 20, ctypes.byref(enabled), 4)
            except Exception:
                logging.getLogger(__name__).debug("当前系统不支持深色标题栏", exc_info=True)

    def closeEvent(self, event):
        if self.worker is not None:
            self._close_when_finished = True
            self.update_status("正在完成当前任务，结束后自动关闭…")
            event.ignore()
            return
        self.engine.unload_model()
        if self.player:
            self.player.stop()
        event.accept()
