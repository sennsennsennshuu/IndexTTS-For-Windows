"""IndexTTS 桌面工作台：文本、声音设置与生成结果拥有清晰的操作层级。"""
from pathlib import Path
import logging
import os

from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QDesktopServices, QPalette, QColor
from PyQt6.QtWidgets import (
    QDialog, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QTextEdit, QLineEdit, QPushButton, QComboBox, QButtonGroup, QSlider,
    QSpinBox, QDoubleSpinBox, QFileDialog, QScrollArea,
    QAbstractSpinBox, QCheckBox, QProgressBar, QSizePolicy,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from .engine import IndexTTSEngine
from .paths import MODEL_DIR, MODEL_VERSION, LANGUAGES, OUTPUT_DIR, validate_models
from .worker import IndexTTSWorker, MinnanPrepareWorker
from .theme import STYLESHEET
from .widgets import Card, AudioDrop, Waveform, StudioSlider, icon, label
from .voice_panel import VoiceLibraryPanel


# 台语语言代码由 paths.LANGUAGES 提供；界面只负责识别并呈现对应输入方式。
MINNAN_LANGUAGE_CODE = "ZH-NAN-TW"
MINNAN_INPUT_MODES = (
    ("mandarin", "普通话（自动转台语，推荐）"),
    ("hanzi", "台语汉字"),
    ("tailo", "带声调台罗"),
)
MINNAN_TRANSLATION_MODES = (
    ("full", "完整", "完整翻译模型（7B）：质量优先，需要更多磁盘和运行资源。"),
    ("compact", "精简", "精简翻译模型（约 313 MB）：速度更快，资源占用更低。"),
    ("none", "不使用", "仅使用词库精确对译；未命中会报错，不会把普通话原文当作台语。"),
)


class IndexTTSWindow(QDialog):
    """双栏深色工作台；模型与后台线程继续使用独立项目原有实现。"""
    audio_generated = pyqtSignal(str, str)

    def __init__(self, parent=None, *, voice_library=None):
        super().__init__(parent)
        self._voice_library_store = voice_library
        self.engine = IndexTTSEngine()
        self.worker = None
        self.player = None
        self.audio_output = None
        self._close_when_finished = False
        self._result_path = None
        self._duration_ms = 0
        # 台语预览使用独立线程；同一时间只保留一个任务，输入变化时合并到下一次。
        self.minnan_preview_worker = None
        self._pending_minnan_preview = None
        self._minnan_request_id = 0
        self._minnan_preview_source = ""
        self._minnan_preview_mode = "mandarin"
        self._minnan_preview_translation_mode = "full"
        self._minnan_preview_ready = False
        self._minnan_unresolved = []
        self._minnan_preview_edited = False
        self._setting_minnan_preview = False
        self._minnan_preview_timer = QTimer(self)
        self._minnan_preview_timer.setSingleShot(True)
        self._minnan_preview_timer.setInterval(350)
        self._minnan_preview_timer.timeout.connect(self._start_minnan_preview)
        self.setObjectName("studio")
        self.setWindowTitle("IndexTTS 2.5 语音合成")
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
        self.settings_panel = side
        settings = QVBoxLayout(side)
        settings.setContentsMargins(0, 0, 7, 0)
        settings.setSpacing(14)
        settings.addWidget(self._build_speech_controls())
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
        # 初始化完成后统一同步语言显隐，避免默认五语言时闪现台语区域。
        self._on_language_changed()
        self.voice_library_panel.restore_selection()

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
        name.addStretch()
        brand.addLayout(name)
        brand.addWidget(label("zeroshot 语音合成", "subtitle"))
        header.addLayout(brand, 1)
        header.addWidget(label(f"本地运行 · {MODEL_VERSION}", "pill"))
        root.addLayout(header)

    def _build_editor(self):
        card = Card("合成文本", "01")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        clear = QPushButton("清空")
        clear.setObjectName("ghost")
        clear.clicked.connect(lambda: self.text_input.clear())
        card.heading.addWidget(clear)
        # 台语预览与发音帮助展开后允许正文滚动，避免内容最小高度将普通窗口撑出屏幕。
        # 卡片标题保持固定，左侧结果播放器和底部生成按钮也始终留在窗口内。
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor_content = QWidget()
        editor_body = QVBoxLayout(editor_content)
        editor_body.setContentsMargins(0, 0, 5, 0)
        editor_body.setSpacing(12)
        self.editor_scroll.setWidget(editor_content)
        card.body.addWidget(self.editor_scroll, 1)
        self.text_input = QTextEdit()
        self.text_input.setAcceptRichText(False)
        self.text_input.setPlaceholderText("在这里写下想说的话。\n\n一段旁白，一句对白，或一个新的声音。")
        self.text_input.setMinimumHeight(170)
        self.text_input.textChanged.connect(self._update_text_count)
        editor_body.addWidget(self.text_input, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(label("自然停顿，从标点开始。", "quiet"))
        bottom.addStretch()
        self.char_count = label("0 字", "muted")
        bottom.addWidget(self.char_count)
        editor_body.addLayout(bottom)
        self.pronunciation_toggle = QPushButton("发音标注帮助")
        self.pronunciation_toggle.setObjectName("ghost")
        self.pronunciation_toggle.setCheckable(True)
        editor_body.addWidget(self.pronunciation_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.pronunciation_help = label(
            "中文：银<行|HANG2>　英文：<minute|M IH1 . N AH0 T>\n"
            "日语：<上手|じょうず>　用 <文字|读音> 指定发音。", "quiet")
        self.pronunciation_help.setTextFormat(Qt.TextFormat.PlainText)
        self.pronunciation_help.setWordWrap(True)
        self.pronunciation_help.hide()
        self.pronunciation_toggle.toggled.connect(self.pronunciation_help.setVisible)
        editor_body.addWidget(self.pronunciation_help)

        # 台语专属区域默认隐藏；只有选择台语时才占用编辑区空间。
        self.minnan_panel = QWidget()
        self.minnan_panel.setObjectName("minnanPanel")
        minnan_layout = QVBoxLayout(self.minnan_panel)
        minnan_layout.setContentsMargins(0, 4, 0, 0)
        minnan_layout.setSpacing(8)

        # 每个字段采用上方标签、下方选择框，避免普通窗口下四个控件横向争抢空间。
        # 两列按 2:1 分配剩余宽度，输入方式保留更宽的完整文本显示区域。
        self.minnan_controls = QWidget()
        self.minnan_controls_grid = QGridLayout(self.minnan_controls)
        self.minnan_controls_grid.setContentsMargins(0, 0, 0, 0)
        self.minnan_controls_grid.setHorizontalSpacing(14)
        self.minnan_controls_grid.setVerticalSpacing(6)
        self.minnan_controls_grid.setColumnStretch(0, 2)
        self.minnan_controls_grid.setColumnStretch(1, 1)
        self.minnan_mode_label = label("台语输入方式", "muted")
        self.minnan_controls_grid.addWidget(self.minnan_mode_label, 0, 0)
        self.minnan_mode_combo = QComboBox()
        for value, title in MINNAN_INPUT_MODES:
            self.minnan_mode_combo.addItem(title, value)
        self.minnan_mode_combo.setAccessibleName("台语输入方式")
        self.minnan_mode_combo.setToolTip(
            "普通话会自动转成台语；也可以直接输入台语汉字或带声调台罗。")
        self.minnan_mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.minnan_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.minnan_mode_combo.setMinimumContentsLength(1)
        self.minnan_mode_combo.setMinimumWidth(0)
        self.minnan_mode_combo.currentIndexChanged.connect(self._on_minnan_input_mode_changed)
        self.minnan_controls_grid.addWidget(self.minnan_mode_combo, 1, 0)

        self.minnan_translation_label = label("翻译模型", "muted")
        self.minnan_controls_grid.addWidget(self.minnan_translation_label, 0, 1)
        self.minnan_translation_combo = QComboBox()
        self.minnan_translation_combo.setObjectName("minnanTranslationCombo")
        self.minnan_translation_combo.setAccessibleName("翻译模型")
        self.minnan_translation_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.minnan_translation_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.minnan_translation_combo.setMinimumContentsLength(1)
        self.minnan_translation_combo.setMinimumWidth(0)
        for value, title, tooltip in MINNAN_TRANSLATION_MODES:
            self.minnan_translation_combo.addItem(title, value)
            self.minnan_translation_combo.setItemData(
                self.minnan_translation_combo.count() - 1,
                tooltip,
                Qt.ItemDataRole.ToolTipRole)
        self.minnan_translation_combo.setToolTip(
            "完整：7B 模型；精简：约 313 MB；不使用：仅词库精确对译。")
        self.minnan_translation_combo.currentIndexChanged.connect(
            self._on_minnan_translation_mode_changed)
        self.minnan_controls_grid.addWidget(self.minnan_translation_combo, 1, 1)
        minnan_layout.addWidget(self.minnan_controls)

        preview_heading = QHBoxLayout()
        preview_heading.setSpacing(8)
        preview_heading.addWidget(label("台语转换结果", "muted"))
        preview_heading.addStretch()
        preview_heading.addWidget(label("可直接编辑", "quiet"))
        minnan_layout.addLayout(preview_heading)

        self.minnan_preview = QTextEdit()
        self.minnan_preview.setAcceptRichText(False)
        self.minnan_preview.setPlaceholderText("转换后的台语文本会显示在这里；也可以直接修改。")
        self.minnan_preview.setMinimumHeight(72)
        self.minnan_preview.setMaximumHeight(118)
        self.minnan_preview.textChanged.connect(self._on_minnan_preview_edited)
        minnan_layout.addWidget(self.minnan_preview)

        self.minnan_status = label("输入普通话后，会自动准备台语文本。", "status")
        self.minnan_status.setWordWrap(True)
        minnan_layout.addWidget(self.minnan_status)
        self.minnan_status_hint = label(
            "未命中的词会明确列出，不会把普通话原文当作已译台语。", "quiet")
        self.minnan_status_hint.setWordWrap(True)
        minnan_layout.addWidget(self.minnan_status_hint)

        self.minnan_panel.hide()
        editor_body.addWidget(self.minnan_panel)
        return card

    def _build_speech_controls(self):
        """2.5 原生控制；时长与速度互为倒数，避免把系数方向解释反。"""
        card = Card("语言与节奏")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(7)
        grid.addWidget(label("合成语言", "muted"), 0, 0)
        grid.addWidget(label("时长系数", "muted"), 0, 1)
        self.language_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self.language_combo.addItem(name, code)
        self.language_combo.setAccessibleName("合成语言")
        # 语言切换只控制台语区域的显隐与预览任务，不改变其他五种语言的参数。
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 2.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.duration_spin.setSingleStep(0.05)
        self.duration_spin.setValue(1.0)
        self.duration_spin.setSuffix(" ×")
        self.duration_spin.setAccessibleName("时长系数")
        self.duration_spin.setToolTip("1.00 为正常时长；0.50 更快，2.00 更慢。由模型生成时控制节奏。")
        grid.addWidget(self.language_combo, 1, 0)
        grid.addWidget(self.duration_spin, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        card.body.addLayout(grid)
        hint = label("小于 1 更快 · 大于 1 更慢", "quiet")
        hint.setWordWrap(True)
        card.body.addWidget(hint)
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
        card.body.addWidget(self.speaker_drop)
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        card.body.addWidget(divider)
        # 音色库保存项目内参考副本，通过同一接口交给所有语言的语音引擎。
        self.voice_library_panel = VoiceLibraryPanel(self._voice_library_store, card)
        self.voice_library_panel.reference_selected.connect(self._set_speaker_audio)
        self.speaker_audio_input.textChanged.connect(self._sync_speaker_reference)
        card.body.addWidget(self.voice_library_panel)
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
        # 台语模式下文本变化会重新准备预览；其他语言不启动任何转译线程。
        if hasattr(self, "minnan_panel") and self._is_minnan_language():
            self._schedule_minnan_preview()

    def _is_minnan_language(self):
        """按语言代码识别台语；兼容下划线写法，避免配置细节泄漏到界面。"""
        if not hasattr(self, "language_combo"):
            return False
        code = str(self.language_combo.currentData() or "").strip().upper().replace("_", "-")
        return code == MINNAN_LANGUAGE_CODE

    def _set_minnan_status(self, text, state="ready"):
        """使用现有状态标签样式，让成功、警告和异常在深色主题下都清晰可见。"""
        self.minnan_status.setText(text)
        self.minnan_status.setProperty("state", state)
        self.minnan_status.style().unpolish(self.minnan_status)
        self.minnan_status.style().polish(self.minnan_status)

    def _set_minnan_preview_text(self, text):
        """程序写入预览时屏蔽 textChanged，避免把自动结果误判为手动修改。"""
        self._setting_minnan_preview = True
        try:
            self.minnan_preview.setPlainText(text)
        finally:
            self._setting_minnan_preview = False

    def _reset_minnan_preview_state(self):
        """切换语言或输入方式时清空旧结果，防止上一轮文本被误当成当前译文。"""
        self._minnan_preview_ready = False
        self._minnan_unresolved = []
        self._minnan_preview_edited = False
        self._minnan_preview_source = ""
        self._minnan_preview_mode = self.minnan_mode_combo.currentData() or "mandarin"
        self._minnan_preview_translation_mode = self._current_translation_mode()
        self._set_minnan_preview_text("")
        self._set_minnan_status("输入普通话后，会自动准备台语文本。", "ready")
        self.minnan_status_hint.setText(
            "请核对词义、声调与专有名词；未命中项会明确列出。")

    def _on_language_changed(self, *args):
        """台语区域仅在选中台语时显示，并立即刷新一次预览。"""
        if not hasattr(self, "minnan_panel"):
            return
        if not self._is_minnan_language():
            self.minnan_panel.hide()
            self._cancel_minnan_preview()
            return
        self.minnan_panel.show()
        self._update_minnan_translation_visibility()
        self._reset_minnan_preview_state()
        self._schedule_minnan_preview()

    def _current_translation_mode(self):
        """直接汉字/台罗模式不使用翻译器；普通话模式读取用户选择。"""
        input_mode = self.minnan_mode_combo.currentData() or "mandarin"
        if input_mode != "mandarin":
            return "none"
        return self.minnan_translation_combo.currentData() or "full"

    def _update_minnan_translation_visibility(self):
        """翻译模型仅对普通话自动转换有意义，隐藏时将空间归还输入方式字段。"""
        visible = (self.minnan_mode_combo.currentData() or "mandarin") == "mandarin"
        self.minnan_translation_label.setVisible(visible)
        self.minnan_translation_combo.setVisible(visible)
        self.minnan_controls_grid.setColumnStretch(1, 1 if visible else 0)

    def _on_minnan_input_mode_changed(self, *args):
        """输入方式变化会切换翻译可用性，并废弃上一档的在途结果。"""
        self._update_minnan_translation_visibility()
        if self._is_minnan_language():
            self._cancel_minnan_preview()
            self._schedule_minnan_preview()

    def _on_minnan_translation_mode_changed(self, *args):
        """切换模型档位时立即失效旧请求，禁止回用另一档译文。"""
        if not self._is_minnan_language() or self._current_translation_mode() != self.minnan_translation_combo.currentData():
            return
        self._cancel_minnan_preview()
        self._schedule_minnan_preview()

    def _cancel_minnan_preview(self):
        """使在途结果失效；线程本身由 finished 信号安全回收。"""
        self._minnan_preview_timer.stop()
        self._minnan_request_id += 1
        self._pending_minnan_preview = None

    def _schedule_minnan_preview(self, *args):
        """防抖并合并输入：线程忙时只保留最新请求，避免堆积 QThread。"""
        if not hasattr(self, "minnan_panel") or not self._is_minnan_language():
            return
        if self._close_when_finished or self.worker is not None:
            return
        source = self.text_input.toPlainText().strip()
        mode = self.minnan_mode_combo.currentData() or "mandarin"
        translation_mode = self._current_translation_mode()
        # 预览线程已在处理上一版输入时，不再启动防抖计时器；否则计时器结束后
        # 因线程忙碌而直接返回，会留下一个永远不触发的请求。
        self._minnan_preview_timer.stop()
        self._minnan_request_id += 1
        request_id = self._minnan_request_id
        self._minnan_preview_ready = False
        self._minnan_unresolved = []
        self._minnan_preview_edited = False
        self._minnan_preview_source = source
        self._minnan_preview_mode = mode
        self._minnan_preview_translation_mode = translation_mode
        if not source:
            self._pending_minnan_preview = None
            self._set_minnan_preview_text("")
            self._set_minnan_status("等待输入普通话文本。", "ready")
            self.minnan_status_hint.setText("输入后会自动转成台语，也可以切换到台语汉字或台罗。")
            return
        # 旧结果在新输入下不再可信，先清空，避免用户把过期译文当作当前结果。
        self._set_minnan_preview_text("")
        self._pending_minnan_preview = (request_id, source, mode, translation_mode)
        self._set_minnan_status("正在准备台语文本…", "ready")
        self.minnan_status_hint.setText("正在本地查询词库与转换文字；完成后可直接编辑结果。")
        if self.minnan_preview_worker is None:
            self._minnan_preview_timer.start()
        # 线程忙时不启动计时器；cleanup_minnan_preview_worker 会立即续跑最新请求。

    def _start_minnan_preview(self):
        """在独立线程创建处理器并准备文本，绝不在 Qt 主线程执行词库逻辑。"""
        if self._close_when_finished or not self._is_minnan_language():
            return
        if self.minnan_preview_worker is not None or not self._pending_minnan_preview:
            return
        request_id, source, mode, translation_mode = self._pending_minnan_preview
        self._pending_minnan_preview = None
        worker = MinnanPrepareWorker(
            source, mode, request_id=request_id, translation_mode=translation_mode)
        self.minnan_preview_worker = worker
        worker.text_prepared.connect(
            lambda prepared, rid=request_id: self._on_minnan_prepared(rid, prepared))
        worker.preparation_ready.connect(
            lambda details, rid=request_id: self._on_minnan_details(rid, details))
        worker.preparation_failed.connect(
            lambda message, rid=request_id: self._on_minnan_failed(rid, message))
        worker.finished.connect(self.cleanup_minnan_preview_worker)
        worker.start()

    def _on_minnan_prepared(self, request_id, prepared):
        """只更新台语结果区；上方用户输入保持不变。"""
        if request_id != self._minnan_request_id or not self._is_minnan_language():
            return
        self._minnan_preview_ready = True
        self._minnan_preview_source = self.text_input.toPlainText().strip()
        self._minnan_preview_mode = self.minnan_mode_combo.currentData() or "mandarin"
        self._minnan_preview_translation_mode = self._current_translation_mode()
        if self._minnan_preview_edited:
            self._set_minnan_status("转换已完成，但保留了你的手动修改。", "done")
            self.minnan_status_hint.setText("请检查后生成；生成将使用你编辑后的台语文本。")
            return
        self._set_minnan_preview_text(prepared)
        self._set_minnan_status("自动译文已准备，请检查后生成。", "done")
        self.minnan_status_hint.setText("可直接修改结果区；确认词义与声调后再生成语音。")

    @staticmethod
    def _format_minnan_items(items, limit=8):
        """把未命中项压缩成用户能读懂的短提示，不暴露内部对象结构。"""
        labels = []
        for item in items or []:
            if isinstance(item, str):
                label_text = item
            elif isinstance(item, dict):
                label_text = (item.get("text") or item.get("token") or
                              item.get("word") or str(item))
            else:
                label_text = str(item)
            label_text = str(label_text).strip()
            if label_text:
                labels.append(label_text)
        shown = labels[:limit]
        suffix = "…" if len(labels) > limit else ""
        return "、".join(shown) + suffix

    def _on_minnan_details(self, request_id, details):
        """根据匹配与未命中结果给出明确状态，未命中时使用错误色提示。"""
        if request_id != self._minnan_request_id or not self._is_minnan_language():
            return
        unresolved = list(getattr(details, "unresolved", None) or [])
        matches = getattr(details, "matches", None) or []
        warnings = list(getattr(details, "warnings", None) or [])
        self._minnan_unresolved = unresolved
        if unresolved:
            names = self._format_minnan_items(unresolved)
            self._set_minnan_status(f"自动译文已准备，但有 {len(unresolved)} 处未命中 · 请检查。", "error")
            self.minnan_status_hint.setText(
                f"未命中：{names}。请检查并编辑结果，或改用台语汉字／带声调台罗。")
            return
        match_count = len(matches) if hasattr(matches, "__len__") else 0
        if match_count:
            self._set_minnan_status(
                f"台语文本已准备 · 找到 {match_count} 条来源对照 · 请检查", "done")
        else:
            self._set_minnan_status("自动译文已准备，请检查后使用。", "done")
        self.minnan_status_hint.setText(
            "；".join(warnings) if warnings else "请继续核对词义、声调与专有名词。")

    def _on_minnan_failed(self, request_id, message):
        """异常必须可见；失败时清空自动结果，绝不保留普通话冒充译文。"""
        if request_id != self._minnan_request_id or not self._is_minnan_language():
            return
        self._minnan_preview_ready = False
        self._minnan_unresolved = []
        if not self._minnan_preview_edited:
            self._set_minnan_preview_text("")
            self._set_minnan_status("台语转译暂不可用。", "error")
        else:
            self._set_minnan_status("自动转译失败，保留了你编辑的台语文本。", "error")
        self.minnan_status_hint.setText(message[:180])
        logging.getLogger(__name__).error("台语转译失败：%s", message)

    def _on_minnan_preview_edited(self):
        """用户手动修改后立即生效；后续迟到的自动结果不会覆盖这段文本。"""
        if self._setting_minnan_preview or not self._is_minnan_language():
            return
        edited = self.minnan_preview.toPlainText().strip()
        if edited:
            self._minnan_preview_edited = True
            self._set_minnan_status("已手动修改转换结果。", "done")
            self.minnan_status_hint.setText("生成时将使用这段台语文本；请确认声调与用字。")
        else:
            self._minnan_preview_edited = False
            self._set_minnan_status("转换结果为空。", "ready")
            self.minnan_status_hint.setText("可重新输入原文，或切换到其他台语输入方式。")

    @staticmethod
    def _detect_minnan_text_mode(text, fallback):
        """编辑后的文本按字符形态选择输入方式，避免再次自动翻译。"""
        if any("㐀" <= char <= "鿿" for char in text):
            return "hanzi"
        if any(char.isalpha() for char in text):
            return "tailo"
        return fallback

    def _effective_minnan_input(self, source):
        """转换结果有内容时直达引擎；汉字结果标记 hanzi，纯台罗标记 tailo。"""
        mode = self.minnan_mode_combo.currentData() or "mandarin"
        translation_mode = self._current_translation_mode()
        preview = self.minnan_preview.toPlainText().strip()
        current = self.text_input.toPlainText().strip()
        # 自动结果、原文、输入方式和翻译档位必须全部一致才可复用；
        # 任一变化都回退到重新准备，禁止混用另一档模型的旧译文。
        if (preview and self._minnan_preview_source == current and
                self._minnan_preview_mode == mode and
                self._minnan_preview_translation_mode == translation_mode):
            return preview, self._detect_minnan_text_mode(preview, mode)
        return source, mode

    def collect_minnan_params(self):
        """收集台语生成参数；非台语返回空字典，保证原五语言调用不变。"""
        if not self._is_minnan_language():
            return {}
        text = self.text_input.toPlainText().strip()
        prepared_text, mode = self._effective_minnan_input(text)
        # 预览区有可用文本时，生成必须直接使用它，避免引擎再次翻译。
        # 预览为空时返回原文与所选方式，由引擎统一准备并在回调中报告最终文本。
        return {
            "minnan_input_mode": mode,
            "minnan_text": prepared_text,
            "minnan_translation_mode": self._current_translation_mode(),
        }

    def cleanup_minnan_preview_worker(self):
        """预览线程结束后回收对象；若有更新请求则只启动最新一轮。"""
        worker = self.minnan_preview_worker
        if worker:
            worker.deleteLater()
            self.minnan_preview_worker = None
        if self._close_when_finished:
            self._close_if_ready()
            return
        if self._is_minnan_language() and self._pending_minnan_preview:
            QTimer.singleShot(0, self._start_minnan_preview)

    def _close_if_ready(self):
        """合成线程和转译线程都结束后才真正关闭窗口。"""
        if (self._close_when_finished and self.worker is None and
                self.minnan_preview_worker is None):
            self.close()

    def _toggle_advanced(self, expanded):
        self.advanced_content.setVisible(expanded)
        self.advanced_indicator.setText("−" if expanded else "+")

    def _set_speaker_audio(self, path):
        self.speaker_audio_input.setText(path)

    def _sync_speaker_reference(self, path):
        """参考路径、库内选择及展示名称同步，用户无需辨认内部文件编号。"""
        self.voice_library_panel.set_reference(path)
        self.speaker_drop.set_file(path, self.voice_library_panel.reference_name)

    def get_indextts_model_path(self):
        validate_models(require_emotion=self.emo_method_combo.currentIndex() == 3)
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
        except (FileNotFoundError, ValueError) as exc:
            self._show_input_error(str(exc))
            return
        advanced = {"temperature": self.temp_spin.value(), "top_p": self.top_p_spin.value(),
                    "top_k": self.top_k_spin.value(), "max_text_tokens_per_segment": self.max_tokens_spin.value(),
                    "lang": self.language_combo.currentData(), "duration_factor": self.duration_spin.value()}
        if self._is_minnan_language():
            # 先转换、检查、再生成：普通话语义失败时不能静默把原文当台语。
            mode = self.minnan_mode_combo.currentData() or "mandarin"
            preview = self.minnan_preview.toPlainText().strip()
            if mode == "mandarin" and not preview:
                self._show_input_error(
                    "请先等待台语转换完成，或手动填写台语转换结果。",
                    self.minnan_preview)
                return
            # 有转换结果时直接送已准备文本；否则由引擎按所选方式准备。
            text, minnan_mode = self._effective_minnan_input(text)
            advanced["minnan_input_mode"] = minnan_mode
            advanced["minnan_translation_mode"] = self._current_translation_mode()
            # 生成开始后停止预览防抖与旧任务，避免与引擎准备结果竞争。
            self._cancel_minnan_preview()
            self._minnan_preview_source = self.text_input.toPlainText().strip()
            self._minnan_preview_mode = minnan_mode
            self._minnan_preview_translation_mode = self._current_translation_mode()
        self.settings_panel.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("正在生成…")
        self.busy_indicator.show()
        self.update_status("准备生成…")
        self.status_hint.setText("首次生成需要加载模型，后续任务会复用已加载的模型。")
        self.worker = IndexTTSWorker(text, speaker, emotion, advanced, self.engine)
        self.worker.progress_update.connect(self.update_status)
        self.worker.text_prepared.connect(self.on_text_prepared)
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

    def on_text_prepared(self, text):
        """接收引擎最终准备文本；只刷新结果区，绝不覆盖用户原始输入。"""
        if not self._is_minnan_language():
            return
        prepared = str(text or "").strip()
        if not prepared:
            return
        # 用户已编辑时尊重当前文本；引擎回调只更新状态，不覆盖手动内容。
        if not self._minnan_preview_edited:
            self._set_minnan_preview_text(prepared)
        self._minnan_preview_source = self.text_input.toPlainText().strip()
        self._minnan_preview_mode = self._detect_minnan_text_mode(
            self.minnan_preview.toPlainText().strip() or prepared, "tailo")
        self._minnan_preview_translation_mode = self._current_translation_mode()
        self._set_minnan_status("生成引擎已准备台语文本。", "done")
        self.minnan_status_hint.setText("本次生成使用上方结果区文本；自动译文仍需人工核对。")

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
        self.settings_panel.setEnabled(True)
        self.generate_btn.setText("生成语音")
        self.busy_indicator.hide()
        if self._close_when_finished:
            self._close_if_ready()

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
        # 任一后台线程仍在运行时延迟关闭，防止 QThread 被窗口析构时强制终止。
        if self.worker is not None or self.minnan_preview_worker is not None:
            self._close_when_finished = True
            self._cancel_minnan_preview()
            self.update_status("正在结束后台任务，完成后自动关闭…")
            event.ignore()
            return
        self.engine.unload_model()
        if self.player:
            self.player.stop()
        event.accept()
