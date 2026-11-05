"""声音参考中的音色库区域：名称保存、即选即用和受限删除。"""
from pathlib import Path
import logging

from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox

from .voice_library import VoiceLibrary, VoiceLibraryError
from .widgets import label


class VoiceLibraryPanel(QWidget):
    """只管理参考录音预设；通过信号复用窗口已有的声音参考和合成接口。"""
    reference_selected = pyqtSignal(str)

    def __init__(self, library=None, parent=None):
        super().__init__(parent)
        self.library = library if library is not None else VoiceLibrary()
        self._reference = ""
        self._presets = ()
        self._restored_id = None
        self._pending_delete = None
        self._available = False
        self.setObjectName("voiceLibrary")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        heading = QHBoxLayout()
        heading.addWidget(label("音色库", "fileTitle"))
        heading.addStretch()
        self.count_label = label("0 个音色", "quiet")
        heading.addWidget(self.count_label)
        layout.addLayout(heading)

        selection = QHBoxLayout()
        selection.setSpacing(8)
        self.combo = QComboBox()
        self.combo.setAccessibleName("已保存的音色")
        self.combo.setMinimumWidth(0)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(8)
        self.combo.setToolTip("选择后立即用于声音参考；音色可配合任一已支持语言使用。")
        self.combo.currentIndexChanged.connect(self._activate)
        selection.addWidget(self.combo, 1)
        self.delete_button = QPushButton("删除")
        self.delete_button.setObjectName("ghost")
        self.delete_button.clicked.connect(self._ask_delete)
        selection.addWidget(self.delete_button)
        layout.addLayout(selection)

        saving = QHBoxLayout()
        saving.setSpacing(8)
        self.name_input = QLineEdit()
        self.name_input.setMaxLength(48)
        self.name_input.setPlaceholderText("为当前参考录音命名")
        self.name_input.setAccessibleName("音色名称")
        self.name_input.installEventFilter(self)
        self.name_input.textChanged.connect(self._update_buttons)
        self.name_input.returnPressed.connect(self.save_current)
        saving.addWidget(self.name_input, 1)
        self.save_button = QPushButton("保存")
        self.save_button.setToolTip("复制当前参考录音到音色库，保留原始文件。")
        self.save_button.clicked.connect(self.save_current)
        saving.addWidget(self.save_button)
        layout.addLayout(saving)

        self.detail = label("保存参考录音，随时切换音色。", "quiet")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.detail)
        self.confirmation = QWidget()
        confirm_layout = QHBoxLayout(self.confirmation)
        confirm_layout.setContentsMargins(0, 0, 0, 0)
        self.confirm_button = QPushButton("确认删除")
        self.confirm_button.setObjectName("voiceDelete")
        self.confirm_button.clicked.connect(self.confirm_delete)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._cancel_delete)
        confirm_layout.addWidget(self.confirm_button)
        confirm_layout.addWidget(self.cancel_button)
        layout.addWidget(self.confirmation)
        self.confirmation.hide()
        # QDialog 内管理按钮不能成为默认提交按钮，输入名称回车只触发保存。
        for button in (self.save_button, self.delete_button, self.confirm_button, self.cancel_button):
            button.setAutoDefault(False)
        self.reload()

    def eventFilter(self, watched, event):
        """消费名称框回车，阻止 QDialog 再触发“清空”等其他默认按钮。"""
        if (watched is self.name_input and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            self.save_current()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _message(self, text, error=False):
        """就地反馈，避免错误弹窗打断文本编辑；长错误可通过提示完整查看。"""
        self.detail.setText(text)
        self.detail.setToolTip(text)
        self.detail.setObjectName("voiceError" if error else "quiet")
        self.detail.style().unpolish(self.detail)
        self.detail.style().polish(self.detail)

    def reload(self, selected_id=None):
        """清单错误时停用库操作，但保持原来的临时参考及语音合成可用。"""
        try:
            state = self.library.load()
        except (OSError, VoiceLibraryError) as exc:
            self._available = False
            self._message(str(exc), True)
            self._update_buttons()
            return False
        self._available = True
        self._presets = state.presets
        self._restored_id = state.selected_id
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("临时参考 / 未选音色", None)
        for preset in self._presets:
            self.combo.addItem(preset.name, preset.id)
            missing = not self.library.audio_path(preset).is_file()
            self.combo.setItemData(self.combo.count() - 1,
                                   f"{preset.name} · {'录音已丢失' if missing else f'{preset.duration:.1f} 秒'}",
                                   Qt.ItemDataRole.ToolTipRole)
        index = self.combo.findData(selected_id) if selected_id else 0
        self.combo.setCurrentIndex(max(0, index))
        self.combo.blockSignals(False)
        self.count_label.setText(f"{len(self._presets)} 个音色")
        self._update_buttons()
        return True

    def restore_selection(self):
        """由窗口在信号连接完成后调用，防止初始化恢复的录音未传入合成接口。"""
        if self._available and self._restored_id:
            index = self.combo.findData(self._restored_id)
            if index > 0:
                self.combo.setCurrentIndex(index)

    def _selected(self):
        return next((item for item in self._presets if item.id == self.combo.currentData()), None)

    @property
    def reference_name(self):
        """向录音展示区提供用户名称，内部 UUID 仅用于文件管理。"""
        preset = self._selected()
        return preset.name if preset and self._reference else None

    def _update_buttons(self):
        """保存须有参考与新名称；选择已保存音色后，可换名另存，不误覆盖同名条目。"""
        self.combo.setEnabled(self._available and bool(self._presets))
        self.delete_button.setEnabled(self._available and self._selected() is not None)
        name = self.name_input.text().strip()
        duplicate = any(self.library._name_key(item.name) == self.library._name_key(name)
                        for item in self._presets)
        self.save_button.setEnabled(self._available and bool(self._reference)
                                    and Path(self._reference).is_file() and bool(name) and not duplicate)

    def set_reference(self, path):
        """手动浏览、拖放和音色选择共用此同步接口，防止界面音色名与实际录音不一致。"""
        path = str(path or "")
        self._reference = path
        if not self._available:
            self._update_buttons()
            return
        preset = next((item for item in self._presets
                       if path and self.library.audio_path(item).resolve() == Path(path).resolve()), None)
        selected_id = preset.id if preset else None
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(max(0, self.combo.findData(selected_id)))
        self.combo.blockSignals(False)
        self._pending_delete = None
        self.confirmation.hide()
        try:
            self.library.select(selected_id)
        except (OSError, VoiceLibraryError) as exc:
            self._message(str(exc), True)
            self._update_buttons()
            return
        if preset:
            self.name_input.setText(preset.name)
            self._message(f"{preset.duration:.1f} 秒 · {preset.sample_rate / 1000:g} kHz · 已应用")
        else:
            self.name_input.setText(Path(path).stem[:48] if path else "")
            self._message("当前为临时参考，命名后可保存到音色库。" if path else "保存参考录音，随时切换音色。")
        self._update_buttons()

    def _activate(self, index):
        """下拉即选即用，保持库录音路径与既有引擎参考缓存签名一致。"""
        if not self._available or not self.isEnabled():
            return
        self._pending_delete = None
        self.confirmation.hide()
        try:
            preset = self.library.select(self.combo.currentData())
            path = str(self.library.audio_path(preset)) if preset else ""
        except (OSError, VoiceLibraryError) as exc:
            self._reference = ""
            self.reference_selected.emit("")
            # 保留失效条目的选择，允许用户直接删除，不继续使用此前的其他音色。
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(index)
            self.combo.blockSignals(False)
            self._message(str(exc), True)
            self._update_buttons()
            return
        self.set_reference(path)
        self.reference_selected.emit(path)

    def save_current(self):
        if not self._available or not self.isEnabled():
            return
        try:
            preset = self.library.save(self.name_input.text(), self._reference)
            if self.reload(preset.id):
                path = str(self.library.audio_path(preset))
                self.set_reference(path)
                self.reference_selected.emit(path)
                self._message(f"已保存并应用 · {preset.duration:.1f} 秒 · 原录音已保留")
        except (OSError, VoiceLibraryError) as exc:
            logging.getLogger(__name__).warning("保存音色未完成：%s", exc)
            self._message(f"保存失败：{exc}", True)
        self._update_buttons()

    def _ask_delete(self):
        preset = self._selected()
        if preset is None or not self.isEnabled():
            return
        self._pending_delete = preset.id
        self._message(f"删除“{preset.name}”及库内录音副本？原始文件保留。")
        self.confirmation.show()

    def _cancel_delete(self):
        self._pending_delete = None
        self.confirmation.hide()
        self._message("已取消删除。")

    def confirm_delete(self):
        """确认绑定稳定 ID；切换音色自动取消旧确认，避免删掉新选择。"""
        preset = self._selected()
        if (not self.isEnabled() or preset is None
                or self._pending_delete != preset.id):
            return
        try:
            path = self.library.audio_path(preset)
            clear_reference = bool(self._reference) and Path(self._reference).resolve() == path.resolve()
            self.library.delete(preset.id)
            self._pending_delete = None
            self.confirmation.hide()
            self.reload()
            if clear_reference:
                self.set_reference("")
                self.reference_selected.emit("")
            self._message(f"已删除“{preset.name}”，原始录音保留。")
        except (OSError, VoiceLibraryError) as exc:
            logging.getLogger(__name__).warning("删除音色未完成：%s", exc)
            self._message(f"删除失败：{exc}", True)
        self._update_buttons()
