"""桌面工作台的小型控件：真实波形、音频拖放和统一的线条图标。"""
from pathlib import Path
import math

from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSlider, QSizePolicy

from .theme import COLORS


def icon(name, color="#a8b8ce", size=20):
    """以矢量笔画生成图标，不依赖 emoji 字体或外部图片。"""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 24, size / 24)
    painter.setPen(QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    if name in ("wave", "brand"):
        for x, height in zip((4, 8, 12, 16, 20), (5, 11, 18, 11, 5)):
            painter.drawLine(QPointF(x, 12 - height / 2), QPointF(x, 12 + height / 2))
    elif name == "folder":
        path = QPainterPath(QPointF(3, 7))
        for x, y in ((3, 5), (10, 5), (12, 8), (21, 8), (21, 19), (3, 19), (3, 7)):
            path.lineTo(x, y)
        painter.drawPath(path)
    elif name == "upload":
        painter.drawLine(QPointF(12, 15), QPointF(12, 4))
        painter.drawLine(QPointF(8, 8), QPointF(12, 4))
        painter.drawLine(QPointF(12, 4), QPointF(16, 8))
        path = QPainterPath(QPointF(4, 15))
        path.lineTo(4, 20); path.lineTo(20, 20); path.lineTo(20, 15)
        painter.drawPath(path)
    elif name == "pause":
        painter.setBrush(QColor(color)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(7, 5, 3, 14), 1, 1)
        painter.drawRoundedRect(QRectF(14, 5, 3, 14), 1, 1)
    elif name == "play":
        painter.setBrush(QColor(color)); painter.setPen(Qt.PenStyle.NoPen)
        path = QPainterPath(QPointF(8, 5)); path.lineTo(19, 12); path.lineTo(8, 19); path.closeSubpath()
        painter.drawPath(path)
    elif name == "chevron":
        painter.drawLine(QPointF(8, 10), QPointF(12, 14))
        painter.drawLine(QPointF(12, 14), QPointF(16, 10))
    elif name == "spark":
        path = QPainterPath(QPointF(12, 3))
        for x, y in ((14.5, 9.5), (21, 12), (14.5, 14.5), (12, 21), (9.5, 14.5), (3, 12), (9.5, 9.5), (12, 3)):
            path.lineTo(x, y)
        painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


def label(text, role="muted"):
    """统一标签层级，避免在各处重复写行内颜色。"""
    widget = QLabel(text)
    widget.setObjectName(role)
    return widget


class Card(QFrame):
    """只有细边框的平面卡片，标题与内容按统一节奏留白。"""
    def __init__(self, title, number=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(14)
        self.heading = QHBoxLayout()
        self.heading.setSpacing(10)
        if number:
            self.heading.addWidget(label(number, "eyebrow"))
        self.heading.addWidget(label(title, "sectionTitle"))
        self.heading.addStretch()
        self.body.addLayout(self.heading)


class StudioSlider(QSlider):
    """绘制轻量平面滑杆，避免 Windows 原生子控件叠加出粗重凹槽。"""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFixedHeight(18)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right = 6.0, float(max(6, self.width() - 6))
        middle = self.height() / 2
        ratio = (self.value() - self.minimum()) / max(1, self.maximum() - self.minimum())
        position = left + (right - left) * ratio
        painter.setPen(QPen(QColor("#35404e"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(left, middle), QPointF(right, middle))
        painter.setPen(QPen(QColor("#879bb6"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(left, middle), QPointF(position, middle))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#bbc9da"))
        painter.drawEllipse(QPointF(position, middle), 4.5, 4.5)
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#788fae"), 1))
            painter.drawEllipse(QPointF(position, middle), 7, 7)

    def _value_at(self, event):
        ratio = max(0.0, min(1.0, (event.position().x() - 6) / max(1, self.width() - 12)))
        self.setValue(round(self.minimum() + ratio * (self.maximum() - self.minimum())))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.setSliderDown(True)
            self._value_at(event)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self._value_at(event)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class AudioDrop(QFrame):
    """只接受现存音频文件的拖放区，文件信息和绝对路径用不同方式展示。"""
    file_selected = pyqtSignal(str)
    browse_requested = pyqtSignal()
    SUPPORTED = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("audioDrop")
        self.setAcceptDrops(True)
        self.setMinimumHeight(112)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 14, 13)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        mark = QLabel(); mark.setPixmap(icon("wave", size=22).pixmap(22, 22))
        heading.addWidget(mark)
        self.title = label("拖入音频，定义音色", "fileTitle")
        self.title.setMinimumWidth(0)
        heading.addWidget(self.title, 1)
        layout.addLayout(heading)
        self.detail = label("WAV / MP3 / FLAC · 建议清晰的单人录音", "quiet")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.browse = QPushButton("选择参考音频")
        self.browse.setIcon(icon("upload", size=15))
        self.browse.clicked.connect(self.browse_requested)
        layout.addWidget(self.browse)

    def set_file(self, path):
        """只做轻量元数据读取，不在界面中加载声学模型。"""
        file = Path(path)
        if not path or not file.is_file():
            self.setProperty("loaded", False)
            self.title.setText("拖入音频，定义音色")
            self.detail.setText("WAV / MP3 / FLAC · 建议清晰的单人录音")
            self.title.setToolTip("")
            self.browse.setText("选择参考音频")
            self.style().unpolish(self); self.style().polish(self)
            return
        from PyQt6.QtGui import QFontMetrics
        self.title.setText(QFontMetrics(self.title.font()).elidedText(file.name, Qt.TextElideMode.ElideMiddle, 235))
        self.title.setToolTip(str(file))
        try:
            import soundfile as sf
            info = sf.info(str(file))
            self.detail.setText(f"{info.duration:.1f} 秒   /   {info.samplerate / 1000:g} kHz   /   {file.suffix[1:].upper()}")
        except Exception:
            self.detail.setText(f"{file.suffix[1:].upper()} 音频 · 已选择")
        self.browse.setText("更换参考音频")
        self.setProperty("loaded", True)
        self.style().unpolish(self); self.style().polish(self)

    def _file_from_mime(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.is_file() and path.suffix.lower() in self.SUPPORTED:
                    return str(path)
        return None

    def dragEnterEvent(self, event):
        if self._file_from_mime(event):
            self.setProperty("dragging", True)
            self.style().unpolish(self); self.style().polish(self)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setProperty("dragging", False)
        self.style().unpolish(self); self.style().polish(self)

    def dropEvent(self, event):
        path = self._file_from_mime(event)
        self.dragLeaveEvent(event)
        if path:
            self.file_selected.emit(path)
            event.acceptProposedAction()


class Waveform(QWidget):
    """从真实音频抽取峰值，绘制播放进度；空状态不伪造音频数据。"""
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks = []
        self.progress = 0.0
        self.setMinimumHeight(70)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def load_audio(self, path):
        """分块聚合幅值，长音频也不一次性占用大量内存。"""
        import numpy as np
        import soundfile as sf
        peaks = []
        with sf.SoundFile(str(path)) as stream:
            block_size = max(1, math.ceil(len(stream) / 180))
            for block in stream.blocks(blocksize=block_size, dtype="float32", always_2d=True):
                peaks.append(float(np.max(np.abs(block))))
        maximum = max(peaks, default=1.0) or 1.0
        self.peaks = [value / maximum for value in peaks]
        self.progress = 0.0
        self.update()

    def set_progress(self, ratio):
        self.progress = max(0.0, min(1.0, ratio))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        if not self.peaks:
            painter.setPen(QPen(QColor("#303a47"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(0, height // 2, width, height // 2)
            painter.setPen(QColor("#788596"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "生成后，声音会出现在这里")
            return
        count = min(len(self.peaks), max(1, width // 4))
        step = width / count
        for i in range(count):
            value = self.peaks[min(len(self.peaks) - 1, int(i * len(self.peaks) / count))]
            bar_height = max(3, value * (height - 10))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#b2c2d7" if i / count <= self.progress else "#4b5c73"))
            painter.drawRoundedRect(QRectF(i * step, (height - bar_height) / 2, max(1.5, step - 2), bar_height), 1, 1)

    def mousePressEvent(self, event):
        if self.peaks and event.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(max(0.0, min(1.0, event.position().x() / max(1, self.width()))))
