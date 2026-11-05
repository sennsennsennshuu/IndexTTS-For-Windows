"""生成窗口截图供人工检查布局，使用项目内 Qt 环境且不加载模型。"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtTest import QTest
from app.window import IndexTTSWindow


def main():
    app = QApplication([])
    # Qt 离屏平台不会枚举 Windows 字体，截图时显式加载系统字体，避免方框字。
    font_path = Path(os.environ["WINDIR"]) / "Fonts" / "msyh.ttc"
    QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei", 9))
    window = IndexTTSWindow()
    window.show()
    app.processEvents()
    window.grab().save("docs/window_preview.png")
    window.emo_method_combo.setCurrentIndex(2)
    app.processEvents()
    QTest.qWait(60)
    window.grab().save("docs/window_emotion_preview.png")
    # 完成状态使用已验证的真实输出文件，不绘制虚构波形。
    window.emo_method_combo.setCurrentIndex(0)
    window.auto_play.setChecked(False)
    window.text_input.setPlainText("你好，这是独立项目的语音合成测试。")
    window.speaker_audio_input.setText("test/fixtures/reference.wav")
    window.play_generated_audio("outputs/2.5验收_情感模式0.wav")
    window.update_status("语音已生成", "done")
    window.status_hint.setText("音频已保存。可以继续调整文本，或试听当前结果。")
    app.processEvents()
    QTest.qWait(60)
    window.grab().save("docs/window_result_preview.png")
    window.resize(1000, 700)
    window.emo_method_combo.setCurrentIndex(2)
    window.advanced_toggle.setChecked(True)
    app.processEvents()
    QTest.qWait(60)
    window.grab().save("docs/window_compact_preview.png")
    window.close()


if __name__ == "__main__":
    main()
