"""真实 Windows 桌面入口测试：调用 start.main，截图后通过 Qt 正常关闭。"""
from pathlib import Path
import json
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import start


def main():
    original_exec = QApplication.exec
    result = {"visible": False, "exit_code": None}

    def verify_and_close():
        """分帧检查真实窗口的空、完成和向量三种状态，等待布局与媒体初始化。"""
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if widget.isVisible() and widget.windowTitle() == "IndexTTS 语音合成":
                result["visible"] = True
                result["screenshot_saved"] = widget.grab().save("docs/desktop_window.png")
                widget.auto_play.setChecked(False)
                widget.text_input.setPlainText("你好，这是独立项目的语音合成测试。")
                widget.speaker_audio_input.setText("test/fixtures/reference.wav")
                widget.play_generated_audio("outputs/独立项目验收_模式0.wav")
                widget.update_status("语音已生成", "done")
                widget.status_hint.setText("音频已保存。可以继续调整文本，或试听当前结果。")

                def capture_result(window=widget):
                    result["result_preview_saved"] = window.grab().save("docs/desktop_result.png")
                    window.emo_method_combo.setCurrentIndex(2)
                    QTimer.singleShot(160, lambda: capture_emotion(window))

                def capture_emotion(window):
                    result["emotion_preview_saved"] = window.grab().save("docs/desktop_emotion.png")
                    result["waveform_bins"] = len(window.waveform.peaks)
                    result["window_logical_size"] = [window.width(), window.height()]
                    window.close()
                    app.quit()

                QTimer.singleShot(180, capture_result)
                return
        app.quit()

    def timed_exec(*args):
        QTimer.singleShot(1200, verify_and_close)
        QTimer.singleShot(6000, QApplication.instance().quit)
        return original_exec()

    with patch.object(sys, "argv", ["start.py"]), patch.object(QApplication, "exec", timed_exec):
        result["exit_code"] = start.main()
    Path("docs/desktop_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["visible"] or result["exit_code"] != 0 or not all(result.get(key) for key in (
            "screenshot_saved", "result_preview_saved", "emotion_preview_saved")):
        raise RuntimeError(f"桌面启动验收失败：{result}")
    print("Windows 桌面入口显示、截图及正常关闭均通过")


if __name__ == "__main__":
    main()
