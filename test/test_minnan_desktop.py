"""真实 Windows Qt 全流程：普通话后台翻译、台语生成、截图与可解码音频验收。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    prepare_runtime()
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication
    from app.window import IndexTTSWindow
    import numpy as np
    import soundfile as sf

    app = QApplication(["IndexTTS 台语验收"])
    window = IndexTTSWindow()
    window.auto_play.setChecked(False)
    window.language_combo.setCurrentIndex(window.language_combo.findData("ZH-NAN-TW"))
    window.speaker_audio_input.setText(
        "models/extra_language/zh-nan-tw/model/heldout_audio/original_audio/e063fd6482c1ca793587313422420859722f39f6353a36da89a31a6efe14bf89.wav")
    window.text_input.setPlainText("我今天下午要去医院看医生。")
    window.show()
    started = time.monotonic()
    state = {"phase": "preview", "source": window.text_input.toPlainText(),
             "qt_platform": QGuiApplication.platformName(), "native_listening_passed": False}
    timer = QTimer()

    def finish(error=None):
        timer.stop()
        if error:
            state["error"] = str(error)
        state["seconds"] = time.monotonic() - started
        Path("logs/minnan_desktop_validation.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state, ensure_ascii=False), flush=True)
        window.close()
        app.exit(1 if error else 0)

    def poll():
        try:
            if time.monotonic() - started > 420:
                raise TimeoutError("桌面全流程超过七分钟，请检查阶段日志")
            if state["phase"] == "preview":
                if window._minnan_preview_ready and window.minnan_preview_worker is None:
                    prepared = window.minnan_preview.toPlainText().strip()
                    state["translated"] = prepared
                    state["input_mode"] = window.minnan_mode_combo.currentData()
                    state["translation_mode"] = window.minnan_translation_combo.currentData()
                    assert prepared and prepared != state["source"], f"译文异常：{prepared!r}，状态={window.minnan_status.text()}"
                    state["translated"] = prepared
                    print("实际台语转换：" + prepared, flush=True)
                    state["phase"] = "synthesis"
                    window.generate_speech()
            elif state["phase"] == "synthesis" and window._result_path and window.worker is None:
                audio = Path(window._result_path)
                samples, rate = sf.read(audio, dtype="float32")
                assert rate == 22050 and np.isfinite(samples).all() and len(samples) > 4410
                assert np.sqrt(np.mean(samples**2)) > 1e-5
                state.update({"phase": "complete", "audio": audio.as_posix(), "sample_rate": rate,
                              "audio_seconds": len(samples) / rate})
                window.grab().save("docs/minnan_desktop.png")
                finish()
        except Exception as error:
            finish(error)

    timer.timeout.connect(poll)
    timer.start(500)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
