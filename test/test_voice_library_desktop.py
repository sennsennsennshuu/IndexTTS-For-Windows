"""真实 Windows 音色库验收；可选执行一次使用库内录音的真实模型合成。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesize", action="store_true", help="额外运行真实模型合成")
    args = parser.parse_args()
    prepare_runtime()
    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication
    from app.voice_library import VoiceLibrary
    from app.window import IndexTTSWindow
    import numpy as np
    import soundfile as sf

    # 独立验证库使用真实宿主机项目目录，不导入用户的现有音色库。
    temporary = tempfile.TemporaryDirectory(prefix="voice-desktop-test-", dir="cache/tmp")
    library = VoiceLibrary(Path(temporary.name) / "library")
    source = Path("test/fixtures/reference.wav")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    app = QApplication(["IndexTTS 音色库验收"])
    window = IndexTTSWindow(voice_library=library)
    window.auto_play.setChecked(False)
    window._set_speaker_audio(str(source))
    panel = window.voice_library_panel
    panel.name_input.setText("温柔女声 · 验收示例")
    panel.save_button.click()
    first = library.load().presets[0]
    panel.name_input.setText("叙述音色 · 验收示例")
    panel.save_button.click()
    assert len(library.load().presets) == 2, panel.detail.text()
    panel.combo.setCurrentIndex(panel.combo.findData(first.id))
    assert hashlib.sha256(library.audio_path(first).read_bytes()).hexdigest() == source_sha
    window.close()
    # 创建第二个真实窗口，验证无需重新选文件即可恢复。
    window = IndexTTSWindow(voice_library=VoiceLibrary(library.root))
    window.auto_play.setChecked(False)
    panel = window.voice_library_panel
    assert panel.combo.currentData() == first.id
    assert Path(window.speaker_audio_input.text()) == library.audio_path(first)
    window.text_input.setPlainText("你好，这是保存音色后的语音合成测试。")
    window.show()
    started = time.monotonic()
    result = {"qt_platform": QGuiApplication.platformName(), "source_sha256": source_sha,
              "copy_sha256": first.sha256, "saved_count": 2, "restart_restored": True,
              "real_synthesis": args.synthesize, "phase": "layout", "screenshots": []}
    timer = QTimer()

    def finish(error=None):
        timer.stop()
        if error:
            result["error"] = str(error)
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        window.close()
        if window.worker is None:
            temporary.cleanup()
            result["temporary_library_cleaned"] = not Path(temporary.name).exists()
        receipt = "logs/voice_library_desktop.json" if args.synthesize else "logs/voice_library_layout.json"
        Path(receipt).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        app.exit(1 if error else 0)

    def check_delete():
        panel.delete_button.click()
        assert not panel.confirmation.isHidden()
        panel.cancel_button.click()
        assert library.audio_path(first).is_file()
        panel.delete_button.click()
        panel.confirm_button.click()
        assert len(library.load().presets) == 1
        assert not library.audio_path(first).exists()
        assert window.speaker_audio_input.text() == ""
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
        result.update({"delete_confirmation_passed": True, "original_preserved": True,
                       "selected_reference_cleared": True, "remaining_count": 1, "phase": "complete"})
        finish()

    def poll():
        try:
            if result.get("pending_error"):
                if window.worker is None:
                    finish(result["pending_error"])
                return
            if time.monotonic() - started > 240:
                raise TimeoutError("音色库桌面验收超过四分钟")
            if result["phase"] == "layout":
                assert window.isVisible() and window.width() == 1180
                window.resize(1000, 700)
                result["phase"] = "compact"
            elif result["phase"] == "compact":
                assert panel.save_button.geometry().right() <= panel.width()
                assert panel.combo.width() > 100
                path = "docs/voice_library_compact.png"
                assert window.grab().save(path)
                result["screenshots"].append(path)
                window.resize(1180, 820)
                result["phase"] = "normal"
            elif result["phase"] == "normal":
                if args.synthesize:
                    window.generate_speech()
                    assert window.worker is not None
                    assert not panel.delete_button.isEnabled()
                    result["controls_locked_during_synthesis"] = True
                    result["phase"] = "synthesis"
                else:
                    path = "docs/voice_library_desktop.png"
                    assert window.grab().save(path)
                    result["screenshots"].append(path)
                    check_delete()
            elif result["phase"] == "synthesis" and window.worker is None:
                assert window._result_path, window.status_hint.text()
                path = Path(window._result_path)
                samples, rate = sf.read(path, dtype="float32")
                assert rate == 22050 and len(samples) > 4410
                assert np.isfinite(samples).all() and np.sqrt(np.mean(samples ** 2)) > 1e-5
                result.update({"audio": path.as_posix(), "audio_seconds": len(samples) / rate,
                               "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "controls_restored": panel.delete_button.isEnabled()})
                image_path = "docs/voice_library_desktop.png"
                assert window.grab().save(image_path)
                result["screenshots"].append(image_path)
                check_delete()
        except Exception as exc:
            # 正常验收不会在推理中异常退出；遇到失败仍由原窗口生命周期等待线程结束。
            if window.worker is not None:
                result["pending_error"] = str(exc)
                return
            finish(exc)

    timer.timeout.connect(poll)
    timer.start(250)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
