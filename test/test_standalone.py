"""验证独立项目契约、中文路径与窗口后台任务生命周期，不加载大模型。"""
from pathlib import Path
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime, validate_models
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication, QMessageBox
from app.window import IndexTTSWindow
from app.engine import IndexTTSEngine


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """复用 QApplication，避免测试间销毁 Qt 全局状态。"""
        cls.qt = QApplication.instance() or QApplication([])

    def test_root_files(self):
        self.assertEqual({p.name for p in Path('.').iterdir() if p.is_file()},
                         {"run.bat", "start.py", "requirements.txt"})

    def test_local_python_and_dependencies(self):
        self.assertEqual(Path(sys.prefix).resolve(), Path("even").resolve())
        import torch
        import indextts.infer_v2_5
        self.assertTrue(Path(torch.__file__).resolve().is_relative_to(Path("even").resolve()))
        self.assertTrue(Path(indextts.infer_v2_5.__file__).resolve().is_relative_to(Path("vendor").resolve()))
        # 使用 run.bat 的真实版本检查器，防止标记缺少换行导致无限重复安装。
        import subprocess
        from app.paths import APP_VERSION
        marker = subprocess.run(["findstr.exe", "/x", "/c:" + APP_VERSION, str(Path("even/.indextts_environment_version"))], capture_output=True)
        self.assertEqual(marker.returncode, 0, marker.stderr)

    def test_complete_offline_models(self):
        self.assertEqual(validate_models(), 14)
        self.assertEqual(validate_models(require_emotion=True), 18)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["HF_HUB_CACHE"], "cache/huggingface/hub")

    def test_emotion_modes_and_sampling_bounds(self):
        window = IndexTTSWindow()
        for mode in range(4):
            window.emo_method_combo.setCurrentIndex(mode)
            self.assertEqual(window.collect_emotion_params()["emo_mode"], mode)
            self.assertEqual(not window.emo_audio_widget.isHidden(), mode == 1)
            self.assertEqual(not window.emo_vector_widget.isHidden(), mode == 2)
            self.assertEqual(not window.emo_text_widget.isHidden(), mode == 3)
        self.assertGreater(window.top_p_spin.minimum(), 0)
        self.assertEqual(Path(window.get_indextts_model_path()), Path("models/IndexTTS-2.5"))
        window.close()

    def test_invalid_input_fails_before_loading(self):
        engine = IndexTTSEngine()
        with patch.object(engine, "load_model") as loader:
            with self.assertRaises(ValueError):
                engine.infer("  ", "missing.wav")
            with self.assertRaises(FileNotFoundError):
                engine.infer("你好", "missing.wav")
            loader.assert_not_called()

    def test_chinese_output_path_and_existing_model_reuse(self):
        """用轻量替身检查中文路径及调用边界，真实声学推理由独立测试执行。"""
        import wave
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(prefix="中文路径_", dir="cache") as folder:
            reference = Path(folder) / "参考.wav"
            reference.write_bytes(b"reference")
            output = Path(folder) / "结果.wav"
            engine = IndexTTSEngine()
            engine.is_loaded = True
            engine.tts_model = Mock()
            def render(**kwargs):
                with wave.open(kwargs["output_path"], "wb") as wav:
                    wav.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                    wav.writeframes(b"\0\0" * 100)
            engine.tts_model.infer.side_effect = render
            self.assertEqual(engine.infer("你好", reference, output, emo_mode=2, emo_vector=[0.1]*8), str(output))
            engine.tts_model.normalize_emo_vec.assert_called_once()
            self.assertTrue(output.is_file())

    def test_missing_output_is_an_error(self):
        from unittest.mock import Mock
        engine = IndexTTSEngine()
        engine.is_loaded = True
        engine.tts_model = Mock()
        with tempfile.TemporaryDirectory(dir="cache") as folder:
            reference = Path(folder) / "reference.wav"
            reference.touch()
            with self.assertRaises(RuntimeError):
                engine.infer("你好", reference, Path(folder) / "result.wav")

    def test_completion_survives_worker_cleanup_during_playback(self):
        """播放提示框中的嵌套事件循环可能先清理 worker，结果信号仍须正常发出。"""
        from unittest.mock import Mock
        window = IndexTTSWindow()
        window.worker = Mock()
        window.worker.text = "本次合成的文本"
        received = []
        window.audio_generated.connect(lambda path, text: received.append((path, text)))
        with patch.object(window, "play_generated_audio", side_effect=lambda path: window.cleanup_worker()):
            window.on_generation_finished("outputs/test.wav")
        self.assertEqual(received, [("outputs/test.wav", "本次合成的文本")])
        window.close()

    def test_close_waits_for_generation_thread(self):
        """关闭期间保留线程对象，任务结束后自动清理，防止 QThread 崩溃。"""
        from app.worker import IndexTTSWorker
        class SlowEngine:
            is_loaded = True
            def load_model(self):
                pass
            def infer(self, *args, **kwargs):
                time.sleep(0.15)
                return "outputs/thread_test.wav"
        window = IndexTTSWindow()
        window.show()
        worker = IndexTTSWorker("测试", "reference.wav", {}, {}, SlowEngine())
        window.worker = worker
        window.generate_btn.setEnabled(False)
        worker.finished.connect(window.cleanup_worker)
        worker.start()
        self.assertFalse(window.close())
        self.assertIs(window.worker, worker)
        self.assertTrue(window._close_when_finished)
        deadline = time.monotonic() + 5
        while window.worker is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.assertIsNone(window.worker)
        self.assertFalse(window.isVisible())


if __name__ == "__main__":
    unittest.main(verbosity=2)
