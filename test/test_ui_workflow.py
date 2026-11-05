"""界面交互回归：覆盖任务提交、真实结果预览、失败恢复和拖放，不重复运行大模型。"""
from pathlib import Path
import os
import sys
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import QMimeData, QUrl, QPointF, Qt
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication
from app.window import IndexTTSWindow


class UIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = IndexTTSWindow()
        self.window.auto_play.setChecked(False)

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def wait_for_worker(self):
        """运行真实 Qt 事件队列，确保 finished 清理与结果回调顺序正确。"""
        deadline = time.monotonic() + 4
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.worker)

    def test_mode_buttons_match_parameter_contract(self):
        self.window.show()
        for index, button in enumerate(self.window.mode_buttons):
            button.click()
            self.app.processEvents()
            self.assertFalse(self.window.grab().isNull())
            self.assertEqual(self.window.collect_emotion_params()["emo_mode"], index)
            self.assertEqual(sum(b.isChecked() for b in self.window.mode_buttons), 1)
        self.window.emo_method_combo.setCurrentIndex(2)
        self.window.emo_sliders[0][0].setValue(37)
        self.assertEqual(self.window.collect_emotion_params()["emo_vector"][0], 0.37)
        self.assertEqual(self.window.emo_sliders[0][1].text(), "0.37")

    def test_flat_slider_pointer_and_keyboard(self):
        """验证自绘滑杆的点击位置、键盘微调及绘制，而不只检查赋值。"""
        from PyQt6.QtTest import QTest
        from PyQt6.QtCore import QPoint
        self.window.show()
        self.window.mode_buttons[2].click()
        self.app.processEvents()
        slider = self.window.emo_sliders[0][0]
        QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width() // 2, 9))
        self.assertAlmostEqual(slider.value(), 50, delta=2)
        before = slider.value()
        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertEqual(slider.value(), before + 1)
        self.assertFalse(slider.grab().isNull())

    def test_advanced_fold_and_character_count(self):
        self.assertTrue(self.window.advanced_content.isHidden())
        self.window.advanced_toggle.click()
        self.assertFalse(self.window.advanced_content.isHidden())
        self.window.text_input.setPlainText("你好，世界")
        self.assertEqual(self.window.char_count.text(), "5 字")

    def test_input_errors_do_not_start_worker(self):
        self.window.generate_speech()
        self.assertIsNone(self.window.worker)
        self.assertEqual(self.window.status_label.property("state"), "error")
        self.window.text_input.setPlainText("你好")
        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        self.window.mode_buttons[1].click()
        self.window.generate_speech()
        self.assertIsNone(self.window.worker)
        self.assertIn("情感参考", self.window.status_label.text())

    def test_successful_background_task_updates_real_preview(self):
        """真实工作线程调用轻量替身，播放器使用随项目保存的真实 WAV，避免依赖历史输出。"""
        engine = Mock()
        engine.is_loaded = True
        engine.infer.return_value = "test/fixtures/reference.wav"
        self.window.engine = engine
        self.window.text_input.setPlainText("你好，这是工作台测试。")
        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        self.window.mode_buttons[2].click()
        self.window.emo_sliders[0][0].setValue(20)
        self.window.generate_speech()
        self.assertFalse(self.window.generate_btn.isEnabled())
        self.wait_for_worker()
        engine.infer.assert_called_once()
        self.assertEqual(engine.infer.call_args.kwargs["emo_vector"][0], 0.2)
        self.assertEqual(engine.infer.call_args.kwargs["temperature"], 0.8)
        self.assertTrue(self.window.generate_btn.isEnabled())
        self.assertTrue(self.window.busy_indicator.isHidden())
        self.assertTrue(self.window.play_btn.isEnabled())
        self.assertGreater(len(self.window.waveform.peaks), 50)
        self.assertGreater(self.window._duration_ms, 0)
        self.assertEqual(self.window.status_label.property("state"), "done")
        # setSource 由媒体线程异步解码，等到可定位后再检查真实 seek 行为。
        deadline = time.monotonic() + 3
        while not self.window.player.isSeekable() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(self.window.player.isSeekable())
        self.window._seek_to(0.5)
        self.assertAlmostEqual(self.window.player.position() / self.window._duration_ms, 0.5, delta=0.02)

    def test_failed_background_task_restores_controls(self):
        engine = Mock()
        engine.is_loaded = True
        engine.infer.side_effect = RuntimeError("测试中的预期失败")
        self.window.engine = engine
        self.window.text_input.setPlainText("你好")
        self.window.speaker_audio_input.setText("test/fixtures/reference.wav")
        self.window.generate_speech()
        self.wait_for_worker()
        self.assertTrue(self.window.generate_btn.isEnabled())
        self.assertEqual(self.window.status_label.property("state"), "error")
        self.assertIn("预期失败", self.window.status_hint.text())

    def test_audio_drop_and_clear(self):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(Path("test/fixtures/reference.wav").resolve()))])
        event = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime,
                           Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.window.speaker_drop.dropEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertTrue(Path(self.window.speaker_audio_input.text()).is_file())
        self.assertTrue(self.window.speaker_drop.property("loaded"))
        self.window.speaker_audio_input.clear()
        self.assertFalse(self.window.speaker_drop.property("loaded"))
        self.assertEqual(self.window.speaker_drop.browse.text(), "选择参考音频")


if __name__ == "__main__":
    unittest.main(verbosity=2)
