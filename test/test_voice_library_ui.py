"""真实 Qt 音色库交互：保存、应用、确认删除、重启和后台合成参数。"""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtWidgets import QApplication
from app.voice_library import VoiceLibrary
from app.window import IndexTTSWindow


class VoiceLibraryUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="voice-ui-test-", dir="cache/tmp")
        self.library = VoiceLibrary(Path(self.temporary.name) / "library")
        self.window = IndexTTSWindow(voice_library=self.library)
        self.window.auto_play.setChecked(False)
        self.panel = self.window.voice_library_panel

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.temporary.cleanup()

    def save_voice(self, name="我的中文音色"):
        self.window._set_speaker_audio("test/fixtures/reference.wav")
        self.panel.name_input.setText(name)
        self.panel.save_button.click()
        self.app.processEvents()
        return self.library.load().presets[-1]

    def test_empty_state_and_external_reference_enables_save(self):
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertFalse(self.panel.delete_button.isEnabled())
        self.window._set_speaker_audio("test/fixtures/reference.wav")
        self.assertTrue(self.panel.save_button.isEnabled())
        self.assertEqual(self.panel.name_input.text(), "reference")
        self.assertIsNone(self.panel.combo.currentData())

    def test_save_updates_reference_to_managed_copy(self):
        preset = self.save_voice()
        self.assertEqual(self.panel.combo.currentData(), preset.id)
        self.assertEqual(Path(self.window.speaker_audio_input.text()), self.library.audio_path(preset))
        self.assertIn("1 个", self.panel.count_label.text())
        self.assertIn("已保存", self.panel.detail.text())
        self.assertEqual(self.window.speaker_drop.title.text(), preset.name)
        self.assertTrue(self.panel.delete_button.isEnabled())
        self.assertFalse(self.panel.save_button.isEnabled())

    def test_selection_applies_saved_reference_and_manual_change_deselects(self):
        first = self.save_voice("第一音色")
        self.save_voice("第二音色")
        self.panel.combo.setCurrentIndex(self.panel.combo.findData(first.id))
        self.assertEqual(Path(self.window.speaker_audio_input.text()), self.library.audio_path(first))
        self.window._set_speaker_audio("test/fixtures/reference.wav")
        self.assertIsNone(self.panel.combo.currentData())
        self.assertIsNone(self.library.load().selected_id)
        self.assertFalse(self.panel.delete_button.isEnabled())

    def test_restart_restores_selected_reference(self):
        preset = self.save_voice()
        self.window.close()
        self.window = IndexTTSWindow(voice_library=VoiceLibrary(self.library.root))
        self.panel = self.window.voice_library_panel
        self.assertEqual(self.panel.combo.currentData(), preset.id)
        self.assertEqual(Path(self.window.speaker_audio_input.text()), self.library.audio_path(preset))

    def test_enter_in_name_only_saves_without_generating_or_clearing_text(self):
        """QDialog 默认按钮可能吞掉回车，必须实际验证名称框的键盘路径。"""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest
        engine = Mock()
        engine.is_loaded = True
        self.window.engine = engine
        self.window.text_input.setPlainText("保存名称时保留这段文字。")
        self.window._set_speaker_audio("test/fixtures/reference.wav")
        self.panel.name_input.setText("键盘保存的音色")
        self.window.show()
        self.panel.name_input.setFocus()
        self.app.processEvents()
        QTest.keyClick(self.panel.name_input, Qt.Key.Key_Return)
        self.app.processEvents()
        self.assertEqual(len(self.library.load().presets), 1)
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.window.text_input.toPlainText(), "保存名称时保留这段文字。")
        engine.infer.assert_not_called()

    def test_delete_requires_confirmation_and_preserves_original(self):
        preset = self.save_voice()
        source = Path("test/fixtures/reference.wav")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        self.panel.delete_button.click()
        self.assertFalse(self.panel.confirmation.isHidden())
        self.assertTrue(self.library.audio_path(preset).is_file())
        self.panel.cancel_button.click()
        self.assertTrue(self.library.audio_path(preset).is_file())
        self.panel.delete_button.click()
        self.panel.confirm_button.click()
        self.assertEqual(self.library.load().presets, ())
        self.assertEqual(self.window.speaker_audio_input.text(), "")
        self.assertFalse(self.window.speaker_drop.property("loaded"))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertFalse(self.panel.delete_button.isEnabled())

    def test_selection_change_cancels_pending_delete(self):
        first = self.save_voice("第一音色")
        second = self.save_voice("第二音色")
        self.panel.delete_button.click()
        self.panel.combo.setCurrentIndex(self.panel.combo.findData(first.id))
        self.assertTrue(self.panel.confirmation.isHidden())
        self.panel.confirm_delete()
        self.assertEqual(len(self.library.load().presets), 2)
        self.assertTrue(self.library.audio_path(second).is_file())

    def test_missing_restored_reference_is_not_used_for_synthesis(self):
        preset = self.save_voice()
        self.library.audio_path(preset).unlink()
        self.window.close()
        self.window = IndexTTSWindow(voice_library=VoiceLibrary(self.library.root))
        self.panel = self.window.voice_library_panel
        self.assertEqual(self.window.speaker_audio_input.text(), "")
        self.assertIn("丢失", self.panel.detail.text())
        self.assertTrue(self.panel.delete_button.isEnabled())
        self.panel.delete_button.click()
        self.panel.confirm_button.click()
        self.assertEqual(self.library.load().presets, ())

    def test_saved_reference_reaches_real_worker_and_is_locked_during_generation(self):
        preset = self.save_voice()
        engine = Mock()
        engine.is_loaded = True
        engine.infer.return_value = "test/fixtures/reference.wav"
        self.window.engine = engine
        self.window.text_input.setPlainText("验证音色库参考传递。")
        self.window.generate_speech()
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertFalse(self.panel.combo.isEnabled())
        self.panel.confirm_delete()
        deadline = time.monotonic() + 5
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(self.window.worker)
        self.assertEqual(Path(engine.infer.call_args.args[1]), self.library.audio_path(preset))
        self.assertTrue(self.panel.combo.isEnabled())
        self.assertTrue(self.panel.delete_button.isEnabled())
        self.assertEqual(len(self.library.load().presets), 1)
        self.assertGreater(self.window._duration_ms, 0)
        self.assertTrue(self.window.play_btn.isEnabled())

    def test_corrupt_library_keeps_manual_reference_workflow_available(self):
        self.save_voice()
        self.library.index_path.write_text("{broken", encoding="utf-8")
        self.window.close()
        self.window = IndexTTSWindow(voice_library=self.library)
        self.panel = self.window.voice_library_panel
        self.window._set_speaker_audio("test/fixtures/reference.wav")
        self.assertTrue(self.window.speaker_drop.property("loaded"))
        self.assertEqual(self.window.speaker_audio_input.text(), "test/fixtures/reference.wav")
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertIn("保留原文件", self.panel.detail.text())

    def test_long_name_keeps_controls_inside_panel_at_minimum_size(self):
        self.save_voice("中文音色名称" * 8)
        self.window.resize(1000, 700)
        self.window.show()
        self.app.processEvents()
        self.assertFalse(self.window.grab().isNull())
        self.assertLessEqual(self.panel.name_input.geometry().right(), self.panel.width())
        self.assertLessEqual(self.panel.save_button.geometry().right(), self.panel.width())
        self.assertGreater(self.panel.combo.width(), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
