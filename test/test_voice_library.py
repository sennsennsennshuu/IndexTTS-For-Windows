"""真实文件音色库回归：复制摘要、持久化、失败回滚和受管删除范围。"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime
prepare_runtime()
from app.voice_library import VoiceLibrary, VoiceLibraryError


class VoiceLibraryTests(unittest.TestCase):
    def setUp(self):
        # 所有验证文件都在真实项目的 cache/tmp 内，退出时只清理本次独占目录。
        self.temporary = tempfile.TemporaryDirectory(prefix="voice-library-test-", dir="cache/tmp")
        self.root = Path(self.temporary.name)
        self.library = VoiceLibrary(self.root / "library")
        self.source = self.root / "原始录音.wav"
        self.source.write_bytes(Path("test/fixtures/reference.wav").read_bytes())
        self.source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def assert_original_untouched(self):
        self.assertTrue(self.source.is_file())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_empty_library_does_not_create_files(self):
        state = self.library.load()
        self.assertEqual(state.presets, ())
        self.assertIsNone(state.selected_id)
        self.assertFalse(self.library.root.exists())

    def test_save_copies_audio_metadata_and_recovers_after_restart(self):
        preset = self.library.save("台湾女声 / 温柔", self.source)
        path = self.library.audio_path(preset)
        self.assertNotEqual(path.resolve(), self.source.resolve())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.source_hash)
        self.assertEqual(preset.sha256, self.source_hash)
        self.assertEqual(preset.original_filename, self.source.name)
        self.assertGreater(preset.duration, 0)
        self.assertGreater(preset.sample_rate, 0)
        self.assertFalse(Path(preset.audio_file).is_absolute())
        reopened = VoiceLibrary(self.library.root).load()
        self.assertEqual(reopened.presets, (preset,))
        self.assertEqual(reopened.selected_id, preset.id)
        self.assertIn("台湾女声", self.library.index_path.read_text(encoding="utf-8"))
        self.assert_original_untouched()

    def test_copy_survives_original_moved(self):
        preset = self.library.save("音色", self.source)
        moved = self.source.with_name("搬走的原始录音.wav")
        self.source.rename(moved)
        selected = self.library.select(preset.id)
        self.assertEqual(hashlib.sha256(self.library.audio_path(selected).read_bytes()).hexdigest(), self.source_hash)
        self.assertEqual(hashlib.sha256(moved.read_bytes()).hexdigest(), self.source_hash)

    def test_duplicate_name_never_overwrites(self):
        self.library.save("Voice A", self.source)
        before = self.library.index_path.read_bytes()
        for name in ("voice a", "  Voice   A  ", "Ｖｏｉｃｅ Ａ"):
            with self.subTest(name=name), self.assertRaisesRegex(VoiceLibraryError, "同名"):
                self.library.save(name, self.source)
        self.assertEqual(self.library.index_path.read_bytes(), before)
        self.assertEqual(len(list((self.library.root / "audio").iterdir())), 1)

    def test_invalid_name_does_not_create_copy(self):
        for name in ("", "  ", "x" * 49, "坏\n名称", "隐藏\u200b名称"):
            with self.subTest(name=name), self.assertRaises(VoiceLibraryError):
                self.library.save(name, self.source)
        self.assertFalse(self.library.root.exists())

    def test_invalid_audio_and_silence_leave_no_entry(self):
        import numpy as np
        import soundfile as sf
        bad = self.root / "损坏.wav"
        bad.write_bytes(b"not an audio file")
        silent = self.root / "静音.wav"
        sf.write(silent, np.zeros(22050, dtype="float32"), 22050)
        for source in (bad, silent):
            with self.subTest(source=source), self.assertRaises(VoiceLibraryError):
                self.library.save("坏录音", source)
        self.assertEqual(self.library.load().presets, ())
        self.assertEqual(list((self.library.root / "audio").iterdir()), [])
        self.assert_original_untouched()

    def test_uncompressed_nonfinite_audio_is_rejected(self):
        import numpy as np
        import soundfile as sf
        path = self.root / "非法浮点.wav"
        sf.write(path, np.full(200, np.nan, dtype="float32"), 22050, subtype="FLOAT")
        with self.assertRaisesRegex(VoiceLibraryError, "异常采样"):
            self.library.save("异常", path)
        self.assertEqual(self.library.load().presets, ())

    def test_size_limit_is_checked_before_copy(self):
        with patch.object(VoiceLibrary, "MAX_BYTES", 16):
            with self.assertRaisesRegex(VoiceLibraryError, "不能超过"):
                self.library.save("大录音", self.source)
        self.assertFalse(self.library.root.exists())

    def test_failed_save_preserves_previous_index_and_removes_new_copy(self):
        preset = self.library.save("原音色", self.source)
        before = self.library.index_path.read_bytes()
        with patch.object(self.library, "_write", side_effect=OSError("模拟磁盘写入失败")):
            with self.assertRaises(OSError):
                self.library.save("新音色", self.source)
        self.assertEqual(self.library.index_path.read_bytes(), before)
        self.assertEqual(list((self.library.root / "audio").iterdir()), [self.library.audio_path(preset)])
        self.assert_original_untouched()

    def test_atomic_replace_failure_keeps_index_and_cleans_temporary(self):
        first = self.library.save("原音色", self.source)
        before = self.library.index_path.read_bytes()
        with patch("app.voice_library.os.replace", side_effect=OSError("模拟原子替换失败")):
            with self.assertRaises(OSError):
                self.library.select(None)
        self.assertEqual(self.library.index_path.read_bytes(), before)
        self.assertEqual(self.library.load().selected_id, first.id)
        self.assertFalse(list(self.library.root.glob(".library-*.tmp")))

    def test_select_and_clear_are_persistent(self):
        first = self.library.save("第一音色", self.source)
        self.library.save("第二音色", self.source)
        self.library.select(first.id)
        self.assertEqual(VoiceLibrary(self.library.root).load().selected_id, first.id)
        self.library.select(None)
        self.assertIsNone(self.library.load().selected_id)
        with self.assertRaisesRegex(VoiceLibraryError, "不存在"):
            self.library.select("unknown")

    def test_delete_selected_only_removes_its_copy(self):
        first = self.library.save("第一音色", self.source)
        second = self.library.save("第二音色", self.source)
        sentinel = self.library.root / "保留文件.txt"
        sentinel.write_text("不属于音色副本", encoding="utf-8")
        self.library.delete(second.id)
        self.assertFalse(self.library.audio_path(second).exists())
        self.assertTrue(self.library.audio_path(first).is_file())
        self.assertEqual(self.library.load().presets, (first,))
        self.assertIsNone(self.library.load().selected_id)
        self.assertTrue(sentinel.is_file())
        self.assert_original_untouched()

    def test_failed_delete_restores_audio_and_index(self):
        preset = self.library.save("保留音色", self.source)
        before = self.library.index_path.read_bytes()
        with patch.object(self.library, "_write", side_effect=OSError("模拟清单提交失败")):
            with self.assertRaises(OSError):
                self.library.delete(preset.id)
        self.assertEqual(self.library.index_path.read_bytes(), before)
        self.assertEqual(hashlib.sha256(self.library.audio_path(preset).read_bytes()).hexdigest(), self.source_hash)
        self.assertFalse(list((self.library.root / "audio").glob(".delete-*")))

    def test_locked_copy_keeps_original_entry(self):
        preset = self.library.save("占用音色", self.source)
        before = self.library.index_path.read_bytes()
        original_unlink = Path.unlink

        def occupied(path, *args, **kwargs):
            if path.name.startswith(".delete-"):
                raise PermissionError("模拟文件占用")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", occupied):
            with self.assertRaises(PermissionError):
                self.library.delete(preset.id)
        self.assertEqual(self.library.index_path.read_bytes(), before)
        self.assertTrue(self.library.audio_path(preset).is_file())

    def test_missing_audio_is_visible_but_can_be_deleted(self):
        preset = self.library.save("丢失音色", self.source)
        self.library.audio_path(preset).unlink()
        self.assertEqual(self.library.load().presets, (preset,))
        with self.assertRaisesRegex(VoiceLibraryError, "丢失"):
            self.library.select(preset.id)
        self.library.delete(preset.id)
        self.assertEqual(self.library.load().presets, ())
        self.assert_original_untouched()

    def test_corrupt_index_is_preserved_and_blocks_mutation(self):
        self.library.save("原音色", self.source)
        self.library.index_path.write_text("{broken", encoding="utf-8")
        before = self.library.index_path.read_bytes()
        with self.assertRaisesRegex(VoiceLibraryError, "保留原文件"):
            self.library.save("新音色", self.source)
        self.assertEqual(self.library.index_path.read_bytes(), before)

    def test_path_escape_in_index_cannot_delete_original(self):
        preset = self.library.save("安全音色", self.source)
        payload = json.loads(self.library.index_path.read_text(encoding="utf-8"))
        for path in ("../../原始录音.wav", str(self.source.resolve()), "audio/../原始录音.wav"):
            with self.subTest(path=path):
                payload["presets"][0]["audio_file"] = path
                self.library.index_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(VoiceLibraryError):
                    self.library.delete(preset.id)
                self.assert_original_untouched()

    def test_audio_path_rejects_mismatched_id(self):
        preset = self.library.save("安全音色", self.source)
        with self.assertRaises(VoiceLibraryError):
            self.library.audio_path(replace(preset, id="../escape"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
