"""2.5 升级回归：新接口、惰性加载、缓存失效、发音标注及无损峰值保护。"""
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime, MODEL_DIR, validate_models
prepare_runtime()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from app.engine import IndexTTSEngine
from app.window import IndexTTSWindow
from PyQt6.QtWidgets import QApplication


class UpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="cache", prefix="test_v25_")
        self.folder = Path(self.temporary.name)
        self.reference = self.folder / "参考.wav"
        self.reference.write_bytes(b"reference")
        self.engine = IndexTTSEngine()
        self.engine.is_loaded = True
        self.engine.tts_model = Mock(qwen_emo=None, cfg=SimpleNamespace(qwen_emo_path="qwen0.6bemo4-merge"))
        self.engine.tts_model.normalize_emo_vec.side_effect = lambda values, **_: values
        self.engine.tts_model.infer.side_effect = self.render

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def render(**kwargs):
        """输出可解码 PCM 文件，使原子提交与解码检查实际参与测试。"""
        import wave
        with wave.open(kwargs["output_path"], "wb") as wav:
            wav.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x10\0" * 100)

    def test_language_duration_and_annotations_reach_engine_unchanged(self):
        text = "<minute|M IH1 . N AH0 T>"
        for lang in ("ZH", "EN", "JA", "ES", "AR"):
            self.engine.infer(text, self.reference, self.folder / f"{lang}.wav", lang=lang, duration_factor=1.25)
            call = self.engine.tts_model.infer.call_args.kwargs
            self.assertEqual((call["lang"], call["duration_factor"], call["text"]), (lang, 1.25, text))
        self.assertIsNone(self.engine.tts_model.qwen_emo)

    def test_invalid_controls_fail_before_model_load(self):
        with patch.object(self.engine, "load_model") as loader:
            for factor in (0.49, 2.01, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    self.engine.infer("你好", self.reference, duration_factor=factor)
            with self.assertRaises(ValueError):
                self.engine.infer("你好", self.reference, lang="FR")
            with self.assertRaises(ValueError):
                self.engine.infer("你好", self.reference, emo_mode=2, emo_vector=[float("nan")] * 8)
            loader.assert_not_called()

    def test_emotion_classifier_is_loaded_once_on_cpu(self):
        with patch("indextts.infer_v2_5.QwenEmotion") as classifier:
            self.engine.infer("你好", self.reference, self.folder / "first.wav", emo_mode=3)
            self.engine.infer("再见", self.reference, self.folder / "second.wav", emo_mode=3)
            classifier.assert_called_once_with(str(MODEL_DIR / "qwen0.6bemo4-merge"), device="cpu")
        self.assertTrue(self.engine.tts_model.infer.call_args.kwargs["use_emo_text"])

    def test_failed_render_preserves_existing_audio_and_removes_partial(self):
        target = self.folder / "结果.wav"
        target.write_bytes(b"previous audio")
        def fail(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"broken")
            raise RuntimeError("预期的模拟失败")
        self.engine.tts_model.infer.side_effect = fail
        with self.assertRaises(RuntimeError):
            self.engine.infer("你好", self.reference, target)
        self.assertEqual(target.read_bytes(), b"previous audio")
        self.assertFalse(list(self.folder.glob("*.partial.wav")))

    def test_overwritten_reference_invalidates_condition_cache(self):
        self.engine.infer("你好", self.reference, self.folder / "first.wav")
        self.engine.tts_model.cache_spk_cond = "previous condition"
        self.reference.write_bytes(b"a new recording of a different length")
        self.engine.infer("你好", self.reference, self.folder / "second.wav")
        self.assertIsNone(self.engine.tts_model.cache_spk_cond)

    def test_optional_emotion_files_do_not_block_other_modes(self):
        model = self.folder / "模型"
        # 仅建立 2.5 核心模型的结构替身，验证文字情感确为可选项。
        for path in MODEL_DIR.rglob("*"):
            if path.is_file() and "qwen0.6bemo4-merge" not in path.parts:
                target = model / path.relative_to(MODEL_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes() if path.suffix in {".yaml", ".json"} else b"x" * 512)
        self.assertEqual(validate_models(model), 14)
        with self.assertRaises(FileNotFoundError):
            validate_models(model, require_emotion=True)
        (model / "config.yaml").write_text("version: 2.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不是 IndexTTS 2.5"):
            validate_models(model)

    def test_ui_controls_and_pronunciation_help(self):
        window = IndexTTSWindow()
        try:
            self.assertEqual([window.language_combo.itemData(i) for i in range(5)], ["ZH", "EN", "JA", "ES", "AR"])
            self.assertEqual((window.duration_spin.minimum(), window.duration_spin.maximum(), window.duration_spin.value()), (0.5, 2.0, 1.0))
            self.assertTrue(window.pronunciation_help.isHidden())
            window.pronunciation_toggle.click()
            self.assertFalse(window.pronunciation_help.isHidden())
            self.assertIn("<上手|じょうず>", window.pronunciation_help.text())
            self.assertIn("2.5", window.windowTitle())
        finally:
            window.close()

    def test_cmu_annotations_survive_low_vram_prechunking(self):
        from indextts.infer_v2_5 import IndexTTS2, apply_pronunciation_annotations
        annotation = "<minute|M IH1 . N AH0 T>"
        text = "This introduction is long. " + annotation + " refers to time."
        chunks = IndexTTS2.split_text_by_punctuation(text, 20)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(any(annotation in chunk for chunk in chunks))
        self.assertIn("SPECIAL_TOKEN_1", apply_pronunciation_annotations(annotation))
        self.assertEqual(apply_pronunciation_annotations("<上手|じょうず>"), " じょうず ")

    def test_peak_protection_preserves_waveform_shape(self):
        import torch
        from indextts.utils.common import attenuate_audio_peak
        audio = torch.tensor([[-2.0, -0.4, 0.0, 0.8, 2.0]])
        protected = attenuate_audio_peak(audio)
        self.assertTrue(torch.allclose(protected, audio * 0.49))
        self.assertLessEqual(float(protected.abs().max()), 0.980001)
        quiet = torch.tensor([[0.1, -0.2]])
        self.assertTrue(torch.equal(attenuate_audio_peak(quiet), quiet))
        with self.assertRaises(RuntimeError):
            attenuate_audio_peak(torch.tensor([float("nan")]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
