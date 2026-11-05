"""台语文本处理门禁：真实简繁索引、歧义、未命中翻译与内部标记隔离。"""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.minnan import MinnanTokenizer
from app.minnan_words import MinnanLexicon, MinnanTextProcessor


class MinnanTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="cache", prefix="test_minnan_")
        self.folder = Path(self.temporary.name)
        entries = [
            {"id": "sentence-1", "hanzi": "你水斟了傷滇。", "tailo": "Lí tsuí thîn liáu siunn tīnn",
             "mandarin": ["你把水倒得太滿了。"], "verified": True, "source": "test"},
            {"id": "ambiguous-1", "hanzi": "行", "tailo": "kiânn", "mandarin": ["行"], "verified": True},
            {"id": "ambiguous-2", "hanzi": "行", "tailo": "hâng", "mandarin": ["行"], "verified": True},
        ]
        (self.folder / "entries.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries), encoding="utf-8")
        content = (self.folder / "entries.jsonl").read_bytes()
        (self.folder / "manifest.json").write_text(json.dumps({"outputs": {"entries": {
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}), encoding="utf-8")
        self.processor = MinnanTextProcessor(self.folder)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_translation_matches_simplified_and_preserves_tone(self):
        with patch("app.minnan_words.TaigiMinnanTranslator.translate") as translator:
            result = self.processor.prepare("你把水倒得太满了。")
        self.assertEqual(result.method, "dictionary_exact")
        self.assertEqual(result.text, "Lí tsuí thîn liáu siunn tīnn")
        translator.assert_not_called()

    def test_corrupted_dictionary_is_rejected_before_translation(self):
        with (self.folder / "entries.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.processor.prepare("你把水倒得太满了。")

    def test_new_sentence_uses_translator_instead_of_mandarin_passthrough(self):
        with patch("app.minnan_words.TaigiMinnanTranslator.translate", return_value="我今仔日欲出門。") as translator:
            result = self.processor.prepare("我今天要出门。")
        translator.assert_called_once_with("我今天要出门。")
        self.assertEqual(result.text, "我今仔日欲出門。")
        self.assertTrue(result.warnings)

    def test_ambiguous_reading_is_not_selected_arbitrarily(self):
        rows = self.processor.lexicon.find("行")
        self.assertIsNone(MinnanLexicon.unique_reading(rows))
        result = self.processor.prepare("行", "hanzi")
        self.assertEqual(result.text, "行")
        self.assertTrue(result.unresolved)

    def test_translation_failure_is_not_silently_synthesized(self):
        with patch("app.minnan_words.TaigiMinnanTranslator.translate", side_effect=RuntimeError("模型加载失败")):
            with self.assertRaisesRegex(RuntimeError, "模型加载失败"):
                self.processor.prepare("辞典中没有的新句子")

    def test_no_translation_mode_uses_dictionary_and_rejects_unknown_sentences(self):
        with patch("app.minnan_words.TaigiMinnanTranslator.translate") as full, patch("app.minnan_words.LocalMinnanTranslator.translate") as compact:
            known = self.processor.prepare("你把水倒得太满了。", translation_mode="none")
            self.assertEqual(known.method, "dictionary_exact")
            with self.assertRaisesRegex(ValueError, "不使用翻译模型"):
                self.processor.prepare("词库没有的全新普通话句子", translation_mode="none")
            full.assert_not_called()
            compact.assert_not_called()

    def test_compact_translation_is_explicit_and_releases_full_model(self):
        with patch("app.minnan_words.TaigiMinnanTranslator.unload") as unload, patch("app.minnan_words.LocalMinnanTranslator.translate", return_value="Guá beh tshut-mn̂g.") as translator:
            result = self.processor.prepare("我想要出门。", translation_mode="compact")
            self.assertEqual(result.method, "compact_translation")
            self.assertTrue(result.warnings)
            translator.assert_called_once()
            unload.assert_called_once()

    def test_dictionary_only_unknown_text_fails_before_loading_speech_model(self):
        from app.engine import IndexTTSEngine
        reference = self.folder / "reference.wav"
        reference.write_bytes(b"reference")
        engine = IndexTTSEngine()
        engine._minnan_processor = self.processor
        with patch.object(engine, "load_model") as loader:
            with self.assertRaisesRegex(ValueError, "不使用翻译模型"):
                engine.infer("词库没有的新句子", reference, lang="ZH-NAN-TW", minnan_translation_mode="none")
            loader.assert_not_called()

    def test_tailo_does_not_require_dictionary_or_rewrite_diacritics(self):
        processor = MinnanTextProcessor(self.folder / "missing")
        result = processor.prepare("Tsi̍t-sì-lâng", "tailo")
        self.assertEqual(result.text, "Tsi̍t-sì-lâng")
        self.assertFalse(processor.lexicon.loaded)

    def test_control_tokens_and_wrong_input_mode_are_rejected(self):
        for text, mode in [("<|zh|> test", "tailo"), ("中文", "tailo"), ("test", "invalid")]:
            with self.subTest(text=text, mode=mode), self.assertRaises(ValueError):
                self.processor.prepare(text, mode)

    def test_tokenizer_uses_training_prefix_without_changing_other_languages(self):
        class FakeTokenizer:
            def encode(self, text, **kwargs):
                return text
        tokenizer = MinnanTokenizer(FakeTokenizer())
        self.assertEqual(tokenizer.encode("<|minnan|> Lí"), "<|zh|> Lí")
        self.assertEqual(tokenizer.encode("<|en|> Hello"), "<|en|> Hello")

    def test_low_vram_encoder_returns_to_cpu_after_success_and_error(self):
        import torch
        from unittest.mock import Mock
        from indextts.infer_v2_5 import IndexTTS2
        features = torch.ones((1, 2, 3))
        encoder = Mock(return_value=SimpleNamespace(hidden_states=[features]*18))
        instance = SimpleNamespace(low_vram=True, semantic_model=encoder, device="cpu",
                                   semantic_mean=torch.zeros(3), semantic_std=torch.ones(3))
        output = IndexTTS2.get_emb(instance, features, torch.ones((1, 2)))
        self.assertTrue(torch.equal(output, features))
        self.assertEqual(encoder.to.call_args.args, ("cpu",))
        encoder.side_effect = RuntimeError("编码失败")
        with self.assertRaisesRegex(RuntimeError, "编码失败"):
            IndexTTS2.get_emb(instance, features, torch.ones((1, 2)))
        self.assertEqual(encoder.to.call_args.args, ("cpu",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
