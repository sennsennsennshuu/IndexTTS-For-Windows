"""云端文本前端的关键边界：保留声调、固定词表编号、禁止截断错配。"""

from pathlib import Path
import sys
import unittest
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/cloud"))
from runtime import canonical_text, encode_text, MinnanTextFrontend
from evaluate import MinnanTokenizer
from prepare_data import canonical_hanzi


class RecordingTokenizer:
    """只记录前端送入的完整文本，测试不加载真实模型。"""

    def __init__(self):
        self.received = None

    def encode(self, text, **kwargs):
        self.received = text
        return list(range(len(text)))


class TestCloudTextFrontend(unittest.TestCase):
    def test_tone_marks_survive_nfc_and_whitespace(self):
        text = "  Lí\t hó， tsi\u030dt  "
        normalized = canonical_text(text)
        self.assertEqual(normalized, "Lí hó， tsi̍t")
        self.assertTrue(unicodedata.is_normalized("NFC", normalized))

    def test_training_and_inference_share_existing_prefix(self):
        training, inference = RecordingTokenizer(), RecordingTokenizer()
        encode_text(training, "Lí hó，臺灣")
        MinnanTokenizer(inference).encode("<|minnan|> Lí hó，臺灣", allowed_special="all")
        self.assertEqual(training.received, inference.received)
        self.assertEqual(training.received, "<|zh|> Lí hó，臺灣")

    def test_long_transcript_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "禁止截断"):
            encode_text(RecordingTokenizer(), "Tâi-uân" * 100, maximum=10)

    def test_control_tokens_and_empty_text_are_rejected(self):
        for text in ("", "   ", "<|endoftext|>", "你好 <|zh|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                encode_text(RecordingTokenizer(), text)

    def test_romanized_dictionary_headwords_do_not_share_empty_key(self):
        self.assertNotEqual(canonical_hanzi("a-lú-mih"), canonical_hanzi("bi-tá-bín"))
        self.assertEqual(canonical_hanzi("Lí hó"), canonical_hanzi("lí-hó"))
        self.assertNotEqual(canonical_hanzi("Lí hó"), canonical_hanzi("Li ho"))

    def test_minnan_frontend_preserves_tone_and_punctuation_without_fst(self):
        frontend = MinnanTextFrontend().load()
        text = "Lí hó，𥍉 tsi̍t ē。"
        cleaned = frontend.clean_pattern.sub(lambda match: frontend.char_rep_map[match.group()], text)
        self.assertEqual(cleaned, text)
        self.assertEqual(frontend.normalize(text), text)


if __name__ == "__main__":
    unittest.main()
