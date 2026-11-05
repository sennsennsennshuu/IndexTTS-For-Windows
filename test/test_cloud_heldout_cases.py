"""未见文本验收必须覆盖成对汉字和带调台罗，且参考不能与目标重合。"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/cloud"))
from heldout_cases import select_paired_cases


class TestHeldoutCases(unittest.TestCase):
    def rows(self):
        result = []
        for index in range(5):
            for kind, text in (("hanzi", f"臺語例句{index}"), ("tailo", f"Tâi-gí lí-hó {index}")):
                result.append({"audio_sha256": str(index) * 64, "audio": f"training/audio/{index}.wav",
                               "text_type": kind, "text": text, "split": "test", "source": "leku",
                               "duration": 3.0, "word_id": str(index), "speaker_id": None})
        return result

    def test_pairs_keep_tones_and_share_a_different_reference(self):
        cases = select_paired_cases(self.rows())
        self.assertEqual(len(cases), 8)
        for begin in range(0, 8, 2):
            hanzi, tailo = cases[begin:begin + 2]
            self.assertEqual((hanzi["text_type"], tailo["text_type"]), ("hanzi", "tailo"))
            self.assertEqual(hanzi["target_audio_sha256"], tailo["target_audio_sha256"])
            self.assertEqual(hanzi["reference_audio"], tailo["reference_audio"])
            self.assertNotEqual(hanzi["target_audio_sha256"], hanzi["reference_audio_sha256"])
            self.assertIn("Tâi-gí", tailo["text"])

    def test_order_is_independent_of_manifest_enumeration(self):
        rows = self.rows()
        self.assertEqual(select_paired_cases(rows), select_paired_cases(list(reversed(rows))))

    def test_train_rows_and_unpaired_views_cannot_fill_quota(self):
        rows = self.rows()[:6]
        rows += [{**row, "split": "train"} for row in self.rows()[6:]]
        with self.assertRaises(ValueError):
            select_paired_cases(rows)
        with self.assertRaises(ValueError):
            select_paired_cases([row for row in self.rows() if row["text_type"] == "hanzi"])

    def test_conflicting_transcripts_are_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            select_paired_cases(rows + [{**rows[0], "text": "不同轉寫"}])

    def test_validation_preview_cannot_read_test_rows(self):
        valid = [{**row, "split": "valid", "audio_sha256": "v" + row["audio_sha256"][1:]} for row in self.rows()]
        cases = select_paired_cases(self.rows() + valid, utterances=1, split="valid")
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(row["split"] == "valid" and row["target_audio_sha256"].startswith("v") for row in cases))
        self.assertTrue(all(row["evaluation_scope"] == "validation_preview_paired_views" for row in cases))
        with self.assertRaises(ValueError):
            select_paired_cases(self.rows(), utterances=1, split="valid")

    def test_train_is_not_an_allowed_preview_split(self):
        with self.assertRaises(ValueError):
            select_paired_cases(self.rows(), utterances=1, split="train")


if __name__ == "__main__":
    unittest.main()
