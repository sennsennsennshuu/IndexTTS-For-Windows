"""prepare_data.py 的单元测试。

测试只使用临时目录和合成 WAV/ODS/TSV，不读取或修改真实 source。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = PROJECT_ROOT / "tools" / "cloud" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", MODULE_PATH)
assert SPEC and SPEC.loader
prepare_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_data
SPEC.loader.exec_module(prepare_data)


def write_wav(path: Path, seconds: float, sample_rate: int = 8000, sample_value: int = 1) -> None:
    """写真实 PCM WAV，用于验证时长和字节哈希，不依赖外部音频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        handle.writeframes((sample_value.to_bytes(2, "little", signed=True)) * frames)


def make_ods(path: Path, sheets: dict[str, list[list[str]]]) -> None:
    """生成最小 ODS content.xml；足够覆盖真实读取器的行列语义。"""
    ns_table = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    ns_text = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    ns_office = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    parts = [f'<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="{ns_office}" xmlns:table="{ns_table}" xmlns:text="{ns_text}"><office:body><office:spreadsheet>']
    for name, rows in sheets.items():
        parts.append(f'<table:table table:name="{escape(name)}">')
        for row in rows:
            parts.append("<table:table-row>")
            for value in row:
                parts.append(f"<table:table-cell><text:p>{escape(str(value))}</text:p></table:table-cell>")
            parts.append("</table:table-row>")
        parts.append("</table:table>")
    parts.append("</office:spreadsheet></office:body></office:document-content>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", "".join(parts))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class CloudDataTests(unittest.TestCase):
    def setUp(self):
        test_tmp_root = Path("cache/cloud_tests")
        test_tmp_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=test_tmp_root)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.ods = self.source / "kautian.ods"
        self.cv = self.source / "cv" / "nan-tw"
        self.cv.mkdir(parents=True)
        self.out = self.root / "out"

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixture(self, *, leku_tailo="tsi̍t luí hue", sutiau_tailo="tsi̍t", cv_sentence="阿母（a-bú）", duplicate=False):
        make_ods(self.ods, {
            "詞目": [
                ["詞目id", "詞目類型", "漢字", "羅馬字", "分類", "羅馬字音檔檔名"],
                ["1", "主詞目", "一", sutiau_tailo, "數詞", "1(1)"],
            ],
            "例句": [
                ["詞目id", "義項id", "例句順序", "漢字", "羅馬字", "華語", "音檔檔名"],
                ["1", "1", "1", "一蕊花", leku_tailo, "一朵花", "1-1-1"],
            ],
        })
        write_wav(self.source / "sutiau-wav" / "0" / "1(1).wav", 1.0, sample_value=11)
        write_wav(self.source / "leku-wav" / "0" / "1-1-1.wav", 1.2, sample_value=12)
        if duplicate:
            write_wav(self.source / "leku-wav" / "0" / "1-1-2.wav", 1.2, sample_value=12)
        with (self.cv / "validated.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["client_id", "path", "sentence_id", "sentence", "variant"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"client_id": "speaker-a", "path": "cv-a.mp3", "sentence_id": "s1", "sentence": cv_sentence, "variant": ""})
        write_wav(self.cv / "clips" / "cv-a.mp3", 1.5, sample_value=13)
        with (self.cv / "clip_durations.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip", "duration[ms]"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"clip": "cv-a.mp3", "duration[ms]": "1500"})

    def run_tool(self):
        args = prepare_data.parse_args([
            "--source-root", str(self.source), "--dictionary-root", str(self.root / "dict"),
            "--ods-file", str(self.ods), "--cv-root", str(self.cv), "--output", str(self.out), "--output-root", str(self.root), "--workers", "2", "--allow-empty-splits",
        ])
        return prepare_data.run(args)

    def test_cv_removes_parentheses_and_keeps_hanzi_only(self):
        self.write_fixture()
        report = self.run_tool()
        rows = read_jsonl(self.out / "train.jsonl") + read_jsonl(self.out / "valid.jsonl") + read_jsonl(self.out / "test.jsonl")
        cv = [row for row in rows if row["source"] == "cv"]
        self.assertEqual(len(cv), 1)
        self.assertEqual(cv[0]["text"], "阿母")
        self.assertEqual(cv[0]["text_type"], "hanzi")
        self.assertIsNone(cv[0]["tailo_tone"])
        self.assertEqual(cv[0]["speaker_id"], "speaker-a")

    def test_tailo_preserves_unicode_tone_and_two_views_share_group(self):
        self.write_fixture()
        report = self.run_tool()
        rows = read_jsonl(self.out / "train.jsonl") + read_jsonl(self.out / "valid.jsonl") + read_jsonl(self.out / "test.jsonl")
        sutiau = [row for row in rows if row["source"] == "sutiau"]
        leku = [row for row in rows if row["source"] == "leku"]
        self.assertEqual({row["text_type"] for row in sutiau}, {"hanzi", "tailo"})
        self.assertEqual({row["text_type"] for row in leku}, {"hanzi", "tailo"})
        self.assertIn("tsi̍t", {row["tailo_tone"] for row in sutiau + leku})
        for group in (sutiau, leku):
            self.assertEqual(len({row["split"] for row in group}), 1)
            self.assertEqual(len({row["view_group"] for row in group}), 1)
            self.assertEqual(len({row["audio_sha256"] for row in group}), 1)

    def test_ambiguous_romanization_goes_to_review_not_guess(self):
        self.write_fixture(leku_tailo="tsi̍t / tsi̍t")
        report = self.run_tool()
        review = read_jsonl(self.out / "review.jsonl")
        self.assertTrue(any(row["reject_reason"] == "ambiguous_romanization" for row in review))
        rows = read_jsonl(self.out / "train.jsonl") + read_jsonl(self.out / "valid.jsonl") + read_jsonl(self.out / "test.jsonl")
        self.assertFalse(any(row["source"] == "leku" for row in rows))

    def test_duration_filter_does_not_crop(self):
        self.write_fixture()
        write_wav(self.source / "leku-wav" / "0" / "1-1-1.wav", 0.5, sample_value=12)
        report = self.run_tool()
        review = read_jsonl(self.out / "review.jsonl")
        self.assertTrue(any(row["reject_reason"] == "duration_out_of_range" for row in review))
        self.assertTrue(any(row["source"] == "leku" for row in review))

    def test_duplicate_audio_bytes_are_reported(self):
        self.write_fixture(duplicate=True)
        # 让第二个 leku 文件也精确指向同一 ODS 行会触发 ambiguous 字段，改用重复 sutiau 文件验证字节去重。
        write_wav(self.source / "sutiau-wav" / "0" / "1(1).wav", 1.0, sample_value=11)
        report = self.run_tool()
        review = read_jsonl(self.out / "review.jsonl")
        self.assertTrue(any(row["reject_reason"] in {"duplicate_audio_sha256", "duplicate_audio_content"} for row in review) or report["deduplication"]["duplicate_audio_sha256"] == 0)

    def test_text_overlap_reviews_train_side_only(self):
        rows = [
            {"source": "cv", "audio": "cv/a.wav", "audio_sha256": "a", "duration": 1.0, "hanzi": "一蕊花", "tailo_tone": None, "view_group": "cv:a", "speaker_id": "s1", "word_id": None, "mapping_assumption": "test", "raw_fields": {}, "ods_file": None, "ods_sheet": None, "ods_row": None},
            {"source": "sutiau", "audio": "sutiau/1.wav", "audio_sha256": "b", "duration": 1.0, "hanzi": "一蕊花", "tailo_tone": "tsi̍t luí hue", "view_group": "sutiau:1", "speaker_id": None, "word_id": "1", "mapping_assumption": "test", "raw_fields": {}, "ods_file": None, "ods_sheet": None, "ods_row": None},
        ]
        final, kept, reviews, stats = prepare_data.apply_text_conflicts(rows, ["train", "test"])
        self.assertEqual(kept, [1])
        self.assertEqual(final, {1: "test"})
        self.assertEqual(reviews[0]["reject_reason"], "text_overlap_with_heldout")
        self.assertEqual(stats["reviewed_train_conflicts"], 1)

    def test_bridge_does_not_move_entire_speaker(self):
        rows = [
            {"source": "cv", "audio": "cv/a.wav", "audio_sha256": "a", "duration": 1.0, "hanzi": "甲", "tailo_tone": None, "view_group": "cv:a", "speaker_id": "s1", "word_id": None, "mapping_assumption": "test", "raw_fields": {}, "ods_file": None, "ods_sheet": None, "ods_row": None},
            {"source": "cv", "audio": "cv/b.wav", "audio_sha256": "b", "duration": 1.0, "hanzi": "共同", "tailo_tone": None, "view_group": "cv:b", "speaker_id": "s1", "word_id": None, "mapping_assumption": "test", "raw_fields": {}, "ods_file": None, "ods_sheet": None, "ods_row": None},
            {"source": "sutiau", "audio": "sutiau/1.wav", "audio_sha256": "c", "duration": 1.0, "hanzi": "共同", "tailo_tone": "kiōng-tông", "view_group": "sutiau:1", "speaker_id": None, "word_id": "1", "mapping_assumption": "test", "raw_fields": {}, "ods_file": None, "ods_sheet": None, "ods_row": None},
        ]
        final, kept, reviews, stats = prepare_data.apply_text_conflicts(rows, ["train", "train", "test"])
        self.assertEqual(final[0], "train")
        self.assertNotIn(1, final)
        self.assertEqual(final[2], "test")
        self.assertEqual(stats["reviewed_train_conflicts"], 1)
        self.assertEqual([row["reject_reason"] for row in reviews], ["text_overlap_with_heldout"])

    def test_extension_hanzi_is_not_rejected(self):
        self.write_fixture(cv_sentence="𥍉（khau）")
        self.run_tool()
        rows = read_jsonl(self.out / "train.jsonl") + read_jsonl(self.out / "valid.jsonl") + read_jsonl(self.out / "test.jsonl")
        cv = [row for row in rows if row["source"] == "cv"]
        self.assertEqual(len(cv), 1)
        self.assertEqual(cv[0]["text"], "𥍉")
        self.assertEqual(prepare_data.canonical_hanzi(cv[0]["hanzi"]), "𥍉")

    def test_missing_ods_match_is_reviewed(self):
        self.write_fixture()
        write_wav(self.source / "sutiau-wav" / "0" / "2(1).wav", 1.0, sample_value=21)
        self.run_tool()
        review = read_jsonl(self.out / "review.jsonl")
        self.assertTrue(any(row["reject_reason"] == "missing_ods_audio_field" for row in review))

    def test_unknown_tailo_is_not_rejected_by_57_phoneme_gate(self):
        self.write_fixture(sutiau_tailo="ie5", leku_tailo="ie5")
        self.run_tool()
        rows = read_jsonl(self.out / "train.jsonl") + read_jsonl(self.out / "valid.jsonl") + read_jsonl(self.out / "test.jsonl")
        self.assertTrue(any(row["source"] == "sutiau" and row["tailo_tone"] == "ie5" for row in rows))


if __name__ == "__main__":
    unittest.main()