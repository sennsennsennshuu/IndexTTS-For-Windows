"""台湾闽南语词库构建产物与字段边界的回归测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORDS_DIR = PROJECT_ROOT / "models" / "extra_language" / "zh-nan-tw" / "words"
ENTRIES_PATH = WORDS_DIR / "entries.jsonl"
MANIFEST_PATH = WORDS_DIR / "manifest.json"
SQLITE_PATH = WORDS_DIR / "lexicon.sqlite3"
BUILDER_PATH = PROJECT_ROOT / "tools" / "build_minnan_words.py"
EXPECTED_FIELDS = [
    "id", "hanzi", "tailo", "mandarin", "source", "source_id",
    "license", "kind", "verified",
]


def file_sha256(path: Path) -> str:
    """分块计算 SHA-256，避免把大词库一次读入内存。"""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_builder():
    """按文件路径加载构建器，避免依赖尚未安装成包的 tools 目录。"""
    spec = importlib.util.spec_from_file_location("minnan_words_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载词库构建器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MinnanBuilderUnitTests(unittest.TestCase):
    """不访问网络的纯函数测试。"""

    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()

    def test_han_lo_headword_preserves_latin_and_punctuation(self):
        self.assertEqual(self.builder.extract_hanzi("食 pháng"), "食 pháng")
        self.assertEqual(self.builder.extract_hanzi("用齒抿仔洗喙時,愛先jio̍k 齒膏"),
                         "用齒抿仔洗喙時,愛先jio̍k 齒膏")
        self.assertEqual(self.builder.extract_hanzi("tsia̍h-tshài"), "")

    def test_split_tailo_keeps_each_candidate(self):
        self.assertEqual(
            self.builder.split_tailo_readings("tsia̍h(白)/si̍t(文)", ""),
            ["tsia̍h(白)", "si̍t(文)"],
        )
        self.assertEqual(self.builder.split_tailo_readings("", "tsi̍t-puànn"), ["tsi̍t-puànn"])

    def test_short_exact_translations_are_split_without_choosing_one(self):
        self.assertEqual(
            self.builder.extract_mandarin_candidates("吃、食、喝", max_length=16),
            ["吃", "食", "喝"],
        )

    def test_long_definition_is_not_misreported_as_exact_translation(self):
        self.assertEqual(
            self.builder.extract_mandarin_candidates(
                "指一種以米漿蒸熟，再一層一層加油蔥製成的食物。",
                max_length=24,
            ),
            [],
        )
        self.assertEqual(
            self.builder.extract_mandarin_candidates("例：食飯 tsia̍h pn̄g（吃飯）"),
            [],
        )

    def test_comma_sentence_stays_one_translation(self):
        text = "我昨天去散步，淋到了雨"
        self.assertEqual(self.builder.extract_mandarin_candidates(text, max_length=16), [text])

    def test_pehoe_whole_mandarin_field_is_preserved(self):
        text = "第十五章，第二十三節"
        self.assertEqual(
            self.builder.extract_mandarin_candidates(text, preserve_whole=True),
            [text],
        )

    def test_stable_id_is_independent_of_source_order(self):
        first = self.builder.make_entry(
            source="fixture",
            source_id="42",
            hanzi="食",
            tailo="tsia̍h",
            mandarin=["吃"],
            license="CC0-1.0",
            verified=True,
        )
        second = self.builder.make_entry(
            source="fixture",
            source_id="42",
            hanzi="食",
            tailo="tsia̍h",
            mandarin=["吃"],
            license="CC0-1.0",
            verified=True,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["id"], second["id"])

    def test_thuocl_boundary_is_explicit(self):
        entry = self.builder.make_entry(
            source="thuocl",
            source_id="data/THUOCL_food.txt:1",
            hanzi="",
            tailo="",
            mandarin=["土豆"],
            license="MIT",
            verified=False,
        )
        self.assertEqual(entry["hanzi"], "")
        self.assertEqual(entry["tailo"], "")
        self.assertFalse(entry["verified"])


class MinnanGeneratedArtifactTests(unittest.TestCase):
    """验证本次真实构建的产物、清单和 SHA 是否闭环。"""

    @classmethod
    def setUpClass(cls):
        missing = [str(path) for path in (ENTRIES_PATH, MANIFEST_PATH, SQLITE_PATH) if not path.is_file()]
        if missing:
            raise unittest.SkipTest("真实词库产物尚未生成：" + ", ".join(missing))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.rows = []
        with ENTRIES_PATH.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"entries.jsonl 第 {line_number} 行不是合法 JSON：{exc}") from exc
                cls.rows.append(row)

    def test_entry_schema_and_value_types(self):
        self.assertTrue(self.rows)
        ids = set()
        for index, row in enumerate(self.rows, start=1):
            self.assertEqual(list(row), EXPECTED_FIELDS, f"第 {index} 行字段或顺序错误")
            self.assertIsInstance(row["id"], str)
            self.assertTrue(row["id"])
            self.assertIsInstance(row["hanzi"], str)
            self.assertIsInstance(row["tailo"], str)
            self.assertIsInstance(row["mandarin"], list)
            self.assertTrue(all(isinstance(item, str) for item in row["mandarin"]))
            self.assertIsInstance(row["source"], str)
            self.assertIsInstance(row["source_id"], str)
            self.assertIsInstance(row["license"], str)
            self.assertIn(row["kind"], {"word", "sentence"})
            self.assertIsInstance(row["verified"], bool)
            self.assertNotIn(row["id"], ids, f"重复 ID：{row['id']}")
            ids.add(row["id"])

    def test_counts_match_manifest(self):
        summary = self.manifest["summary"]
        self.assertEqual(summary["entries"], len(self.rows))
        self.assertEqual(summary["with_tailo"], sum(bool(row["tailo"]) for row in self.rows))
        self.assertEqual(summary["with_mandarin"], sum(bool(row["mandarin"]) for row in self.rows))
        self.assertEqual(summary["verified"], sum(bool(row["verified"]) for row in self.rows))
        self.assertEqual(summary["word"], sum(row["kind"] == "word" for row in self.rows))
        self.assertEqual(summary["sentence"], sum(row["kind"] == "sentence" for row in self.rows))

    def test_output_and_raw_source_sha256(self):
        expected = self.manifest["outputs"]["entries"]
        self.assertEqual(expected["bytes"], ENTRIES_PATH.stat().st_size)
        self.assertEqual(expected["sha256"], file_sha256(ENTRIES_PATH))
        self.assertEqual(self.manifest["outputs"]["sqlite"]["sha256"], file_sha256(SQLITE_PATH))
        self.assertGreater(len(self.manifest["raw_files"]), 20)
        for item in self.manifest["raw_files"]:
            path = WORDS_DIR / item["path"]
            self.assertTrue(path.is_file(), f"缺少原始文件：{item['path']}")
            self.assertEqual(item["bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], file_sha256(path), f"原始文件 SHA 不符：{item['path']}")

    def test_sqlite_matches_jsonl(self):
        with sqlite3.connect(SQLITE_PATH) as connection:
            count = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        self.assertEqual(count, len(self.rows))
        self.assertEqual(count, self.manifest["outputs"]["sqlite"]["rows"])

    def test_thuocl_never_becomes_taiwanese_reading(self):
        candidates = [row for row in self.rows if row["source"] == "thuocl"]
        self.assertTrue(candidates)
        for row in candidates:
            self.assertEqual(row["hanzi"], "")
            self.assertEqual(row["tailo"], "")
            self.assertFalse(row["verified"])
            self.assertTrue(row["mandarin"])

    def test_verified_records_have_readings_and_are_indexable(self):
        for row in self.rows:
            if row["verified"]:
                self.assertTrue(row["tailo"], row["id"])
                self.assertTrue(row["hanzi"] or row["mandarin"], row["id"])

    def test_short_mandarin_fields_do_not_contain_explanations(self):
        for row in self.rows:
            if row["kind"] == "sentence" or row["source"] in {"thuocl", "chhoe_pehoe"}:
                continue
            for candidate in row["mandarin"]:
                self.assertLessEqual(len(candidate), 30, f"短词对译异常过长：{row['id']} {candidate!r}")
                self.assertNotIn("例：", candidate)
                self.assertNotIn("例:", candidate)
                self.assertNotIn("。", candidate)

    def test_known_records_and_sentence_alignment(self):
        self.assertTrue(any(
            row["hanzi"] == "𤺪呢"
            and row["tailo"] == "siān-neh"
            and row["mandarin"] == ["討厭"]
            and row["source"] == "chhoe_itaigi"
            for row in self.rows
        ))
        sentences = [row for row in self.rows if row["source"] == "icorpus_100"]
        self.assertEqual(len(sentences), 100)
        self.assertTrue(all(row["kind"] == "sentence" and row["hanzi"] and row["tailo"]
                            and row["mandarin"] for row in sentences))

    def test_real_mixed_hanlo_is_not_collapsed_to_hanzi_only(self):
        mixed = [
            row for row in self.rows
            if row["source"].startswith("chhoe_")
            and any("\u4e00" <= ch <= "\u9fff" for ch in row["hanzi"])
            and any("a" <= ch.lower() <= "z" for ch in row["hanzi"])
        ]
        self.assertTrue(mixed)
        self.assertTrue(any(re.search(r"[A-Za-zÀ-ž]", row["hanzi"]) for row in mixed))

    def test_real_pehoe_hoabun_with_comma_is_single_value(self):
        row = next(
            row for row in self.rows
            if row["source"] == "chhoe_pehoe" and row["source_id"] == "496"
        )
        self.assertEqual(row["mandarin"], ["第十五章，第二十三節"])

    def test_ambiguous_readings_are_preserved(self):
        readings = {
            row["tailo"]
            for row in self.rows
            if row["source"] == "pts_taigi_words"
            and row["source_id"].startswith("143:reading-")
        }
        self.assertIn("tshiau-tshuē", readings)
        self.assertIn("tshiau-tshē", readings)

    def test_all_required_source_states_exist(self):
        sources = self.manifest["sources"]
        self.assertEqual(sources["thuocl"]["status"], "downloaded_mandarin_candidates_only")
        self.assertEqual(sources["taiwan_tongues"]["status"], "downloaded_and_parsed_linked_dataset")
        self.assertEqual(sources["pts_taigi_words"]["status"], "downloaded_and_parsed")
        self.assertEqual(sources["audits"]["tggl"]["status"], "accessible_public_query_only")
        self.assertEqual(
            sources["audits"]["taiwan_lingu"]["status"],
            "accessible_but_download_restricted",
        )

    def test_manifest_hanzi_semantics_matches_preserved_hanlo_contract(self):
        description = self.manifest["field_semantics"]["hanzi"]
        self.assertIn("保留", description)
        self.assertIn("拉丁拼写", description)
        self.assertIn("标点", description)
        self.assertIn("仅完全无汉字时为空", description)
        self.assertNotIn("只提取汉字串", description)

    def test_license_boundaries_are_not_misrepresented(self):
        licenses = set(self.manifest["summary"]["by_license"])
        self.assertIn("CC0-1.0", licenses)
        self.assertIn("CC-BY-SA-4.0", licenses)
        self.assertIn("CC-BY-ND-3.0-TW", licenses)
        self.assertIn("CC-BY-NC-4.0", licenses)
        self.assertIn("MIT", licenses)
        self.assertTrue(any(value.startswith("Public website;") for value in licenses))
        self.assertNotIn("CC0", licenses)


if __name__ == "__main__":
    unittest.main(verbosity=2)