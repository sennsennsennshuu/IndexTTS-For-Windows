"""只验证交付门禁、成对覆盖和文件篡改拒绝；真实语音另由云端推理验收。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/cloud"))
from delivery_checks import delivery_ready, digest, validate_cases, validate_package, validate_report


def cases():
    result = []
    for index in range(4):
        for kind in ("hanzi", "tailo"):
            result.append({"name": f"heldout_{index}_{kind}", "split": "test", "text_type": kind,
                           "text": f"範例{index}_{kind}", "target_audio_sha256": str(index) * 64,
                           "reference_audio_sha256": str((index + 1) % 4) * 64,
                           "reference_audio": "reference.wav", "evaluation_scope": "heldout_text_paired_views"})
    return result


def report(inputs):
    return {"adapter": "checkpoint", "cases": [
        {"name": row["name"], "text": row["text"], "text_type": row["text_type"],
         "target_audio_sha256": row["target_audio_sha256"], "evaluation_scope": row["evaluation_scope"],
         "reference": row["reference_audio"], "finite": True, "seconds": 0.3, "rms": 0.3,
         "peak": 0.31, "sample_rate": 22050} for row in inputs]}


class TestCloudDelivery(unittest.TestCase):
    def test_old_pipeline_complete_is_not_a_ready_delivery(self):
        self.assertFalse(delivery_ready({"stage": "complete", "sha256": "a" * 64}))
        self.assertTrue(delivery_ready({"stage": "complete", "paired_heldout_verified": True,
                                        "paired_cases": 8, "sampling_audit_steps": 15000}))

    def test_missing_tailo_and_validation_preview_are_rejected(self):
        inputs = cases()
        with self.assertRaises(ValueError):
            validate_cases(inputs[::2])
        with self.assertRaises(ValueError):
            validate_cases([{**row, "split": "valid"} for row in inputs])

    def test_wrong_checkpoint_or_transcript_is_rejected(self):
        inputs = cases()
        result = report(inputs)
        with self.assertRaises(ValueError):
            validate_report(inputs, result, "different_checkpoint")
        result["cases"][0]["text"] = "不相符"
        with self.assertRaises(ValueError):
            validate_report(inputs, result, "checkpoint")

    def test_case_name_cannot_escape_audio_directory(self):
        inputs = cases()
        inputs[0]["name"] = "../../outside"
        with self.assertRaises(ValueError):
            validate_cases(inputs)

    def test_package_files_and_sampling_are_verified(self):
        Path("cache/tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_delivery_", dir="cache/tmp") as directory:
            root = Path(directory)
            inputs = cases()
            payloads = {"adapter/adapter_model.safetensors": b"unit-test-adapter",
                        "extra_state.pt": b"unit-test-language-state",
                        "sampling_audit.json": json.dumps({"optimizer_step": 15000, "sampler_cursor": 120000,
                            "completed_epochs": 2, "all_text_views_seen": True, "unique_text_views_seen": 74741,
                            "matched_last_sample_steps": 15000, "unseen_text_views": 0, "data_fingerprint_matches": True}).encode(),
                        "heldout_audio/cases.json": json.dumps(inputs).encode(),
                        "heldout_audio/report.json": json.dumps(report(inputs)).encode()}
            for name, content in payloads.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            for index, case in enumerate(inputs):
                path = root / "heldout_audio" / f"{index:02d}_{case['name']}.wav"
                with wave.open(str(path), "wb") as stream:
                    stream.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                    stream.writeframes(b"\x10\x27" * 6615)
            entries = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
                       for path in root.rglob("*") if path.is_file()]
            manifest = {"paired_heldout_verified": True, "paired_cases": 8, "sampling_audit_steps": 15000, "files": entries}
            (root / "delivery_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_package(root)["paired_cases"], 8)
            (root / "delivery_receipt.json").write_text('{"sha256":"local-receipt"}', encoding="utf-8")
            self.assertEqual(validate_package(root)["paired_cases"], 8)
            (root / "unexpected.txt").write_text("unlisted", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_package(root)
            (root / "unexpected.txt").unlink()
            (root / "extra_state.pt").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                validate_package(root)


if __name__ == "__main__":
    unittest.main()
