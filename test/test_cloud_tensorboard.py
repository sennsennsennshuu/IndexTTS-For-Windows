"""核对半行恢复、真实事件值和重启偏移，避免 TensorBoard 展示空面板或错曲线。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/cloud"))
from tensorboard_bridge import Bridge, read_complete_records, scalar_values


class TestTensorBoardBridge(unittest.TestCase):
    def setUp(self):
        self.original = Path.cwd()
        Path("cache/tmp").mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test_tensorboard_", dir="cache/tmp")
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.original)
        self.directory.cleanup()

    def test_partial_line_is_read_after_completion(self):
        path = Path("metrics.jsonl")
        first = b'{"step":1}\n'
        path.write_bytes(first + b'{"step":2')
        rows = list(read_complete_records(path, 0))
        self.assertEqual(rows, [(len(first), {"step": 1})])
        with path.open("ab") as stream:
            stream.write(b'}\n')
        self.assertEqual(list(read_complete_records(path, len(first)))[0][1], {"step": 2})

    def test_invalid_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            scalar_values({"loss_per_token": float("nan")})

    def test_checkpoint_validation_is_backfilled_without_duplicate(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        source = Path("training/runs/overfit/metrics.jsonl")
        source.parent.mkdir(parents=True)
        source.write_text('{"step":1,"loss_per_token":6.5}\n', encoding="utf-8")
        checkpoint = source.parent / "checkpoints/step_000050/checkpoint.json"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('{"step":50,"val_loss":6.0}', encoding="utf-8")
        bridge = Bridge()
        try:
            bridge.poll()
            # 模拟桥接读取末尾与检查点生成之间的竞争，下一轮才读到同一步验证行。
            with source.open("a", encoding="utf-8") as stream:
                stream.write('{"event":"validation","step":50,"val_loss":6.0}\n')
            bridge.poll()
        finally:
            bridge.close()
        values = EventAccumulator("training/tensorboard/pilot").Reload().Scalars("validation/loss")
        self.assertEqual([(event.step, event.value) for event in values], [(50, 6.0)])

    def test_real_events_resume_and_new_formal_run(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        source = Path("training/runs/overfit/metrics.jsonl")
        source.parent.mkdir(parents=True)
        source.write_text('{"event":"validation_initial","step":0,"val_loss":7.5}\n'
                          '{"step":1,"loss_per_token":6.5}\n', encoding="utf-8")
        Path("logs").mkdir()
        Path("logs/extraction_progress.json").write_text(json.dumps({"done": 25, "total": 100}), encoding="utf-8")
        bridge = Bridge()
        try:
            bridge.poll()
        finally:
            bridge.close()
        bridge = Bridge()
        try:
            bridge.poll()
            formal = Path("training/runs/minnan-lora/metrics.jsonl")
            formal.parent.mkdir(parents=True)
            formal.write_text('{"step":1,"loss_per_token":5.0}\n', encoding="utf-8")
            bridge.poll()
        finally:
            bridge.close()
        accumulator = EventAccumulator("training/tensorboard/pilot").Reload()
        self.assertEqual([event.value for event in accumulator.Scalars("train/loss_per_token")], [6.5])
        self.assertEqual(accumulator.Scalars("validation/loss")[0].step, 0)
        self.assertEqual(EventAccumulator("training/tensorboard/formal").Reload().Scalars("train/loss_per_token")[0].value, 5.0)
        self.assertEqual(EventAccumulator("training/tensorboard/features").Reload().Scalars("progress/percent")[0].value, 25.0)


if __name__ == "__main__":
    unittest.main()
