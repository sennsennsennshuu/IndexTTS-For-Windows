"""核验监控的进程证据、增量读取与告警边界，不加载训练模型。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/cloud"))
from monitor import assess, collect_metrics, process_info, scalar_snapshot


class TestCloudMonitor(unittest.TestCase):
    def setUp(self):
        self.original = Path.cwd()
        Path("cache/tmp").mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test_monitor_", dir="cache/tmp")
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.original)
        self.directory.cleanup()

    def snapshot(self):
        return {"stage": "formal_training", "pipeline_process": {"alive": True, "identity_matches": True},
                "children": [{"alive": True}], "progress_age_seconds": 10,
                "disk": {"free_gib": 90, "used_percent": 40}, "gpu": {"temperature_c": 50},
                "training": {}, "speed_per_second": 2, "speed_unit": "optimizer_steps"}

    def test_missing_pid_is_not_a_live_job(self):
        self.assertFalse(process_info(2147483647)["alive"])

    @unittest.skipUnless(os.name == "posix", "真实 /proc 身份检查在 Linux 云端执行")
    def test_current_process_identity_is_verified(self):
        self.assertTrue(process_info(os.getpid())["alive"])
        self.assertTrue(process_info(os.getpid())["identity_matches"])
        self.assertFalse(process_info(os.getpid(), "tools/cloud/nonexistent.py")["identity_matches"])

    def test_incremental_metrics_partial_line_and_nonfinite(self):
        path = Path("training/runs/minnan-lora/metrics.jsonl")
        path.parent.mkdir(parents=True)
        path.write_text('{"event":"validation_initial","step":0,"val_loss":7.0}\n'
                        '{"step":1,"loss_per_token":6.0}\n{"step":2,', encoding="utf-8")
        state = {}
        first = collect_metrics(state)
        self.assertEqual(first["rolling_count"], 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write('"loss_per_token":4.0}\n{"step":3,"loss_per_token":NaN}\n')
        second = collect_metrics(state)
        self.assertEqual(second["rolling_loss_100"], 5.0)
        self.assertEqual(second["baseline_validation"], 7.0)
        self.assertEqual(second["invalid"], [{"step": 3, "field": "loss_per_token"}])
        self.assertEqual(collect_metrics(state)["rolling_count"], 2)
        json.dumps(second, allow_nan=False)

    def test_validation_and_resources_raise_specific_alerts(self):
        snapshot = self.snapshot()
        snapshot["training"] = {"baseline_validation": 5.0,
                                "recent_validation": [{"loss": 6}, {"loss": 7}, {"loss": 8}]}
        snapshot["disk"]["free_gib"] = 2
        snapshot["progress_age_seconds"] = 601
        alerts = {alert["code"]: alert["severity"] for alert in assess(snapshot)}
        self.assertEqual(alerts["disk_low"], "critical")
        self.assertIn("validation_rising", alerts)
        self.assertIn("validation_above_baseline", alerts)
        self.assertIn("progress_stalled", alerts)

    def test_completed_pipeline_is_not_reported_as_missing(self):
        snapshot = self.snapshot()
        snapshot.update({"stage": "complete", "pipeline_process": {"alive": False, "identity_matches": False},
                         "progress_age_seconds": 3600})
        snapshot["alerts"] = assess(snapshot)
        self.assertEqual(snapshot["alerts"], [])
        values = scalar_snapshot(snapshot)
        self.assertEqual(values["health/pipeline_alive"], 0)
        self.assertEqual(values["speed/optimizer_steps_per_second"], 2)


if __name__ == "__main__":
    unittest.main()
