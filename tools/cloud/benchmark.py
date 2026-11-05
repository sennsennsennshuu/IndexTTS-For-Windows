"""用同一真实数据和相同有效 batch 比较训练吞吐，不以减少样本数伪造提速。"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 10:
        raise ValueError("至少 10 步，预留初始化与测量区间")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ["TMPDIR"] = str(Path("cache/tmp").resolve())
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    original = json.loads(Path("training/configs/overfit.json").read_text(encoding="utf-8"))
    run_id = str(time.time_ns())
    results = []
    for batch in (1, 4):
        config = {**original, "batch_size": batch, "grad_accum": 8 // batch, "steps": args.steps,
                  "validate_every": 1000, "save_every": 1000, "max_valid": 4,
                  "output_dir": f"training/runs/batch_benchmark_{batch}_{run_id}", "warmup_steps": 3}
        path = Path(f"training/configs/benchmark_{batch}.json")
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        with Path(f"logs/batch_benchmark_{batch}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(["even/bin/python", "-u", "tools/cloud/train.py", "--config", str(path)],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        metrics = [json.loads(line) for line in Path(config["output_dir"], "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        updates = [row for row in metrics if row.get("step", 0) > 0 and "loss_per_token" in row]
        if len(updates) != args.steps or not all(math.isfinite(row["loss_per_token"]) for row in updates):
            raise ValueError("真实更新数或 loss 检查失败")
        seconds = (updates[-1]["elapsed"] - updates[4]["elapsed"]) / (args.steps - 5)
        progress = json.loads(Path(config["output_dir"], "progress.json").read_text(encoding="utf-8"))
        import torch
        state = torch.load(Path(progress["last_checkpoint"]) / "trainer_state.pt", map_location="cpu", weights_only=False)
        if state["cursor"] != args.steps * 8:
            raise ValueError("两个方案实际采样数不同，不能报告为等价提速")
        result = {"batch_size": batch, "grad_accum": 8 // batch, "effective_batch": 8,
                  "output_dir": config["output_dir"],
                  "steps": args.steps, "actual_audio_samples": state["cursor"],
                  "seconds_per_update": seconds, "samples_per_second": 8 / seconds,
                  "initial_ce": sum(row["loss_per_token"] for row in updates[:5]) / 5,
                  "final_ce": sum(row["loss_per_token"] for row in updates[-5:]) / 5,
                  "gpu": progress["gpu"], "measured_while_feature_extraction_active": True}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"runs": results, "speedup": results[0]["seconds_per_update"] / results[1]["seconds_per_update"],
              "same_effective_batch": True, "measured_unix": time.time()}
    Path("logs/batch_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
