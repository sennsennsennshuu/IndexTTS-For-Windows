"""把已验证的 200 步试训适配器及真实试听结果打成私有交付包。"""

import hashlib
import json
from pathlib import Path
import tarfile


def main() -> None:
    checkpoint = Path("training/runs/overfit/checkpoints/step_000200")
    output = Path("outputs/minnan-pilot-200steps.tar.gz")
    if not (checkpoint / "adapter/adapter_model.safetensors").exists():
        raise FileNotFoundError("200 步检查点尚未生成")
    if not json.loads(Path("logs/overfit_gate.json").read_text())["passed"]:
        raise ValueError("小样本可学习性门未通过")
    with tarfile.open(output, "w:gz") as archive:
        archive.add(checkpoint / "adapter", arcname="adapter")
        archive.add(checkpoint / "extra_state.pt", arcname="extra_state.pt")
        archive.add("outputs/baseline", arcname="baseline_audio")
        archive.add("outputs/overfit", arcname="adapted_audio")
        archive.add("outputs/codec_check", arcname="codec_reconstruction")
        archive.add("logs/overfit_gate.json", arcname="overfit_gate.json")
        archive.add("training/configs/overfit.json", arcname="training_config.json")
        archive.add("training/base_model_manifest.json", arcname="base_model_manifest.json")
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {"package": str(output), "bytes": output.stat().st_size, "sha256": digest,
               "training_steps": 200, "training_unique_audio": 8, "scope": "overfit_pilot_only",
               "native_listening_passed": False, "formal_training_complete": False}
    Path("outputs/pilot_delivery.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
