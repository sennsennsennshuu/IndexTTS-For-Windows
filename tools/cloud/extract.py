"""可恢复地预提取 2.5 语义 token、说话人和情感条件，逐条绑定模型与音频摘要。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import FeatureRuntime, configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-dir", default="models/IndexTTS-2.5")
    parser.add_argument("--manifests", nargs="+", default=["training/data/minnan/train.cloud.jsonl", "training/data/minnan/valid.cloud.jsonl", "training/data/minnan/test.cloud.jsonl"])
    args = parser.parse_args()
    configure()
    import torch
    from indextts.utils.tokenizer import get_tokenizer

    rows = {}
    for filename in args.manifests:
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.setdefault(row["audio_sha256"], row)
    selected = list(rows.values())
    if args.limit:
        selected = selected[:args.limit]
    manifest_bytes = Path("training/base_model_manifest.json").read_bytes()
    model_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    # 每个特征文件附带所依赖的实际固定基础模型标识；不能复用 2.0 或其他修订缓存。
    expected = json.loads(manifest_bytes)["files"]
    signature = {"schema": 1, "model_fingerprint": model_fingerprint,
                 "gpt_sha256": expected["gpt.pth"]["sha256"], "codec_sha256": expected["codec.pth"]["sha256"],
                 "features": "w2vbert_layer17_normalized_codec25hz_campplus192_emovec1280"}
    output = Path("training/features")
    output.mkdir(parents=True, exist_ok=True)
    runtime = None
    start = time.monotonic()
    completed = 0
    reused = 0
    for index, row in enumerate(selected, 1):
        target = output / f"{row['audio_sha256']}.pt"
        item_signature = {**signature, "source_audio_sha256": row["audio_sha256"],
                          "cloud_audio_sha256": row.get("cloud_audio_sha256", row["audio_sha256"])}
        if target.is_file():
            cache = torch.load(target, map_location="cpu", weights_only=True)
            if cache.get("signature") == item_signature:
                reused += 1
                continue
            raise RuntimeError(f"特征签名不一致，拒绝覆盖：{target}")
        with Path(row["audio"]).open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != item_signature["cloud_audio_sha256"]:
            raise RuntimeError(f"映射后音频内容变化：{row['audio']}")
        if runtime is None:
            runtime = FeatureRuntime(args.model_dir)
        features = runtime.extract(row["audio"], expected_duration=float(row["duration"]))
        features["signature"] = item_signature
        temporary = target.with_suffix(".pt.part")
        torch.save(features, temporary)
        temporary.replace(target)
        completed += 1
        if index % 25 == 0 or index == len(selected):
            report = {"done": index, "total": len(selected), "new": completed, "reused": reused,
                      "elapsed": time.monotonic() - start, "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "complete": index == len(selected), "limited": bool(args.limit), "signature": signature}
            Path("logs/extraction_progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
    final = {"done": len(selected), "total": len(selected), "new": completed, "reused": reused,
             "complete": True, "limited": bool(args.limit), "signature": signature}
    Path("logs/extraction_progress.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
