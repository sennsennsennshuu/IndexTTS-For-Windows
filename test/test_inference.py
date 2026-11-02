"""显式执行的真实离线合成验收：四种情感模式共用同一个已加载引擎。"""
from pathlib import Path
import argparse
import json
import logging
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
from app.engine import IndexTTSEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="test/fixtures/reference.wav")
    parser.add_argument("--all-modes", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(filename="logs/inference_test.log", encoding="utf-8", level=logging.INFO)
    import numpy as np
    import soundfile as sf
    import torch
    report = {"offline": True, "cuda": torch.cuda.is_available(), "cases": []}
    engine = IndexTTSEngine()
    try:
        start = time.perf_counter()
        engine.load_model()
        report["model_load_seconds"] = round(time.perf_counter() - start, 3)
        model_identity = id(engine.tts_model)
        modes = range(4) if args.all_modes else [0]
        for mode in modes:
            start = time.perf_counter()
            output = f"outputs/独立项目验收_模式{mode}.wav"
            path = engine.infer(
                "你好，这是独立项目的语音合成测试。", args.reference, output_path=output,
                emo_mode=mode, emo_audio=args.reference,
                emo_vector=[0.2, 0, 0, 0, 0, 0, 0, 0.1], emo_text="平静自然",
                temperature=0.8, top_p=0.8, top_k=30, max_text_tokens_per_segment=120)
            audio, sr = sf.read(path)
            case = {"mode": mode, "path": path, "seconds": round(len(audio)/sr, 3),
                    "sample_rate": sr, "channels": 1 if audio.ndim == 1 else audio.shape[1],
                    "peak": float(np.max(np.abs(audio))),
                    "rms": float(np.sqrt(np.mean(audio ** 2))),
                    "finite": bool(np.isfinite(audio).all()),
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "model_reused": id(engine.tts_model) == model_identity}
            if not case["finite"] or case["seconds"] < 0.2 or case["rms"] <= 1e-5:
                raise RuntimeError(f"音频验收失败：{case}")
            report["cases"].append(case)
            Path("docs/inference_verification.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(case, ensure_ascii=False), flush=True)
    finally:
        engine.unload_model()


if __name__ == "__main__":
    main()
