"""显式执行的真实离线合成验收：四种情感模式共用同一个已加载引擎。"""
from pathlib import Path
import argparse
import json
import logging
import sys
import time
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import prepare_runtime
prepare_runtime()
from app.engine import IndexTTSEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="test/fixtures/reference.wav")
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--multilingual", action="store_true", help="实际合成五种语言")
    parser.add_argument("--duration-check", action="store_true", help="固定随机种子比较 0.8 与 1.2 时长系数")
    parser.add_argument("--pronunciation-only", action="store_true", help="只验收三种发音标注，包含低显存英文长文本分段")
    args = parser.parse_args()
    report_path = Path("docs/pronunciation_verification.json" if args.pronunciation_only else "docs/inference_verification.json")
    logging.basicConfig(filename="logs/inference_test.log", encoding="utf-8", level=logging.INFO)
    import numpy as np
    import soundfile as sf
    import torch
    report = {"model_version": "2.5", "offline": True, "cuda": torch.cuda.is_available(), "cases": []}
    engine = IndexTTSEngine()
    try:
        start = time.perf_counter()
        engine.load_model()
        report["model_load_seconds"] = round(time.perf_counter() - start, 3)
        model_identity = id(engine.tts_model)
        modes = range(4) if args.all_modes else [0]
        cases = [(f"情感模式{mode}", "你好，这是新版语音合成测试。", "ZH", mode, 1.0) for mode in modes]
        if args.multilingual:
            cases += [(f"语言_{lang}", text, lang, 0, 1.0) for lang, text in (
                ("EN", "Hello, welcome to our voice studio."),
                ("JA", "こんにちは。今日は良い天気ですね。"),
                ("ES", "Hola, bienvenidos a nuestro estudio de voz."),
                ("AR", "مرحبا بكم في استوديو الصوت."))]
        if args.duration_check:
            cases += [(f"时长_{factor}", "欢迎来到语音工作台。", "ZH", 0, factor) for factor in (0.8, 1.2)]
        if args.pronunciation_only:
            cases = [("拼音标注", "他在银<行|HANG2>工作。", "ZH", 0, 1.0),
                     ("音素标注", "Welcome to the studio. Please give me a <minute|M IH1 . N AH0 T> to begin.", "EN", 0, 1.0),
                     ("假名标注", "彼は料理が<上手|じょうず>です。", "JA", 0, 1.0)]
        for name, text, lang, mode, factor in cases:
            start = time.perf_counter()
            output = f"outputs/2.5验收_{name}.wav"
            # 两种时长控制使用相同随机种子，避免文本采样差异污染时长对比。
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            path = engine.infer(
                text, args.reference, output_path=output, lang=lang, duration_factor=factor,
                emo_mode=mode, emo_audio=args.reference,
                emo_vector=[0.2, 0, 0, 0, 0, 0, 0, 0.1], emo_text="平静自然",
                temperature=0.8, top_p=0.8, top_k=30, max_text_tokens_per_segment=120)
            audio, sr = sf.read(path)
            case = {"name": name, "mode": mode, "lang": lang, "duration_factor": factor,
                    "path": path, "seconds": round(len(audio)/sr, 3),
                    "sample_rate": sr, "channels": 1 if audio.ndim == 1 else audio.shape[1],
                    "peak": float(np.max(np.abs(audio))),
                    "rms": float(np.sqrt(np.mean(audio ** 2))),
                    "finite": bool(np.isfinite(audio).all()),
                    "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "model_reused": id(engine.tts_model) == model_identity,
                    "qwen_loaded": engine.tts_model.qwen_emo is not None,
                    "peak_cuda_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1) if torch.cuda.is_available() else 0}
            if not case["finite"] or case["seconds"] < 0.2 or case["rms"] <= 1e-5 or case["clipped_fraction"] > 0.0001:
                raise RuntimeError(f"音频验收失败：{case}")
            report["cases"].append(case)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(case, ensure_ascii=False), flush=True)
        if args.duration_check:
            durations = {case["duration_factor"]: case["seconds"] for case in report["cases"] if case["name"].startswith("时长_")}
            report["duration_ratio_1_2_over_0_8"] = round(durations[1.2] / durations[0.8], 3)
            if report["duration_ratio_1_2_over_0_8"] < 1.25:
                raise RuntimeError("时长控制未产生预期方向或幅度的变化")
        report["all_passed"] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        engine.unload_model()


if __name__ == "__main__":
    main()
