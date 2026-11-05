"""真实 GPU 验收：中文基座、台罗适配器、中文回切，脚本只在直接执行时运行。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.paths import prepare_runtime


def main():
    prepare_runtime()
    import numpy as np
    import soundfile as sf
    import torch
    from app.engine import IndexTTSEngine

    folder = Path("outputs/minnan_language_pack_validation")
    folder.mkdir(parents=True, exist_ok=True)
    reference = Path("models/extra_language/zh-nan-tw/model/heldout_audio/original_audio/e063fd6482c1ca793587313422420859722f39f6353a36da89a31a6efe14bf89.wav")
    engine = IndexTTSEngine()
    engine.progress_callback = lambda text: print(text, flush=True)
    cases = [("zh_before", "你好，这是中文测试。", "ZH", "hanzi"),
             ("nan_tailo", "Lí tsuí thîn liáu siunn tīnn.", "ZH-NAN-TW", "tailo"),
             ("zh_after", "你好，这是中文测试。", "ZH", "hanzi")]
    reports = []
    try:
        for name, text, language, mode in cases:
            torch.manual_seed(20261105)
            np.random.seed(20261105)
            random.seed(20261105)
            started = time.monotonic()
            path = engine.infer(text, reference, folder / f"{name}.wav", lang=language,
                                minnan_input_mode=mode, do_sample=False, num_beams=1, max_mel_tokens=350)
            samples, rate = sf.read(path, dtype="float32")
            assert rate == 22050 and np.isfinite(samples).all() and len(samples) > 4410
            assert float(np.sqrt(np.mean(samples ** 2))) > 1e-5
            row = {"case": name, "input": text, "language": language, "file": path,
                   "seconds": len(samples) / rate, "elapsed": time.monotonic() - started,
                   "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            reports.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        before, rate = sf.read(folder / "zh_before.wav", dtype="float32")
        after, _ = sf.read(folder / "zh_after.wav", dtype="float32")
        same = before.shape == after.shape and np.allclose(before, after, atol=1e-6, rtol=0)
        result = {"cases": reports, "chinese_roundtrip_audio_equal": bool(same),
                  "adapter_loaded": engine._minnan_adapter.loaded, "native_listening_passed": False}
        Path("logs/minnan_language_inference.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert same, "中文回切音频不一致，需要排查 LoRA 或参考缓存残留"
    finally:
        engine.unload_model()


if __name__ == "__main__":
    main()
