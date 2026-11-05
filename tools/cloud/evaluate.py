"""为基础模型、codec 重建和台语适配器生成可试听的固定案例。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import configure, MinnanTextFrontend


class MinnanTokenizer:
    """在训练与推理中使用同一既有前缀，不修改基础词表的任何编号。"""

    def __init__(self, tokenizer):
        self.base = tokenizer

    def encode(self, text, *args, **kwargs):
        if text.startswith("<|minnan|> "):
            text = "<|zh|> " + text[len("<|minnan|> "):]
        return self.base.encode(unicodedata.normalize("NFC", text), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


def attach_adapter(tts, checkpoint: Path) -> None:
    """恢复 LoRA 和专用语言行，再重建引用新 transformer 的推理包装器。"""
    import torch
    from peft import PeftModel
    from indextts.utils.tokenizer import LANGUAGE_DICT

    assert LANGUAGE_DICT["minnan"] == 100
    # 语言嵌入在 transformer 外部，不属于 PEFT 目录，必须恢复单独保存的实际语言行。
    extras = torch.load(checkpoint / "extra_state.pt", map_location="cpu", weights_only=True)
    if extras.get("format") != 1 or extras["lang"]["index"] != 100:
        raise ValueError("不支持的适配器附加状态或语言编号")
    fingerprint = hashlib.sha256(Path("training/base_model_manifest.json").read_bytes()).hexdigest()
    if extras["base_identity"]["model_fingerprint"] != fingerprint:
        raise ValueError("适配器与当前基础模型清单不匹配")
    row = extras["lang"]["actual_row"]
    if tuple(row.shape) != tuple(tts.gpt.lang_embedding.weight[100].shape):
        raise ValueError("台语语言嵌入维度不匹配")
    with torch.no_grad():
        tts.gpt.lang_embedding.weight[100].copy_(row.to(tts.device))
    tts.gpt.gpt = PeftModel.from_pretrained(tts.gpt.gpt, str(checkpoint / "adapter"), is_trainable=False)
    tts.gpt.gpt.eval()
    tts.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=tts.use_bf16)


def waveform_report(path: Path) -> dict:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    report = {"path": path.as_posix(), "seconds": len(samples) / sample_rate,
              "finite": bool(np.isfinite(samples).all()), "rms": float(np.sqrt(np.mean(samples ** 2))),
              "peak": float(np.max(np.abs(samples))), "sample_rate": sample_rate}
    if not report["finite"] or report["rms"] < 1e-5 or report["seconds"] < 0.2:
        raise RuntimeError(f"合成未通过基础音频检查：{report}")
    return report


def reconstruct(tts, target: str, reference: str, destination: Path) -> dict:
    """把真实目标语音经过 codec 和冻结声学链重建，参考音频使用另一条录音。"""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from indextts.utils.common import attenuate_audio_peak

    def read(path, sr):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.mean(1).copy()).unsqueeze(0)
        if rate != sr:
            waveform = torchaudio.functional.resample(waveform, rate, sr)
        return waveform.to(tts.device)

    with torch.inference_mode():
        target_audio = read(target, 16000)
        target_inputs = tts.extract_features(target_audio.cpu(), sampling_rate=16000, return_tensors="pt")
        semantic = tts.get_emb(target_inputs["input_features"].to(tts.device), target_inputs["attention_mask"].to(tts.device))
        codes, _ = tts.semantic_codec.quantize(semantic)
        decoded = tts.semantic_codec.decode(codes)
        # 重建使用真实目标时长，避免把 GPT 自由合成的经验时长比例当作重建真值。
        length = torch.tensor([round(target_audio.shape[-1] / 16000 * 22050 / 256)], device=tts.device)
        condition = tts.s2mel.models["length_regulator"](decoded, ylens=length, n_quantizers=3, f0=None)[0]
        reference_16 = read(reference, 16000)
        reference_22 = read(reference, 22050)
        ref_inputs = tts.extract_features(reference_16.cpu(), sampling_rate=16000, return_tensors="pt")
        ref_semantic = tts.get_emb(ref_inputs["input_features"].to(tts.device), ref_inputs["attention_mask"].to(tts.device))
        ref_mel = tts.mel_fn(reference_22.float())
        prompt = tts.s2mel.models["length_regulator"](ref_semantic, ylens=torch.tensor([ref_mel.shape[-1]], device=tts.device), n_quantizers=3, f0=None)[0]
        fbank = torchaudio.compliance.kaldi.fbank(reference_16, num_mel_bins=80, dither=0, sample_frequency=16000)
        fbank -= fbank.mean(0, keepdim=True)
        style = tts.campplus_model(fbank.unsqueeze(0))
        merged = torch.cat((prompt, condition), dim=1)
        mel = tts.s2mel.models["cfm"].inference(merged, torch.tensor([merged.shape[1]], device=tts.device), ref_mel, style, None, 25, inference_cfg_rate=0.7)
        waveform = tts.bigvgan(mel[:, :, ref_mel.shape[-1]:].float()).squeeze()
        waveform = attenuate_audio_peak(waveform).float().cpu().numpy()
        sf.write(destination, waveform, 22050, subtype="PCM_16")
        return {**waveform_report(destination), "mode": "codec_reconstruction", "target": target, "reference": reference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--codec-only", action="store_true")
    args = parser.parse_args()
    configure()
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    torch.manual_seed(20261105)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    # 在输出包内保存真正使用的输入清单，不能仅根据旧文件名推断验收覆盖范围。
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    tts = IndexTTS2(cfg_path="models/IndexTTS-2.5/config.yaml", model_dir="models/IndexTTS-2.5", device="cuda:0", use_bf16=True, use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False,
                    text_processor=MinnanTextFrontend())
    with torch.no_grad():
        # 与训练的未学习起点完全相同，避免使用随机 minnan 行作为对照。
        tts.gpt.lang_embedding.weight[100].copy_(tts.gpt.lang_embedding.weight[1])
    tts.tokenizer = MinnanTokenizer(tts.tokenizer)
    if args.adapter:
        attach_adapter(tts, args.adapter)
    reports = []
    started = time.monotonic()
    for index, case in enumerate(cases):
        torch.manual_seed(20261105 + index)
        destination = args.output / f"{index:02d}_{case['name']}.wav"
        if args.codec_only:
            reports.append(reconstruct(tts, case["target_audio"], case["reference_audio"], destination))
        else:
            tts.infer(spk_audio_prompt=case["reference_audio"], text=case["text"], output_path=str(destination),
                      lang="minnan", text_normalization=False, do_sample=True, temperature=0.7,
                      top_p=0.8, top_k=30, max_mel_tokens=900, num_beams=1)
            reports.append({**waveform_report(destination), "text": case["text"], "reference": case["reference_audio"],
                            "name": case["name"], "text_type": case.get("text_type"),
                            "target_audio_sha256": case.get("target_audio_sha256"),
                            "evaluation_scope": case.get("evaluation_scope")})
    result = {"cases": reports, "elapsed": time.monotonic() - started,
              "adapter": str(args.adapter) if args.adapter else None,
              "native_listening_passed": False, "automated_audio_checks_only": True}
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
