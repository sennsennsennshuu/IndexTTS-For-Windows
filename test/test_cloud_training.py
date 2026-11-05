"""train.py 的 CPU mock 测试，不加载真实 GPT、codec 或语料。"""

from __future__ import annotations

import json
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# -I 隔离模式不会把项目根目录放入 sys.path；测试自身必须显式保证可导入。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.cloud.train as train


class TinyPosition(nn.Module):
    """模拟 LearnedPositionEmbeddings 的 .emb 接口。"""

    def __init__(self, count: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(count, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.emb(positions).unsqueeze(0)


class TinyGPT(nn.Module):
    """只把输入 embedding 原样传出，便于验证 head 之前的切片。"""

    def __init__(self, dim: int, positions: int = 100):
        super().__init__()
        self.config = SimpleNamespace(n_positions=positions)
        self.last_inputs: torch.Tensor | None = None

    def forward(self, inputs_embeds: torch.Tensor, **kwargs):
        self.last_inputs = inputs_embeds
        # 加入全局摘要，使 mock 的 mel 段也能感知 text/lang delta；真实 GPT 由因果注意力完成。
        mixed = inputs_embeds + inputs_embeds.sum(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=mixed)


class TinyModel:
    """满足 build_tensors/forward_loss 所需字段的最小模型。"""

    def __init__(self, dim: int = 8, vocab: int = 32):
        self.start_text_token = 0
        self.stop_text_token = 1
        self.start_mel_token = 2
        self.stop_mel_token = 3
        self.text_embedding = nn.Embedding(64, dim)
        self.mel_embedding = nn.Embedding(64, dim)
        self.lang_embedding = train.MinnanLanguageAdapter(nn.Embedding(107, dim))
        self.text_pos_embedding = TinyPosition(64, dim)
        self.mel_pos_embedding = TinyPosition(64, dim)
        self.spk_emb_proj = nn.Linear(192, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.mel_head = nn.Linear(dim, vocab)
        self.gpt = TinyGPT(dim)


class FakeAdapterSaver:
    """模拟 PeftModel.save_pretrained 的最小输出。"""

    def __init__(self):
        self.calls = 0

    def save_pretrained(self, path: str, **kwargs):
        self.calls += 1
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text('{"r": 32, "fan_in_fan_out": true}\n', encoding="utf-8")
        torch.save({"mock": True}, target / "adapter_model.bin")


class CloudTrainingTests(unittest.TestCase):
    def test_batch_sampler_consumes_distinct_audio_and_advances_cursor(self):
        groups = {str(index): [{"audio_sha256": str(index), "text_type": "hanzi", "text": str(index)}] for index in range(8)}
        sampler = train.ViewSampler(groups, seed=1)
        first, cursor = train.take_batch(sampler, 0, 4)
        second, cursor = train.take_batch(sampler, cursor, 4)
        self.assertEqual(cursor, 8)
        self.assertEqual(len({row["audio_sha256"] for row in first + second}), 8)

    def test_ragged_batch_matches_real_causal_gpt_losses_and_gradients(self):
        from transformers import GPT2Config, GPT2Model

        torch.manual_seed(13)
        model = TinyModel()
        model.gpt = GPT2Model(GPT2Config(vocab_size=32, n_embd=8, n_head=2, n_layer=2, n_positions=128,
                                        resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)).eval()
        specs = [([6], [4, 5]), ([7, 8, 9], [6, 7, 8, 9]), ([10, 11], [10]), ([12, 13, 14, 15], [11, 12, 13])]
        def make_inputs():
            return [train.build_tensors(model, torch.tensor(text), torch.tensor(codes), torch.tensor([100]),
                                        torch.zeros(192), torch.zeros(8)) for text, codes in specs]
        individual = [train.forward_loss(model, item) for item in make_inputs()]
        reference_loss = torch.stack([result["loss_per_token"] for result in individual]).mean()
        reference_loss.backward()
        reference_lang_grad = model.lang_embedding.delta.grad.clone()
        reference_gpt_grad = model.gpt.h[0].attn.c_attn.weight.grad.clone()
        model.gpt.zero_grad(set_to_none=True)
        model.lang_embedding.delta.grad = None
        items = make_inputs()
        result = train.forward_loss_batch(model, items)
        self.assertTrue(torch.allclose(reference_loss, result["loss_mean"], atol=2e-5, rtol=2e-5))
        result["loss_mean"].backward()
        self.assertTrue(torch.allclose(reference_lang_grad, model.lang_embedding.delta.grad, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(reference_gpt_grad, model.gpt.h[0].attn.c_attn.weight.grad, atol=2e-5, rtol=2e-5))
        self.assertEqual(result["tokens"], sum(len(codes) + 1 for _, codes in specs))
        self.assertEqual(result["utterances"], 4)
        for row, item in zip(result["attention_mask"], items):
            length = item["total_len"]
            self.assertTrue(bool((row[:length] == 1).all()))
            self.assertTrue(bool((row[length:] == 0).all()))

    def test_invalid_batch_settings_are_rejected(self):
        for config in ({"batch_size": 0}, {"batch_size": -2}, {"grad_accum": 0}):
            with self.assertRaises(ValueError):
                train._validate_batch_settings(config)

    def test_teacher_forcing_has_one_eos_and_correct_slice(self):
        model = TinyModel()
        text_in, audio_in, target = train.make_teacher_forcing_sequences([4, 5], [6, 7], model)
        self.assertEqual(text_in.shape, (1, 4))
        self.assertEqual(audio_in.shape, (1, 3))
        self.assertEqual(target.shape, (1, 3))
        self.assertEqual(text_in.tolist(), [[0, 6, 7, 1]])
        self.assertEqual(audio_in.tolist(), [[2, 4, 5]])
        self.assertEqual(target.tolist(), [[4, 5, 3]])
        self.assertEqual(audio_in.numel(), target.numel())

        tensors = train.build_tensors(
            model,
            torch.tensor([6, 7]),
            torch.tensor([4, 5]),
            torch.tensor([train.MINNAN_INDEX]),
            torch.randn(192),
            torch.randn(8),
        )
        self.assertEqual(tensors["cond_len"], 3)
        self.assertEqual(tensors["text_len"], 4)
        self.assertEqual(tensors["mel_len"], 3)
        self.assertEqual(tensors["total_len"], 10)
        result = train.forward_loss(model, tensors)
        self.assertEqual(result["logits"].shape, (1, 32, 3))
        self.assertEqual(result["target"].tolist(), [[4, 5, 3]])
        manual = F.cross_entropy(result["logits"].float(), result["target"], reduction="sum")
        self.assertTrue(torch.allclose(result["loss_sum"], manual))
        self.assertEqual(model.gpt.last_inputs.shape[1], 10)


    def test_evaluate_rebuilds_current_language_embedding(self):
        class Store:
            def get(self, row):
                return {
                    "codes": torch.tensor([4, 5], dtype=torch.long),
                    "speaker": torch.randn(192),
                    "emotion": torch.randn(8),
                }

        class Refs:
            def refs(self, row):
                return None, None

        class Tokenizer:
            def encode(self, text, allowed_special="all"):
                return [6, 7]

        model = TinyModel()
        row = {
            "audio_sha256": "a" * 64, "speaker_id": None, "text": "测试",
            "text_type": "Hanzi", "split": "valid", "duration": 1.0,
        }
        neutral = torch.zeros(8)
        first = train.evaluate(model, [row], Store(), Refs(), Tokenizer(), neutral, torch.device("cpu"), "fp32", 598)
        with torch.no_grad():
            model.lang_embedding.delta.add_(1.0)
        second = train.evaluate(model, [row], Store(), Refs(), Tokenizer(), neutral, torch.device("cpu"), "fp32", 598)
        self.assertNotEqual(first, second)

    def test_only_minnan_row_delta_gets_gradient(self):
        embedding = nn.Embedding(107, 4)
        with torch.no_grad():
            embedding.weight.zero_()
            embedding.weight[1].fill_(2.0)
            embedding.weight[99].fill_(9.0)
        adapter = train.MinnanLanguageAdapter(embedding, index=train.MINNAN_INDEX, source_index=train.ZH_INDEX)
        self.assertFalse(embedding.weight.requires_grad)
        self.assertTrue(adapter.delta.requires_grad)
        self.assertTrue(torch.equal(embedding.weight[train.MINNAN_INDEX], embedding.weight[train.ZH_INDEX]))

        out = adapter(torch.tensor([train.MINNAN_INDEX, 99]))
        out.sum().backward()
        self.assertIsNone(embedding.weight.grad)
        self.assertIsNotNone(adapter.delta.grad)
        self.assertGreater(float(adapter.delta.grad.abs().sum()), 0.0)

        before = embedding.weight.detach().clone()
        optimizer = torch.optim.SGD([adapter.delta], lr=0.1)
        optimizer.step()
        after = embedding.weight.detach()
        # 只有 delta 改变；基表 106 行全部保持不变。
        self.assertTrue(torch.equal(after[:train.MINNAN_INDEX], before[:train.MINNAN_INDEX]))
        self.assertTrue(torch.equal(after[train.MINNAN_INDEX + 1:], before[train.MINNAN_INDEX + 1:]))

    def test_checkpoint_extra_state_roundtrip(self):
        temp_root = Path("cache/cloud_tests")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            root = Path(temporary)
            embedding = nn.Embedding(107, 4)
            with torch.no_grad():
                embedding.weight[1].fill_(0.5)
            adapter = train.MinnanLanguageAdapter(embedding)
            with torch.no_grad():
                adapter.delta.add_(0.25)
            fake = FakeAdapterSaver()
            checkpoint = train.save_checkpoint(
                root,
                step=3,
                gpt_model=fake,
                lang_adapter=adapter,
                optimizer=None,
                scheduler=None,
                config={"rank": 32, "alpha": 64, "target_modules": train.TARGET_MODULES},
                base_identity={"schema": 1, "gpt_sha256": "a" * 64},
                data_fingerprint="data",
                feature_fingerprint="feature",
                cursor=7,
                best_val_loss=0.9,
                val_loss=0.8,
                is_best=True,
            )
            self.assertEqual(fake.calls, 1)
            self.assertTrue((checkpoint / "adapter" / "adapter_config.json").is_file())
            self.assertTrue((root / "best.json").is_file())
            extra, trainer_state, meta = train.read_checkpoint(checkpoint)
            self.assertEqual(meta["step"], 3)
            self.assertEqual(trainer_state["cursor"], 7)
            self.assertEqual(extra["feature_fingerprint"], "feature")

            new_embedding = nn.Embedding(107, 4)
            new_adapter = train.MinnanLanguageAdapter(new_embedding)
            new_adapter.load_state(extra["lang"])
            self.assertTrue(torch.allclose(new_adapter.export_state()["actual_row"], adapter.export_state()["actual_row"]))

    def test_unknown_speaker_uses_no_reference_and_known_speaker_uses_distinct_refs(self):
        rows = [
            {"audio_sha256": "a" * 64, "text_type": "Hanzi", "text": "a", "split": "train", "speaker_id": "s1"},
            {"audio_sha256": "b" * 64, "text_type": "TaiLo", "text": "b", "split": "train", "speaker_id": "s1"},
            {"audio_sha256": "c" * 64, "text_type": "Hanzi", "text": "c", "split": "train", "speaker_id": "s1"},
            {"audio_sha256": "d" * 64, "text_type": "Hanzi", "text": "d", "split": "train", "speaker_id": None},
        ]
        index = train.ReferenceIndex(rows, seed=1)
        speaker_ref, emotion_ref = index.refs(rows[0])
        self.assertNotEqual(speaker_ref["audio_sha256"], emotion_ref["audio_sha256"])
        self.assertNotEqual(speaker_ref["audio_sha256"], rows[0]["audio_sha256"])
        self.assertEqual(index.refs(rows[3]), (None, None))

    def test_validation_sample_covers_sources_and_writing_modes(self):
        rows = []
        for source, modes in (("cv", ("hanzi",)), ("leku", ("hanzi", "tailo")), ("sutiau", ("hanzi", "tailo"))):
            for mode in modes:
                for index in range(20):
                    rows.append({"source": source, "text_type": mode, "audio_sha256": f"{source}:{mode}:{index}"})
        selected = train.select_validation_rows(rows, 10, 1234)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({(row["source"], row["text_type"]) for row in selected}), 5)
        self.assertEqual(selected, train.select_validation_rows(list(reversed(rows)), 10, 1234))

    def test_sparse_speaker_refs_use_neutral_without_crossing_split(self):
        rows = [{"audio_sha256": str(index) * 64, "text_type": "hanzi", "text": str(index),
                 "split": "train" if index < 2 else "test", "speaker_id": "real-speaker"} for index in range(3)]
        index = train.ReferenceIndex(rows, seed=1)
        speaker, emotion = index.refs(rows[0])
        self.assertEqual(speaker["audio_sha256"], rows[1]["audio_sha256"])
        self.assertIsNone(emotion)
        self.assertEqual(index.refs(rows[2]), (None, None))
        self.assertEqual(rows[0]["speaker_id"], "real-speaker")

    def test_pruning_keeps_old_best_even_when_new_checkpoint_is_worse(self):
        temp_root = Path("cache/cloud_tests")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            root = Path(temporary)
            adapter = train.MinnanLanguageAdapter(nn.Embedding(107, 4))
            for step in range(5):
                train.save_checkpoint(root, step, FakeAdapterSaver(), adapter, None, None, {}, {},
                                      "data", "features", step, 0.5, val_loss=0.5 + step, is_best=step == 0)
            self.assertTrue((root / "checkpoints/step_000000").is_dir())
            self.assertFalse((root / "checkpoints/step_000001").exists())
            self.assertTrue((root / "checkpoints/step_000004").is_dir())

    @unittest.skipUnless(importlib.util.find_spec("peft"), "真实 PEFT 测试在已安装云端环境执行")
    def test_real_peft_checkpointing_with_external_frozen_embeddings(self):
        from transformers import GPT2Config, GPT2Model
        from safetensors.torch import load_file

        torch.manual_seed(9)
        base = GPT2Model(GPT2Config(vocab_size=16, n_embd=16, n_layer=1, n_head=2, n_positions=32))
        # 复现外部 mel embedding：基础 GPT 原 wte 删除后挂接真实冻结别名。
        del base.wte
        base.wte = nn.Embedding(20, 16)
        base.requires_grad_(False)
        identical_base = copy.deepcopy(base)
        model = train._inject_lora(base, {"rank": 2, "alpha": 4, "dropout": 0.0})
        train._enable_gradient_checkpointing(model, True)
        features = torch.randn(1, 5, 16, requires_grad=True)
        output = model(inputs_embeds=features, use_cache=False).last_hidden_state
        output[:, :, 0].sum().backward()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters))
        self.assertFalse(base.wte.weight.requires_grad)
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        optimizer.step()
        model.eval()
        expected = model(inputs_embeds=features, use_cache=False).last_hidden_state.detach()
        temp_root = Path("cache/cloud_tests")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
            directory = Path(temporary) / "adapter"
            train._save_adapter(model, directory)
            saved = load_file(directory / "adapter_model.safetensors")
            self.assertTrue(saved and all("lora_" in name for name in saved))
            restored = train._load_adapter(identical_base, directory)
            restored.eval()
            actual = restored(inputs_embeds=features, use_cache=False).last_hidden_state.detach()
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
