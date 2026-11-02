# 模型放置说明

Release 程序包不包含模型。请从 IndexTTS2 和各辅助模型的原发布方获取有权使用的权重，再按下面的项目相对路径放置。

```text
models/IndexTTS-2/
├── config.yaml
├── gpt.pth
├── s2mel.pth
├── bpe.model
├── wav2vec2bert_stats.pt
├── feat1.pt
├── feat2.pt
├── qwen_0.6b_emo4_merge/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── 其他原有 tokenizer 与配置文件
└── checkpoints/hf_cache/
    ├── models--facebook--w2v-bert-2.0/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    ├── models--amphion--MaskGCT/semantic_codec/model.safetensors
    ├── models--funasr--campplus/campplus_cn_common.bin
    └── models--nvidia--bigvgan_v2_22khz_80band_256x/
        ├── config.json
        └── bigvgan_generator.pt
```

主模型与情感模型来源为 `IndexTeam/IndexTTS-2`；辅助模型对应 `facebook/w2v-bert-2.0`、`amphion/MaskGCT`、`funasr/campplus`、`nvidia/bigvgan_v2_22khz_80band_256x`。这里使用直接包含文件的扁平目录，不是全局 Hugging Face 缓存的 `blobs/snapshots` 结构。

若原模型仓库已完整下载，可复制需要的文件；无需复制训练优化器、其他任务的权重、重复声码器或历史生成音频。现有模型文件的参考路径、大小及 SHA-256 见 `docs/model_manifest.json`。

所有模型放好后，在项目目录运行：

```powershell
.\run.bat --check
```

程序默认使用本地文件，不会将缺少模型静默回退到原助手或其他用户缓存。仅安装程序与 Python 依赖不能替代上述模型文件。
