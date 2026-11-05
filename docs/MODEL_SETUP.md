# IndexTTS 2.5 模型准备

本程序只使用 **IndexTTS 2.5**。模型统一放在项目相对路径 `models/IndexTTS-2.5/`；不同版本的 GPT、codec、声学模型和 tokenizer 不可混用。

## 自动下载与校验

`run.bat` 负责安装 `even/` 运行环境。环境安装后，在项目目录执行：

```powershell
.\even\python.exe -X utf8 -I -B .\tools\download_models.py
```

下载器从官方 API 读取固定主模型修订及辅助模型摘要，优先尝试 Hugging Face 镜像，失败后使用官方地址；支持断点续传，完成后验证大小和 Git/LFS 摘要。完整下载包含文字情感模型。所有文件、缓存与临时文件保留在当前项目内，不会上传模型。

若中断，重新运行同一命令即可。已通过校验的文件直接复用，未完成的文件以 `.part` 保存。完成后会生成当前模型的 `docs/model_manifest.json`。

## 手动准备

主模型来源：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)，固定修订 `c39ce5ba981572cb187443877ff559dfb246ce63`。也可从官方 [ModelScope](https://modelscope.cn/models/IndexTeam/IndexTTS-2.5) 下载相同版本。

```text
models/IndexTTS-2.5/
├── config.yaml
├── gpt.pth
├── codec.pth
├── s2mel.pth
├── multilingual_zh_ja_yue_char_del.tiktoken
├── wav2vec2bert_stats.pt
├── feat1.pt
├── feat2.pt
├── LICENSE
├── qwen0.6bemo4-merge/              # 仅文字情感模式必需
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── 该目录其他原始 tokenizer 配置
└── hf_cache/
    ├── w2v-bert-2.0/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    ├── campplus_cn_common.bin
    └── bigvgan/
        ├── config.json
        └── bigvgan_generator.pt
```

辅助模型分别来自 `facebook/w2v-bert-2.0`、`funasr/campplus`、`nvidia/bigvgan_v2_22khz_80band_256x`。2.5 的 `codec.pth` 已由主模型仓库提供，不再需要单独的 MaskGCT 语义 codec 权重。

模型必须是完整文件，不能是 Git LFS 指针。程序只使用此扁平目录结构，不依赖用户级 Hugging Face 的 `blobs/snapshots` 缓存。

## 验收

```powershell
.\run.bat --check --with-emotion
.\even\python.exe -X utf8 -I -B .\test\test_model_integrity.py
```

前一条验证环境、配置版本和必需文件；后一条按清单逐个检查完整 SHA-256。缺少文字情感模型时，其他三种情感模式仍可使用。

## 台语语言包

台语 LoRA 和专用语言行必须一起放在 `models/extra_language/zh-nan-tw/model/`，
其中包含 `adapter/`、`extra_state.pt`、`base_model_manifest.json` 与原始交付清单。
程序会校验模型包和当前基础模型的摘要。普通话整句输入还需要本地翻译模型与词库：

```powershell
.\even\python.exe -X utf8 -B .\tools\download_minnan_translation.py --backend taigi
.\even\python.exe -X utf8 -B .\tools\build_minnan_words.py
```

模型和原始资料首次下载需要联网，准备完成后转换与合成可离线运行。下载脚本固定
上游修订并核对原始文件摘要；GGUF 放在语言包的 `translation/taigi/`，所有缓存仍
保存在本项目内。翻译模型为 CC-BY-NC-SA-4.0，词库来源和限制见 `MINNAN_WORDS_SOURCES.md`。
