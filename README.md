# IndexTTS For Windows - 便携式分发包

> 零样本情感语音合成 | Zero-Shot Emotional Text-to-Speech
>
> 基于 [IndexTTS](https://github.com/index-tts/index-tts) 项目构建的独立桌面应用

---

## 快速开始

1. **将模型文件放入 `checkpoints/` 文件夹**：
   ```
   checkpoints/
   ├── gpt.pth                      (约 3.4 GB)
   ├── s2mel.pth                    (约 1.1 GB)
   ├── config.yaml
   ├── bpe.model
   ├── wav2vec2bert_stats.pt
   ├── feat1.pt
   ├── feat2.pt
   └── qwen_0.6b_emo4_merge/       (Qwen 情感模型)
   ```

2. **双击 `run.bat`** 启动应用。

   首次运行会自动：
   - 检测系统中已有的 Python 环境
   - 安装所需的依赖包
   - 启动桌面应用

## 目录结构

```
IndexTTS_For_Windows/
├── run.bat                      启动脚本（双击运行）
├── desktop_launcher.py          启动器入口
├── indextts_engine.py           推理引擎封装
├── indextts_window.py           PyQt6 桌面窗口
├── indextts_worker.py           后台工作线程
├── Chaos.ico                    应用图标
├── requirements.txt             依赖声明
├── checkpoints/                 模型文件（放入此处）
├── source/                      IndexTTS 核心源码
│   └── indextts/                全部 TTS 子模块
└── outputs/                     生成的音频（自动创建）
```

## 系统要求

- Windows 10/11（64 位）
- Python 3.10+（`run.bat` 自动检测）
- NVIDIA GPU 8GB+ 显存（推荐）
  - CPU 模式可用但速度较慢

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `INDEXTTS_MODEL_DIR` | 模型文件目录 | `checkpoints/` |
| `INDEXTTS_OUTPUT_DIR` | 音频输出目录 | `outputs/` |

## 相关链接

- **IndexTTS 项目主页**：[https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- **IndexTTS2 论文**：[arXiv:2506.21619](https://arxiv.org/abs/2506.21619)
- **模型下载**：[HuggingFace - IndexTeam/IndexTTS](https://huggingface.co/IndexTeam/IndexTTS)

## 注意事项

- 首次启动需下载约 5GB 的 pip 依赖包，请耐心等待
- 确保已连接网络以下载依赖和 HuggingFace 模型缓存
- 如遇到 GPU 显存不足，可用 `set INDEXTTS_MODEL_DIR=` 指定模型位置后重新启动
