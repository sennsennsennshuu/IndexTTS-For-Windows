# IndexTTS For Windows v1.2.0

本版提供深色蓝灰色语音创作工作台及项目内 Python 运行环境，**不包含任何模型权重**。

## 下载与使用

完整下载以下三个 ZIP，依次解压到**同一个目录**，合并同名的 `IndexTTS-For-Windows` 文件夹：

- `IndexTTS-For-Windows-v1.2.0-program-no-models.zip`：程序、源码、许可证与说明。
- `IndexTTS-For-Windows-v1.2.0-runtime-01.zip`：项目内 Python/CUDA 运行环境第一部分。
- `IndexTTS-For-Windows-v1.2.0-runtime-02.zip`：项目内 Python/CUDA 运行环境第二部分。

三个都是独立的标准 ZIP，可用 Windows 自带解压功能解压，无需分卷合并。解压后项目内应有 `even/python.exe`；无需激活或安装外部 Conda 环境。

按 `docs/MODEL_SETUP.md` 将自行准备的模型放入项目内 `models/IndexTTS-2/`，然后双击 `run.bat`。仅解压程序包、尚未放入模型时，启动检查会明确列出缺少的文件。

首次启动可在项目目录执行：

```powershell
.\run.bat --check
```

`SHA256SUMS.txt` 提供三个 ZIP 的 SHA-256，可用于检查下载是否完整。GitHub 自动生成的 Source code 附件只含源码，**不含 `even/` 环境**；需要直接解压使用时，请下载上述三个程序与运行环境 ZIP。

## 本次更新

- 从桌面助手模块分离为独立 Windows 项目，根目录仅保留 `run.bat`、`start.py`、`requirements.txt` 三个普通文件。
- 全新炭黑与蓝灰色平面界面，采用文本与结果区、声音设置区的双栏布局。
- 保留跟随音色、情感参考音频、八维情感向量、文字描述四种情感模式。
- 支持拖入参考音频，显示音频信息，折叠高级参数。
- 内置真实波形、播放、暂停和定位，可关闭完成后自动播放。
- 连续任务复用模型，关闭时等待当前合成结束，避免工作线程被提前销毁。
- 模型、缓存、输出和日志使用项目内路径；文字情感分类使用 CPU，为声学模型保留显存。

## 发行包范围

包含应用、上游推理源码及许可证、CPython 3.11.15、PyTorch/torchaudio 2.8.0+cu129、PyQt6 和其余运行依赖。

不包含主模型、情感模型、辅助推理权重、原项目中的 JDC 权重、用户音频、测试录音、输出音频、缓存或运行日志。也不包含编译静态库和开发头文件；普通 CUDA 推理继续可用。

## 系统与模型要求

Windows 10/11 x64；CUDA 构建需要兼容的 NVIDIA 驱动。无可用 GPU 时模型可走 CPU，速度较慢。模型准备完成后可按原独立项目方式离线运行。

模型的原有许可证保留在模型来源及项目的 `vendor/` 目录中；程序包不代替模型授权。
