# 本地分发包

三个成品生成在项目的上一级目录，构建命令在项目根目录执行：

```powershell
.\even\python.exe -X utf8 -I -B .\tools\build_distribution.py --kind all
```

`../indextts.zip` 为完整包。解压后进入 IndexTTS-For-Windows/，运行 run.bat。
包含当前项目内 even/、IndexTTS 2.5 全部基础权重、台语正式语言包、
完整／精简翻译权重及词库。面向 Windows x64；CUDA 推理仍需要兼容的 GPU 驱动。
安装环境已随包提供，模型运行不需要另行下载。

`../model.zip` 为台语语言包。将压缩包内 models/ 合并到程序根目录，
安装后路径为 models/extra_language/zh-nan-tw/。
包含正式 model/、words/、translation/、language.json 和相应说明。
不包含 IndexTTS 2.5 基础权重，须与程序的基础模型配合使用；不包含试训归档。

`../release.zip` 为 GitHub Release 用轻量包。解压后进入
IndexTTS-For-Windows/，运行 run.bat，首次运行安装依赖到 even/。
压缩包提供程序、第三方许可证及环境安装器；模型和台语语言包需另行准备。
制作该 ZIP 不会自动发布或覆盖远端 GitHub Release。

三个包均排除 SSH 凭据、Git 元数据、个人音色库、个人输出、训练语料、
缓存、日志、旧版 2.0 模型和试训内容。源码仓库同样排除运行环境和模型。
大文件使用 ZIP64；请使用支持 ZIP64 的现代解压工具。

包内清单列出每个文件的 SHA-256。构建先写候选文件，逐文件读取校验
CRC 与 SHA-256 后再替换指定成品；已有成品在新包确认前保留。
详细清单和验收回执保存在项目 build/distribution/，不会加入源码仓库。

基础模型、翻译模型及词库仍分别遵循其原始许可。Taigi-Llama-2 权重
限非商业用途，词库保留各来源材料和许可，含模型分发不会改变这些条件。
