# 台语 IndexTTS 2.5 云端微调

截至 2026-11-05，本轮 15,000 步正式训练、持续监控、最终成对合成及本机私有交付
均已完成。最新结果与加载方式见文末“最终训练与交付验收”；下文各启动和中途里程碑
保留为历史记录，台语母语者听感验收仍未完成。

本任务使用用户提供的真实 SSH GPU 服务器。所有新环境、基础权重、音频缓存、
日志和检查点仅写入 `/root/autodl-tmp/IndexTTS25_Minnan`；不使用系统盘或持久存储盘
存放训练产物，不改动旧 `IndexTTS/data` 的音频和二进制数据。

## 已确认的训练目标

- 基础版本仅为 IndexTTS 2.5，固定官方模型及辅助模型修订，逐文件校验 SHA-256。
- 输入支持带声调台罗、已有明确对应关系的台语汉字，不把普通话文本翻译成台语。
- GPT 使用 LoRA；codec、S2Mel、BigVGAN、W2V2Bert、CAMPPlus 与情感编码器保持冻结。
- 新增 minnan 语言行的独立增量，从 zh 行初始化。使用已有 `<|zh|> ` 文本前缀及
  minnan 语言 ID 100，训练与推理保持一致，不重排词表编号。
- 汉字和台罗是同一录音的两个输入视图；划分数据时必须共同进入同一个集合。

## 数据来源与重要纠正

`source/kautian.ods` 是词条、例句与辞典录音对应关系的直接来源。
`dictionaries/scripts/build_leku_labs.py` 已实现正确的例句映射：使用“例句”表
完整“音檔檔名”精确匹配 WAV，并额外核对“詞目id”和“例句順序”。不能只按词目 ID
把例句音频配到词条文字，也不能因录音文件名有多个编号而直接排除全部 leku。
词条使用“詞目”表的“羅馬字音檔檔名”匹配 sutiau。

Common Voice 本地 `nan-tw/README.md` 明确说明它是汉字—语音数据集，括号中的
台罗或白话字仅为参考，无法代表每位录音者的实际发音，并建议去括号后使用汉字。
因此首版使用已验证录音的汉字部分，不把括号参考当作准确台罗监督。

台罗监督保留 ODS 原始声调与 Unicode 字符，不使用无调 MFA LAB 替换。
现有 DiffSinger 声库的音素裁剪范围不能用于判断 IndexTTS 是否能够训练某个合法音节。
有斜线、可选读音或来源冲突的项目保留审查理由，不自动猜读音。
来源许可随清单保留；用户私人训练授权不等于新授予任何语料再分发许可。

## 云端复用与数据隔离

优先校验 tmp 盘已有同源数据。辞典 WAV 必须与本地源 SHA-256 完全一致；
Common Voice 已转换 WAV 必须同时满足转换清单中的原始 MP3 SHA-256、目标 WAV
SHA-256 均匹配。只有全部样本校验或补传成功后才提交云端数据清单。

Common Voice 保留真实 client_id；辞典没有可信说话人标识时使用 null，不假造同一说话人。
验证集按来源分组、文本与音频去重，禁止同一录音的两个文本视图跨集合。
有可用同说话人参考时采用另一句作为条件，情感参考与目标代码分开；缺少可信参考
时使用明确记录的保守策略，不把目标完整语义序列作为条件输入。

## 运行方式

以下命令从本地 IndexTTS 项目根目录执行。SSH 密码只从未纳入 Git 的 `ssh.txt`
读取，不放入训练包、日志、命令行或公共仓库。

```powershell
.\even\python.exe -X utf8 -I -B .\tools\cloud\remote.py run --command 'cd /root/autodl-tmp/IndexTTS25_Minnan && tail -40 logs/setup.log'
.\even\python.exe -X utf8 -I -B .\tools\cloud\remote.py run --command 'cd /root/autodl-tmp/IndexTTS25_Minnan && nvidia-smi'
```

训练应依次通过：环境及模型完整性检查、数据清单隔离检查、真实 codec 重建、
小样本可学习性检查、正式训练和未见文本合成。每个阶段都保存真实回执，前置阶段
失败时停止后续训练，不能把单纯的 loss 下降或非静音音频称为台语发音验收通过。

## 当前验收边界

工程检查包括有限 loss／梯度、断点恢复、特征和权重摘要、推理结果及音频有效性。
台语声调、变调、鼻化和入声是否正确仍需实际试听；在完成台语母语者评审前，
不应把模型称为已经达到正式发布或商业配音质量。

## 2026-11-05 实际运行回执

- 真实 GPU：RTX 4090 24 GB；数据盘扩容至 150 GB，所有本次产物位于 tmp 项目中。
- 最终数据：60,447 个音频，约 44.096 小时；95,482 个文本视图；审查记录 9,381 项。
- 训练／验证／测试分别为 47,032／7,588／5,827 个音频，各集合音频、规范文本、
  word_id 和已知 speaker_id 均无交集。
- 云端复用 56,000 个经哈希匹配的同源音频，只补传 4,447 个缺失文件（源字节约 76.7 MB）。
- 固定基础模型 15 个文件全部校验通过；云端环境通过 pip check、训练与推理导入。
- 200 步小样本试训使用 8 个音频；前／后 20 步平均训练 CE 为 7.16794／0.12181，
  下降 98.30%；完成 4 条适配器合成和从 200 步到 201 步的真实续训。
- 已生成 2 条 codec 重建及 4 条基础模型合成。上述均为工程检查，不代表人工听感验收。
- 试训包现归档在 `models/extra_language/zh-nan-tw/validation/pilot/`，交付 SHA 见其
  `delivery_receipt.json`；正式模型随后另行完成，最终验收结果见文末。
- 正式配置为 15,000 步、`batch_size=4`、`grad_accum=2`，有效 batch 8。
  两个方案均实际消费 160 条音频的 20 步基准中，单条累积方案为 1.9448 秒/步，
  变长批量方案为 0.51073 秒/步，实测提速 3.8079 倍；这不是减少训练样本数的提速。
  基准在全量特征提取并行运行时测得，完整语料耗时随句长而变化。
- 28 项不同回归分批通过，其中 12 项训练检查覆盖真实 PEFT、变长 batch 与
  独立前向的损失／梯度一致性、实际采样游标和历史 best 保留。

`logs/cloud_training_live.json` 是后台同步保存的云端状态；工具默认每五分钟刷新，
本轮最终运行实例使用 60 秒间隔。训练完成后的终态文件是历史回执，不能当作仍在训练。
`logs/cloud_delivery_watch.log`、`logs/cloud_delivery_watch.stderr.log` 和
`logs/cloud_delivery_watch_error.json` 分别记录状态、标准错误及重连故障。
本机辅助依赖单独锁定在 `tools/cloud/requirements_local.txt`；不并入桌面推理依赖。

本机后台同步进程运行 `tools/cloud/watch_delivery.py`。正式流水线结束后，它会验证
云端公布的 SHA-256，再安全解包至 `models/extra_language/zh-nan-tw/model/`，
保存 `delivery_receipt.json` 和 `docs/cloud_minnan_delivery.json`，随后更新相应待办。
同步进程停机后可从项目根目录重新启动；云端作业独立运行。

批量方案通过基准后，已执行受控恢复：仅终止本项目流水线及其提取子进程，
保留逐条原子保存的特征并复用。恢复命令和进程身份检查在
`tools/cloud/restart_extraction.py`；禁止停止其他项目的进程。

```powershell
.\even\python.exe -X utf8 -u -I -B .\tools\cloud\watch_delivery.py --interval 300
```

真实云端运行还修复了以下集成问题：Linux 默认构造会强行加载普通话 FST，现可
显式注入台语纯 NFC 前端；PEFT 需要实际 mel embedding 的冻结 wte 别名；
参考录音不足时采用平静情感均值，保留真实 speaker_id 并禁止跨 split 取参考；
验证子集按来源和文字类型分层；新检查点变差时仍保留历史 best；基础模型先测
初始验证 CE，避免把比基础模型更差的训练结果当作改进。

## TensorBoard 实时查看

本机入口：`http://127.0.0.1:16007/#scalars`。服务由项目内的 SSH 隧道转发，
本机与云端项目均需保持运行；重新启动电脑后，需要重新启动隧道。

- `pilot`：8 条训练音频的试训记录，训练 loss 横轴为真实优化步数；
  `validation/loss` 来自实际验证记录和仍保留的检查点，不插补缺失步数。
- `features`：特征提取的完成数、总数、百分比；横轴为已处理的音频数。
- `formal`：本轮正式训练记录，包含训练／验证 loss、梯度与学习率，已完整到第 15,000 步。

推荐先看 `train/loss_per_token`、`validation/loss` 和 `progress/percent`。
历史 JSONL 没有绝对训练时间，因此历史点的 Wall time 是导入时间；
比较训练曲线请使用 Step 横轴。试训验证曲线与正式训练曲线分开，不能用小样本过拟合
证明整个数据集的泛化质量。桥接器与云端服务默认每 10 秒读取一次，浏览器需启用自动刷新。

云端 events、进程状态和临时文件分别放在项目相对路径 `training/tensorboard/`、
`logs/tensorboard_*.json`／`logs/tensorboard_*.log`、`cache/tmp/`。
独立服务仅监听云端回环地址 6008，不改动平台已有的 6007 服务；本机入口也只监听回环地址。
恢复云端查看服务时，在 tmp 项目根目录运行：

```bash
even/bin/python -B tools/cloud/serve_tensorboard.py
```

恢复本机隧道时，在 Windows 项目根目录运行以下命令，并保留该进程：

```powershell
.\even\python.exe -X utf8 -u -I -B .\tools\cloud\tensorboard_tunnel.py
```

本机隧道会自动重连；连接状态见 `logs/tensorboard_tunnel.json`，后台日志见
`logs/tensorboard_tunnel.log` 与 `logs/tensorboard_tunnel.stderr.log`。

## 持续训练与资源监控

独立监控器 `tools/cloud/monitor.py` 每 30 秒采样一次。它直接核对 `/proc` 中的
流水线及子进程、实际工作目录，读取 GPU 遥测与 tmp 数据盘余量，并增量处理正式
训练日志。监控不加载模型，也不创建 CUDA 上下文；所有新产物仍留在数据盘项目内。

TensorBoard 新增 `system` run，横轴 Step 是监控采样序号；查看实际时间跨度可使用
Wall time。主要指标包括 GPU 利用率／总显存占用／温度／功耗、磁盘可用 GiB／
使用百分比、实际进程状态、进度停滞秒数、处理吞吐和告警数量。
`formal/rolling_loss_100` 是最近至多 100 个正式训练步的 token 加权 loss 均值；
`formal/baseline_validation` 和 `formal/best_validation` 分别为初始验证与历史最佳验证。
正式训练尚未开始时，这些正式训练指标不会凭空生成。

正式训练原 run 另记录 `train/optimization_loss`（实际反向传播使用的每句平均 CE
在 microbatch 间的均值）和 `train/samples_per_step`，用于核对变长批量下的优化目标及
实际样本数；`train/loss_per_token` 继续按所有有效 token 加权，二者在句长不同时可不同。

监控规则包括：非有限指标、实际流水线进程缺失或身份不符、超过十分钟无进度、
tmp 空间低于 10 GiB（低于 3 GiB 标为严重）、GPU 温度达到 85℃、最近三次正式验证
连续升高或均高于初始验证。阈值只用于提示检查，不触发自动重启、停训或修改学习率。
GPU 读取失败单独记录，不把观测失败当作训练失败；验证趋势仅取正式训练日志，
不会把 pilot 的过拟合现象混入正式告警。

云端详细快照为 `logs/monitor_latest.json`，历史为 `logs/monitor_history.jsonl`，
异常为 `logs/monitor_error.json`。本机五分钟同步将监控快照一并写入
`logs/cloud_training_live.json` 的 `monitoring` 字段；可比较 `checked_unix` 判断新鲜程度。
手动现场核验回执保存在 `logs/cloud_monitor_latest.json`。
重新执行 `tools/cloud/serve_tensorboard.py` 会核对并复用本项目监控服务，避免重复启动。

## 未见文本的双输入验收

最终试听选样原先在每个音频第一次出现时就去重，实际四条都落在汉字视图，
遗漏了已有配对台罗。`tools/cloud/heldout_cases.py` 现从真实 test 的 leku 例句中
确定性选择四个目标音频，每个保留汉字与原始带调台罗，共八条输入；同组两种输入
使用相同的不同词目参考音频。录音身份和文本对应均由 manifest 提供，不生成猜测转写。
没有已知 speaker_id 时会明确标记，不能把参考音频假称同一说话人。

已生成 `training/data/minnan/heldout_paired_cases.json`，本机与云端目标音频摘要、
参考摘要及文本逐项相同，回执为 `logs/heldout_pairing_audit.json`。四项检查覆盖
双视图及声调保留、枚举顺序独立、split／缺失视图门禁和冲突转写拒绝。
合成器会在输出目录保存实际 `cases.json`，并在 report 中记录文字类型及目标摘要。

本次已经运行的父流水线在修改前加载了旧选样代码，因此仍需在正式训练完成后
针对最佳检查点补跑此成对清单，并把八条实际合成与报告补入最终交付，再核验本地包。
准备好的输入清单和通过的选样检查不代表合成或台语母语者听感已经通过。

## 正式训练启动与首批检查点回执

全量特征已完成 60,447 个，其中本次新提取 38,534 个、复用 21,913 个签名一致的缓存。
流水线自动启动正式训练，现场进程 PID 为 8989；配置为 15,000 步、batch 4、累积 2，
每步日志均记录八条样本，基础模型指纹仍为
`7b49c636676c34fba080938780906eced4a230d98b44e7fd117e7a489a7b0611`。

固定 64 条验证输入的初始 CE 为 7.3747058；第 100／200／300 步分别为
5.9994791／5.6507275／5.4968925。该结果证明首批固定验证有改善，不代表整轮训练
已经完成，也不替代未见文本合成和台语母语者听感验收。

第 100 步检查点的真实 sampler cursor 为 800；适配器摘要与第零步不同，台语语言
delta 的范数为 0.0134480，未把重复同一条数据或仅推进计数器当作训练。
该检查点适配器 SHA-256 为
`6051fd5ffe0d067df75978167ce130614e56962be261f86f5cdc6e23c801da18`。
回执 `logs/formal_first_checkpoint_audit.json` 已从云端取回并验证传输摘要。
历史启动快照为 `logs/cloud_formal_start.json`，后续现场快照为
`logs/cloud_formal_latest.json`；判断实时状态仍需结合云端实际进程及最新采样时间。

## 第 1,500 步验证集试听

为观察实际发声，`tools/cloud/preview_training.py` 从 valid 选取一组成对汉字／台罗，
先将当前最佳适配器及语言状态复制并逐文件核对摘要，固定在
`training/evaluation_checkpoints/step_001500/`，防止正常 checkpoint 清理干扰试听。
基座与适配器推理串行执行，启动前要求至少 12 GiB 空闲显存；本次与训练并行时
观察到总显存约 10.2 GiB，主训练进程继续推进。预览不用最终 test，六项选样检查
在本机与云端均通过，含 valid/test 隔离及禁止 train 冒充泛化输入。

已真实生成两条基座及两条微调音频，并通过非静音、有限值与采样率检查。
本机结果在 `outputs/validation_previews/step_001500/`，包括输入清单、两组报告、
固定权重摘要与下载回执。传输包 SHA-256 为
`e704a59a2ba10f9aa60f7f2cfd8cc04434cc0f49332b68f07e8847741afa3b9f`，
下载后校验一致并已清理本机传输归档；音频逐文件摘要保存在 `download_receipt.json`。

例句为“伊英語講甲足端的的，所以予電視台倩去報英語新聞。”；配对台罗原样保留声调。
另从用户授权 source 复制 `target_original.wav` 作为验证集真实录音，未做格式转换，
SHA-256 为 `cf8c472092d9bbe59750af608410561fee169f61324f1a6a8f5e171c3a6c8fff`，
与原始 manifest 的目标音频摘要完全相同。报告始终保留 `native_listening_passed=false`；
工程音频检查不代表台语声调、变调或可懂度已经通过人工评审。

3,500 步后的现场检查记录在 `logs/cloud_formal_milestone_3500.json`：实际采样时
训练已到第 3,730 步，最佳为第 3,700 步、验证 CE 4.9064957，较初始下降约 33.47%，
没有非有限指标或资源告警。此文件是历史里程碑，不应当作后续实时进度。
本机试听目录含四条合成和一条原始录音，数量已在下载回执中分别列明；
本机与云端临时传输归档均在音频核验后清理，固定试听检查点与音频保留以便复查。

## 第一轮采样覆盖核验

`tools/cloud/audit_sampling.py` 在 CPU 读取真实检查点和相同指纹的 train/valid 清单，
用同版确定性采样器重放至保存的 cursor，并逐步核对日志中的真实末样本音频摘要、
文字类型和样本数。覆盖结论依据检查点游标、采样器重放及逐步日志交叉核对，
不是仅根据界面的 epoch 数推算，也不等同于逐条音频的人工听感验收。

已对第 6,600 步检查点实际运行：cursor=52,800、有效 batch=8，train 含 47,032 个
唯一音频与 74,741 个文本视图。第一轮覆盖全部 47,032 个音频，其中汉字 33,196 次、
台罗 13,836 次；第二轮已处理 5,768 个音频，其中 3,402 个成对视图已切换文字类型。
全部 6,600 条逐步末样本记录与重放一致，没有重复步数日志，数据指纹与检查点一致。

回执 `logs/sampling_audit_step_006600.json` 已下载并验证 SHA-256：
`0f017fec7cf0549b3f315862cfd39a017b8a9da93f0358a938ae7c73be281d29`。
更早的第 6,400 步现场核验保存在 `logs/first_epoch_sampling_audit.json`，轮次标签
已统一为从 1 开始，并保留原始重放计数供核对。最终训练完成后应再次运行采样审计，
确认两轮完整覆盖和最终实际采样总量，不能把本次部分第二轮检查当作全程验收。

第 3,700→3,900 步曾因三次小幅上升触发验证趋势提示；第 4,000、4,100 步随后
均刷新最佳，提示消退。复核记录为 `logs/cloud_validation_review_4000.json`，
该次没有修改训练参数。最近现场检查为第 6,917 步，最佳第 6,600 步 CE 4.7899024；
上述均为历史记录，实时状态以最新进程、采样时间及 TensorBoard 为准。

## 两轮与全部文字视图覆盖核验

采样审计新增唯一文字视图计数，并在完成至少两轮音频覆盖时强制检查全部视图参与。
实际第 12,000 步核验通过：cursor=96,000，两轮各覆盖 47,032 个音频，第三轮已处理
1,936 个。74,741 个唯一文字视图全部出现，`unseen_text_views=0`、
`all_text_views_seen=true`；第二轮 27,709 个成对音频均切换文字类型。
12,000 条逐步末样本日志与采样器重放一致，没有重复步数，数据指纹与检查点一致。

回执 `logs/sampling_audit_step_012000.json` 已下载并验证 SHA-256：
`19aad6a012574cfbfd1f16707631a53cce3c0d7cb584d14bd953e32b5aa94512`。
此项验证实际数据覆盖，不代替台语发音验收；最终第 15,000 步仍须再次核对实际采样总量。
最近现场采样时训练为第 12,620 步，最佳第 12,400 步验证 CE 4.6942112，告警为空。

## 最终成对验收与自动取回门禁

训练完成后运行 `tools/cloud/finalize_delivery.py`：它要求正式训练和父流水线均已完成，
再核验第 15,000 步实际采样、从固定 test 清单生成八条成对输入，使用最佳检查点
真实合成并复核报告与波形，复制未转换的目标／参考录音及来源摘要。
所有阶段状态写入 `logs/final_delivery_status.json`，失败保留现场供排查。
`--check-only` 仅报告是否就绪，不会在训练期间启动最终测试合成。

最终归档含 LoRA、独立台语语言行、固定基础模型清单、训练配置与指标、完整采样回执、
成对测试音频和原始录音、TensorBoard 历史及工具源码摘要。归档成员逐项重新计算
SHA-256，通过后才原子替换旧流水线包并更新发布摘要；没有把官方基础模型或 even/ 打包。
本机接收端还会核验清单与实际文件集合、各成员大小和摘要、采样门禁、八条输入及真实
WAV 头。缓存命中也重新核验文件，本机追加的 delivery_receipt.json 单独允许保留。

本轮最终同步实例采用 60 秒间隔。父流水线仅完成旧四条汉字测试时，同步保持
`waiting_for_paired_delivery`，未提前标记交付完成；八条成对验收和最终审计通过后，
已自动下载并写入模型目录。五项门禁检查在 Windows 本机与 Linux 云端均通过，
它们覆盖交付规则与损坏拒绝；实际模型发声和台语听感仍分别以真实合成及人工评审为准。

## 最终训练与交付验收（2026-11-05）

本轮正式训练已正常结束，没有追加未约定的训练步数。训练、父流水线和补充成对
验收均记录完成终态；训练进程已退出。TensorBoard 与 SSH 转发保留供用户查看历史，
没有关闭云实例，也没有上传基础模型或微调权重到 GitHub。

- 全量特征完成 60,447/60,447；正式训练为 15,000 个优化步，batch 4 × 累积 2，
  共实际消费 120,000 次样本，训练循环耗时 8,211.893 秒，约 2 小时 17 分钟。
- 固定 64 条验证输入产生 151 个验证点：初始 CE 7.374705798430353，最终／最佳
  CE 4.689264848111948，下降 36.4142%；最佳检查点为第 15,000 步。
- 15,000 条训练记录和全部验证记录的指标均为有限值。352 次资源采样没有严重告警；
  期间的 `validation_rising` 为局部验证回升提示，已复核且后续继续改善，最终告警为空。
- 最终采样审计证明两轮各覆盖全部 47,032 个训练音频，第三轮再覆盖 25,936 个；
  74,741 个唯一文字视图全部参与、未见视图为零，全部 27,709 组成对视图完成轮换。
  15,000 条逐步末样本日志与同版采样器重放一致，数据指纹匹配，没有重复步数日志。
- 使用最终最佳检查点真实生成四组汉字／带声调台罗，共八条 test 音频。案例与训练前
  固定清单一致；所有输出实际读取后满足 22,050 Hz、有限值和非静音检查。另保存四条
  未转换的目标／参考原始录音，逐条 SHA-256 与来源一致。
- 本机再次核验交付清单中的 37 个文件；云端归档内清单、云端独立清单和本机清单的
  SHA-256 一致。线上 TensorBoard 的验证曲线有 151 个点，最后一步为 15,000。

本机当前模型目录：`models/extra_language/zh-nan-tw/model/`；2026-11-05 已从原摘要
子目录迁移，37 个交付文件保持不变，迁移回执为 `logs/minnan_language_pack_migration.json`。
其中 `adapter/adapter_model.safetensors` 为 GPT LoRA，`extra_state.pt` 为必须同时
恢复的台语语言行。基础权重须与包内 `base_model_manifest.json` 完全匹配；
不能只加载 adapter 而忽略语言状态。加载实现为 `tools/cloud/evaluate.py` 的
`attach_adapter()`。本轮未实施或验证 Windows 桌面 UI 对此适配器的自动加载。

云端在 tmp 项目根目录复现最终案例的命令如下；它会新生成音频，查看既有结果无需重跑：

```bash
even/bin/python -B tools/cloud/evaluate.py --cases training/data/minnan/heldout_paired_cases.json --adapter training/runs/minnan-lora/checkpoints/step_015000 --output outputs/reproduced_minnan
```

八条结果位于模型目录的 `heldout_audio/`，按 `00_heldout_0_hanzi.wav`、
`01_heldout_0_tailo.wav` 至 `07_heldout_3_tailo.wav` 排列。`cases.json` 保存输入文字
和参考摘要，`original_audio.json` 定位未转换的原始录音；合成音色取决于参考录音，
原始目标录音用于核对内容发音。完整 TensorBoard 历史保存在模型目录的 `tensorboard/`。

云端最终包：`training/runs/minnan-delivery.tar.gz`，64,741,786 字节。
SHA-256：`5aa89cd998419accf8c470e981d926a126ae63622cda7c80bf0bdb2935b1250b`。
LoRA 权重 SHA-256：`256e782561a8467d471104961aab77d9e9421b3fa6f3b181399cf704c0d5ddaa`。

交付回执为 `docs/cloud_minnan_delivery.json`，完整本机验收为
`logs/cloud_final_acceptance.json`，本轮收尾复核为 `logs/cloud_goal_completion_audit.json`，
最终采样回执为 `logs/sampling_audit_step_015000.json`。模型目录受逐文件清单保护，
新增说明或评审记录应保存到项目 `docs/` 或 `logs/`，避免破坏包内容一致性。

工程训练、监控、真实合成与私有交付已完成；`native_listening_passed=false` 仍保留。
台语声调、变调、鼻化、入声及未见文本可懂度需要母语者试听，当前结果不代表已经
通过人工发音验收，也不能仅根据 CE 改善宣称达到商业配音质量。
