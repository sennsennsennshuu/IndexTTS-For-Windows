# 台湾闽南语词库来源、许可与构建说明

本文件记录 `models/extra_language/zh-nan-tw/words/` 的可审计来源边界。词库采用混合许可，**不能据此宣称全部内容都可用于商业发布或再分发**。用户当前私人使用范围不改变各上游许可的约束。

## 产物

- `words/entries.jsonl`：主词库。每行固定九字段：
  - `id`：稳定 ID，由来源和来源内编号计算。
  - `hanzi`：台语词目。汉罗混排来源保留原始汉字、拉丁拼写和标点；只有完全不含汉字时为空。
  - `tailo`：带调台罗。同一来源行的多个候选读音拆成多条，不静默选一个。
  - `mandarin`：明确普通话对译候选；不保存整段释义或例句说明。
  - `source`：内部来源标识。
  - `source_id`：上游行号、词条号或页内编号。
  - `license`：该行对应的许可，按上游逐项标注。
  - `kind`：`word` 或 `sentence`。
  - `verified`：来源字段明确对齐且可用于唯一读音判断。`false` 不代表内容无效；例如 THUOCL 仅作普通话候选，故固定为 `false`。
- `words/manifest.json`：生成时间、数量、许可分布、每个来源限制，以及原始文件和输出文件的 SHA-256。
- `words/lexicon.sqlite3`：便于查询的同一数据副本。
- `words/sources/`：实际下载的 CSV、JSON、HTML、Markdown 原始响应。

当前构建统计：

- 总条目：319,076
- 带台罗：162,098
- 带普通话对译：317,452
- `verified=true`：162,098
- `kind=word`：315,606
- `kind=sentence`：3,470
- `entries.jsonl` SHA-256：`faa899eb598fd23ea545e160436c8ea1aa826b860e9095470095c8c912dbbc1e`

## 用户指定资料源

### 1. THUOCL

- URL：https://github.com/thunlp/THUOCL
- 实际状态：成功下载 README、LICENSE 和 11 个分类词表。
- 许可：MIT。
- 用途：**只作普通话词语候选覆盖**。
- 词库边界：`hanzi=""`、`tailo=""`、`verified=false`。不得将 THUOCL 词形当作台语汉字、台罗或普通话对译证据。
- 当前条目：156,978。

### 2. 国家教育研究院“实用台语词典”

- URL：https://tggl.naer.edu.tw/practical_taiwanese_dictionaries
- 实际状态：网页和“食”查询结果均可匿名访问。查询“食”实测得到完全符合 2 笔、部分符合 96 笔。
- 许可/限制：页面说明该词典由原作者授权网站公开查询，版权仍归原作者所有。它不是开放批量下载许可。
- 词库边界：只保存访问和查询审计证据，**未批量复制为 entries**。
- 原始证据：`words/sources/audit/tggl_practical_taiwanese_dictionaries.html`、`words/sources/audit/tggl_query_food.html`。

### 3. Wikiversity：Taiwan Tongues/台湾台语开源语料库

- 页面 URL：https://zh.wikiversity.org/zh-my/Taiwan_Tongues/臺灣臺語開源語料庫
- 实际状态：raw wikitext 可下载；页面链接的 iCorpus-100 数据接口可下载并解析。
- 页面自身：按 Wikimedia 页面条款处理页面文本。
- iCorpus-100：
  - 数据 URL：https://huggingface.co/datasets/Bohanlu/iCorpus-100
  - 许可：CC BY-NC 4.0。
  - 实际字段：`ZH` 普通话、`HAN` 台语汉字、`TL` 台罗、`POJ` 白话字、`EN` 英文。
  - 当前入库：100 条整句对；只使用 `ZH/HAN/TL`，英文不进入词库。
  - 限制：数据集卡说明汉字经过人工校正；CC BY-NC 许可禁止据此宣称商业可用。
- 页面还链接 Taiwanese-Minnan-Sutiau、TAIDE-14-tasks-Hokkien 等资源。前者主要包含约 8 GB 音频且没有本案所需普通话对译；后者是任务提示集，不是直接词对或句对。本次均未下载成词库。
- 原始证据：`words/sources/taiwan_tongues/`。

### 4. 公视台语台“台语新词辞库”

- URL：https://taigi.acv-vision.com/taigi-words
- 实际状态：成功访问并抓取 74 页；页面提供词目、音读、对应华语，当前解析出 991 条读音记录。
- 许可：页面未发现明确开放再利用许可，manifest 按“公开网页、无明确再利用许可、私人使用，再分发前需授权”标注。
- 字段边界：只收录页面中“词目／音读／对应华语”三者齐全的记录；多个音读拆成多条。
- 原始证据：`words/sources/pts_taigi_words/page-001.html` 至 `page-074.html`。

### 5. 清大“台湾本土语言多样性资源库”

- URL：https://taiwan-lingu.ist/corpus/
- 实际状态：入口可访问；匿名检索接口能用“食”返回教育部辞典断词版 94 行结构化结果。
- 许可/限制：站点整合教育部辞典、民间文学、歌仔戏、田野调查等来源；不同来源权利状态不一致，下载全部结果依赖账号、申请或授权流程。
- 词库边界：未批量镜像该站点内容。已保存匿名检索样本作为接口存在与字段结构证据；实际教育部辞典词条使用 ChhoeTaigi 中注明 CC BY-ND 3.0 TW 的原始导出并逐行标注。
- 原始证据：`words/sources/audit/taiwan_lingu_corpus.html`、`words/sources/audit/taiwan_lingu_search_food.json`。

## 补充主数据源：ChhoeTaigiDatabase

- URL：https://github.com/ChhoeTaigi/ChhoeTaigiDatabase
- 上游 README：https://raw.githubusercontent.com/ChhoeTaigi/ChhoeTaigiDatabase/master/README.md
- 字段说明：
  - `KipUnicode` / `KipUnicodeOthers`：教育部罗马拼音，即带调台罗。
  - `HanLoTaibunKip` / `HanLoTaibunPoj`：汉罗台文；仅提取汉字串到 `hanzi`。
  - `HoaBun`：对应华文，不当作长释义使用。
- 当前采用并逐行标注的四个数据集：

| 数据集 | 当前条目 | 许可 | 实务边界 |
|---|---:|---|---|
| `ChhoeTaigi_iTaigiHoataiTuichiautian.csv` | 20,801 | CC0-1.0 | 多读音按来源候选拆分。 |
| `ChhoeTaigi_TaihoaSoanntengTuichiautian.csv` | 100,967 | CC BY-SA 4.0 | 只使用明确对译候选；无法提取台罗的行跳过。 |
| `ChhoeTaigi_KauiokpooTaigiSutian.csv` | 33,350 | CC BY-ND 3.0 TW | 只从结构字段提取；未把长 `Kaisoeh` 释义塞入 `mandarin`。 |
| `ChhoeTaigi_TaioanPehoeKichhooGiku.csv` | 5,889 | CC BY-SA 4.0 | 该文件没有台语汉字字段，因此 `hanzi` 为空；`HoaBun` 始终作为单个完整字段保留，不按逗号或句号拆成多个对译。 |

未采用 README 中明确为 CC BY-NC-SA、CC BY-NC 或需要进一步审查的数据集；原因是本案只需要可审计的优先子集，避免无意扩大许可风险。

## 字段清洗规则

1. `HoaBun` 只在明确语义为“对应华文/对应华语”时进入 `mandarin`。
2. 含句号、`例：`、括注或释义标记的长内容不直接进入短词 `mandarin`。
3. 用 `、`、`；`、逗号或斜杠分隔的多个对译候选保留为列表，不静默取第一项。
4. 同一条台的 `KipUnicode` 和 `KipUnicodeOthers` 分别入库；主程序只有在 `verified=true` 且 `tailo` 集合唯一时才采用精确读音。
5. 汉罗字段里的拉丁文字和标点会保留在 `hanzi` 中，避免词形与完整台罗错位；来源完全没有汉字时 `hanzi=""`，但仍可用于“普通话→台罗”的歧义感知查找。
6. `verified` 表示“来源字段对齐且可用于唯一读音判断”，不表示所有语境下语义和读音都已由语言专家复核。

## 构造与校验

在项目根目录使用项目 Python：

```powershell
.\even\python.exe tools\build_minnan_words.py --offline
.\even\python.exe -m unittest test.test_minnan_words_build -v
```

需要联网刷新原始来源时使用：

```powershell
.\even\python.exe tools\build_minnan_words.py --refresh
```

下载和中间文件均位于项目内；构建器不写系统临时目录。重复构建使用稳定 ID 和排序，输出可直接逐文件校验 SHA。

## 许可汇总

当前条目许可分布：

- CC0-1.0：20,801
- CC BY-SA 4.0：106,856
- CC BY-ND 3.0 TW：33,350
- CC BY-NC 4.0：100
- MIT：156,978
- 公视页面：991（无明确开放再利用许可，私人使用；再分发需授权）

因此该混合词库不能整体声明为 CC0、CC BY 或可商业发布。若后续需要公开发布或商业交付，应分别处理 CC BY-ND、CC BY-NC 和公视页面记录，并保留各来源署名与许可声明。