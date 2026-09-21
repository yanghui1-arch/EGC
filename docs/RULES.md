# 规则构建与证据标注

`examples/synthetic_rules.json` 是纯虚构软件夹具，没有真实法律意义，不能改个罪名就用于正式实验。

2026-09-21新增`configs/rules_e4_draft.json`：有官方来源的部分机制草案，24条件/11节点，均为draft，无人工审核署名。来源版本、E3标注缺口和下一步见[E4规则先导](E4_RULE_PILOT.md)。它不是完整量刑规则包；尚不安排真实API标注，以下通用命令不代表当前执行步骤。

真实规则包应记录法律来源、版本、适用时期和适用地域；条件中明确主体与时间约束。原论文链只作待审查材料，不能假定适用于所有案件。`source` / `version` 是必填元数据，当前程序只校验存在，不会自动验证法律效力。

## 从上游导入草稿

```powershell
python -m egc import-chains --input data/raw/upstream/legal_chain/fraud/chain.txt --output data/rules/fraud_draft.json --charge 诈骗罪 --source LegalChainReasoner --version c66b3d14d2b6e1c9bb6552b774494c64a08e79b3
```

罪名必须与规范化数据的 `charge` 字符串一致。导入器保留复合条件文本，不自动把自然语言AND/OR解释成正确法理。人工需拆解条件、审核例外与修正规则，补上依赖关系、法律来源/版本，再记录实际 `reviewer` 和 `review_status: approved`。不能让LLM自行署名为审核人。草稿可以用 `--allow-draft` 做探索，但manifest明确标记，不能算最终规则实验。

## Schema

- `conditions`: `id`, `text`。每项表述一个可判断的事实/适用条件。
- `rules`: `id`, `charges`, `when`, `requires`, `blocked_by`, `effect`, `source`, `version`, `review_status`, `reviewer`。
- `when` 可以是条件ID、`{"all": [...]}`、`{"any": [...]}`、`{"not": ...}`，支持嵌套。
- `requires` 是前置规则ID列表；`blocked_by` 是阻却规则ID列表。当前仅支持无环关系。
- `charges: ["*"]` 表示跨罪名候选规则，但仍须在条件中判断是否适用。

一条阻却规则 unknown 时，被它阻却的规则也不能无条件判定 supported。缺失事实不会自动变成 refuted。`effect` 是后果描述，没有自动判刑运算。

## DeepSeek辅助标注

先在本机终端临时设置环境变量。PowerShell可交互隐藏输入：

```powershell
$egcSecret = Read-Host 'DeepSeek API Key' -AsSecureString
$egcCredential = New-Object System.Net.NetworkCredential('', $egcSecret)
$env:DEEPSEEK_API_KEY = $egcCredential.Password
Remove-Variable egcCredential, egcSecret
$env:DEEPSEEK_MODEL = Read-Host '本账号可用的 DeepSeek 模型 ID'
```

先检查计划，再每次最多新标注20个案件：

```powershell
python -m egc annotate --input data/review/pilot.jsonl --rules data/rules/reviewed.json --output data/annotations/pilot.jsonl --model $env:DEEPSEEK_MODEL --limit 20 --dry-run
python -m egc annotate --input data/review/pilot.jsonl --rules data/rules/reviewed.json --output data/annotations/pilot.jsonl --model $env:DEEPSEEK_MODEL --limit 20
```

API只接收案情、给定罪名与条件描述，不接收参考理由和刑期。每个条件给三态判断与原文quote；本机计算Python Unicode字符偏移并验证引用确实存在、可唯一定位。supported/refuted必须有证据，证据匹配仍不能证明语义判断正确。

结果均为draft，需抽查并独立估计错误率。每次 `--limit` 限制新增案件，不是金额上限；临时HTTP错误每案最多3次尝试。缓存与输入/规则/模型/提示版本绑定，重新运行可续做。失败返回不默默变unknown，避免污染条件质量；模糊网络失败不会无条件自动重发。

某次只完成20条时，继续同一命令直至标注覆盖整个pilot，再prepare；prepare遇到未标注案件会失败。当前无自动人工修订UI，可以直接编辑条件状态/证据，但不得修改输入指纹。改了案情或规则则重新标注到新的输出文件。

```powershell
python -m egc prepare --input data/review/pilot.jsonl --variant concat --rules data/rules/reviewed.json --annotations data/annotations/pilot.jsonl --output-dir data/prepared/concat/pilot
python -m egc prepare --input data/review/pilot.jsonl --variant egc --rules data/rules/reviewed.json --annotations data/annotations/pilot.jsonl --output-dir data/prepared/egc/pilot
```

两个变体必须使用同一份规则和条件证据。只有egc增加组合状态，使增益来源可检验。正式test标注只在规则/提示冻结后执行，参考标签始终与模型输入分离。
