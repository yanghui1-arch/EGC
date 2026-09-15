# E2：本机独立事实抽取试验

2026-09-15更新：用户已执行本页第一轮命令，18个任务为11结构通过、4拒绝、3无可用响应；bound v1尚不可扩标。详见[首轮诊断](E2_REGRESSION_V1_REPORT.md)。本页命令保留作v1复现记录；新增的第二轮软件已交付，请使用[新版12请求运行步骤](RUN_FACTS_E2_V2.md)，无需再次prepare或标注60条新样本。新版真实效果仍未验证。

## 三组及实验解释

| 模式 | 处理 |
|---|---|
| joint | 保留旧版分析＋条件联合提示，本轮重新调用以便同批比较 |
| flat | 不生成分析，直接输出条件状态和引用，补充明确反证等判断边界 |
| bound | 与flat共享判断边界；先输出每个观测的主体、事件、时间、来源、极性和引用，本机再组合为状态 |

主比较是flat与bound。joint与flat还存在边界提示差异，不能把其差异仅归因于删除分析。flat与bound的格式及提示长度不同，也需要报告token消耗；这是整个绑定方案的先导比较，不是单个字段的严格因果消融。

bound的主体/事件是逐观测绑定，不是完整实体消歧系统或全案事件图。明确代理退赔可用于退赔条件，其他主体不会直接归给被告；前科应绑定前案，其他条件绑定本案；单次未得逞不扩大到全案。归属门控失败、正反观测冲突或模型声明不足时保留unknown。主体关系、极性和范围仍由模型预测，本机校验不证明它们正确，也可能增加弃答。

## 第一轮：只跑6例、18次请求

本机代码已写入 `D:\workspace\codes\COLING\EGC`，此目录无需再执行git pull。若使用另外克隆的Git checkout，则先在该checkout更新代码并提供相同输入文件。

三个真实输入文件已存在于本机：筛查后train、旧20条输入、旧200条输入。prepare只读取它们并冻结抽样，不调用API，不读取test。6条回归案例及其预期状态来自既有Codex审阅，不是人工金标准。预期状态不进入任何API请求。

在PowerShell中依次执行，某一步报错就先回传该错误，不继续后面的步骤：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc facts-prepare --pool data/cail_screened_v5/train.jsonl --exclude data/cail_pilot_v2/pilot20.jsonl data/cail_pilot_v3/pilot200.jsonl --output-dir data/facts_e2_v1
python -m egc facts-run --input data/facts_e2_v1/regression.jsonl --profile data/facts_e2_v1/profile.json --output-dir data/facts_e2_v1/regression_run --limit 18 --dry-run
```

prepare生成6条回归输入、目标诈骗/抢劫各30条的新样本、冻结的profile及selection_manifest。新样本排除所给历史输入的ID、同来源案号、相同事实和高词面重叠；不足时显示实际数量，不借用dev/test。词面筛查不保证语义独立。此步骤只准备新样本，**不会标注那60条**。

dry-run应显示cases=6、三个modes、total_requests=18，首次new_requests_this_run_at_most=18，api_calls=0、writes=0。重复时剩余请求可能减少。确认这些数值后运行：

```powershell
python -m egc facts-run --input data/facts_e2_v1/regression.jsonl --profile data/facts_e2_v1/profile.json --output-dir data/facts_e2_v1/regression_run --limit 18 --ask-key
python -m egc facts-report --input data/facts_e2_v1/regression.jsonl --run-dir data/facts_e2_v1/regression_run --expectations data/facts_e2_v1/expectations.json --output data/facts_e2_v1/regression_run/comparison.json
```

Key在终端隐藏输入；不要写入命令参数或回传文件。若已有DEEPSEEK_API_KEY环境变量则直接使用，不再次询问。脚本固定使用系统curl和既有DeepSeek端点，无自动网络重试。若报curl不存在，回传报错即可。

本轮最大新增HTTP请求18次（6案件×3组），每次输出上限4096 tokens，输出预算合计最多73728 tokens，输入另计。实际时长和token数写入报告；不预先保证费用或运行速度。没有GPU操作。

## 结果与断点行为

- `selection_manifest.json`：样本及配置哈希、数量、排除情况。
- `regression_run/manifest.json`：固定输入/配置/请求哈希、各状态数量和完成情况。
- `regression_run/records.jsonl`：各案件各模式的校验结果、证据和派生轨迹；不会改原始案件、标签或旧200条输出。
- `regression_run/comparison.json`：完整覆盖率、各条件状态、成对变化、与模型审阅预期的一致情况；失败也计入预期检查总数。
- `raw/`：本机响应缓存，不上传Git。成功HTTP响应在内容校验前落盘，截断或格式错误也不会自动重付费。

首次prepare后不必再prepare同一目录；它会拒绝覆盖已冻结样本。后续相同facts-run命令使用缓存，仅运行未尝试的请求。输入、profile、模型或预算变化需新输出目录。

网络异常/中断留下的请求标为结果不明确并停止当前批次；再次运行会跳过该请求，继续其它未尝试项，不自动重试不明确请求。`complete=true`仅表示所有项目已有处理状态，需同时查看all_structurally_valid和statuses；不代表语义正确。进程强制终止可能留下`.run.lock`，确认没有运行中的进程后才可人工移除；不要删除raw里的pending/error记录以盲目重试。

## 回传这四个文件

1. `data/facts_e2_v1/selection_manifest.json`
2. `data/facts_e2_v1/regression_run/manifest.json`
3. `data/facts_e2_v1/regression_run/comparison.json`
4. `data/facts_e2_v1/regression_run/records.jsonl`

可用下列命令打包后回传；不要包含Key或环境变量：

```powershell
Compress-Archive -LiteralPath data/facts_e2_v1/selection_manifest.json,data/facts_e2_v1/regression_run/manifest.json,data/facts_e2_v1/regression_run/comparison.json,data/facts_e2_v1/regression_run/records.jsonl -DestinationPath data/facts_e2_v1/feedback.zip
```

本轮结束后先分析漏标、反证、过度unknown及token开销。回归一致率不是准确率，也不能用于论文结论；新样本运行和下一轮调整等结果分析后再给命令，当前不启动全量标注或训练。
