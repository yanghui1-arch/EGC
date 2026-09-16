# E3：冻结60例、七条件的独立新样本对照

2026-09-16交付。软件已实现，85项使用虚构输入/模拟API的离线测试通过。用户运行本机数据/API脚本；助手本轮没有启动真实API、数据作业或模型推理。E3实际效果尚未验证。

## 研究范围与冻结规则

E2两轮已结束，不再针对6个已知回归例修改判断规范。第二轮12份响应中11结构通过、1因unknown带引用被拒绝；绑定没有表现出已证实的语义增益，见[实际结果](E2_REGRESSION_V2_REPORT.md)。

本轮比较`flat_e3`与`bound_e3`，版本`fact-seven-context-e3-v1`。仅保留七项条件：主动到案、如实供述、退赔、谅解、前科、犯罪时未成年、次要作用。七项定义逐字保留v3，暂排除范围建模不清的全案未得逞。它不是完整量刑因素集合，条件草案不等于有来源的正式法律规则。

两组继续共享相同原文句段、条件和事件定义；仅移除被排除条件的事件类型，并调整unknown的引用协议。flat的unknown可以引用解释不确定性的原文；bound的unknown也收集相关观测上下文。这些内容仅写入`uncertainty_context`，三态annotation中的unknown仍保持`evidence=[]`。它们不能作为规则已满足/已否定的证据。没有抽到事件的派生原因改为`no_observation_extracted`，避免冒称材料未记载。

旧v1/v2模式、缓存和拒绝结果不变，不重算历史通过率。E3协议和范围已变化，因此不能把E2/E3通过率差异写成语义增益。这是范围缩小与输出协议修改，不是又一轮针对6例的规范调参。仍没有独立语义验证器，事件类型和主体识别仍可能出错。

输入固定为之前选出的60条train：诈骗/抢劫各30。输入内容与顺序哈希为`7c32d6b3ea030124e9f32a755cfe929fba102ae2dd79d66eb1873831f0a9a157`。本轮不重新抽样、不使用dev/test、不重新读取6例给教师。冻结协议在`configs/facts_e3_protocol.json`，包含输入/配置哈希、罪名数量、模型、模式、token上限和120请求总量；运行前必须全部匹配。

## 本机PowerShell命令

本机代码已更新，无需在这个无.git的目录执行git pull。输入已存在于`data/facts_e2_v1/new.jsonl`，无需prepare。

先验证冻结协议和计划，不调用API或写数据：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
powershell -NoProfile -File scripts/local_facts_e3.ps1 -DryRun
```

首次应显示`cases=60`、`conditions=7`、`total_requests=120`、`protocol_verified=true`、`api_calls=0`、`writes=0`。协议校验失败就停止，不改样本凑数量，不手动改哈希绕过检查。

确认后正式运行：

```powershell
powershell -NoProfile -File scripts/local_facts_e3.ps1
```

Key在终端隐藏输入；已设置DEEPSEEK_API_KEY时直接使用。单次最多120个新HTTP请求（60×2），每个输出上限4096 tokens，输出预算上限491520 tokens，输入另计。实际响应长度、时间和费用不能由E2保证。脚本不使用本机GPU，不自动重试不明确的请求。

输出为`data/facts_e3_v1/new_run/`。脚本执行dry-run、正式请求和report；发生结构/API失败时保存已有进度，生成报告后返回退出码3。报错时先交回结果，不重复运行、删缓存或改提示。若后来明确续跑，缓存会保留所有已尝试项，仅处理pending，失败项不会自动重试。

底层入口已实现为`python -m egc facts-run --modes flat_e3 bound_e3 --protocol ...`，E3强制要求协议；包装脚本已固定所有路径与设置。

## 产物与核验方式

| 文件 | 内容 |
|---|---|
| `manifest.json` | 输入和配置、冻结协议及哈希、120任务状态、实际本次新增请求数 |
| `records.jsonl` | 结构校验、三态、原文证据、unknown上下文、绑定派生过程 |
| `comparison.json` | 成对状态差异、token和耗时、逐条件状态/失败数量、逐罪名覆盖率 |
| `review_queue.jsonl` | 60×7=420个条件位置，每项并列两组输出；含unknown、相同判断、pending及失败 |
| `raw/` | 本机原始响应及脱敏故障记录；不向Git上传 |

报告不产生“准确率”，`training_ready`保持false。420项是审阅位置，不是人工标注；每项可能有两组判断，总共最多840个条件预测。重新report会重建review_queue，后续审阅意见应另存文件。

运行后告诉助手查看本机结果即可；需要回传时给manifest、comparison、records和review_queue四个文件，以及终端脱敏错误。报告不会自动进行人工或模型语义审查。

## 预先确定的下一步

先确认全部120个任务的实际覆盖，失败不能被剔除后声称完整集提升。审阅420个位置的原文支持关系，分别记录无依据肯定/否定、漏标、不必要unknown、主体/事件错误和规范分歧；不同罪名及条件分开报告。稀少或没有正负例的条件必须注明无法充分验证，不用大量unknown抬高结论。

模型辅助审阅不冒称人工金标准；法律准确率需要独立人工核验与分歧处理。此轮先用于决定工程方案，不报告benchmark提升或显著性。若绑定没有可解释的语义收益，采用更简单的平面抽取，不强求H3成立，把后续投入转向有来源的规则和同证据egc/concat核心H2对照。若两组都出现系统误判，则先收窄条件或调整任务，不能放行全量标注/训练。

E3结果出来前不继续改提示，也不运行服务器训练。训练及本地模型推理仍只在用户服务器执行。
