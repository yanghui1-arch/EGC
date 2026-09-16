# E2第二轮：编号句段与事件类型映射

2026-09-15结果更新：用户已运行12请求，bound_v2为6结构通过，flat_v2为5通过/1拒绝。已完成全条件模型辅助审阅，详见[实际报告](E2_REGRESSION_V2_REPORT.md)。本页命令保留作复现记录，当前无需重复；2026-09-16已交付[E3七条件新样本入口](RUN_FACTS_E3.md)，由用户按新步骤运行。尚无准确率或benchmark提升证据。

## 本轮改动与研究边界

- 保留v1三个模式、提示和旧缓存；新增`flat_v2`与`bound_v2`，版本`fact-events-v2`，使用新的`configs/evidence_profile_v3.json`。不覆盖旧6例输入或旧回归预期。
- 本机按原始标点划分句段并附ID及字符位置，模型只返回句段ID。本机由ID恢复原文证据，重复姓名不再因重复复制的短引用导致歧义。句段可能包含多个主体/事件，定位成功仍不能证明支持关系。
- 两组共享输入句段、条件、事件定义和判断边界。flat_v2直接给三态；bound_v2只提取事件类型及actor/scope/coverage，由本机固定映射为条件状态。无相关事件允许空列表，不强制八项占位观测。v1逐项时间、陈述来源和重复引用字段在本轮移除，来源语义保留在完整原文中。
- 映射区分明确抓获与归案方式不明、如实供述与只记供述、本人/代理人退赔与追缴。归属不清、明确冲突以及未覆盖全案的结果保留unknown。它不是关键词纠错器，也没有按6个案件ID写特判。
- 尚未实现独立语义验证模型。映射依赖教师正确识别事件类型和归属；错把追缴分类为退赔仍可通过结构校验。下一轮要审查事件类型是否受原文支持，不能用映射测试代替语义准确率。
- 传唤后又称投案、但未独立明确主动性的混合表述，本轮操作定义固定为unknown；这是一项保守研究约定，不是法律结论。规范已版本化，不能改写v1结果使其符合新定义。

主要比较是同批flat_v2/bound_v2。因为事件类型、引用格式、字段和规范共同变化，v2与v1的差异不能归因于单个模块；本轮不是严格的“只增加绑定”消融。核心H2仍须后续同证据的egc/concat实验验证。

## 在本机PowerShell运行

输入是已冻结的`data/facts_e2_v1/regression.jsonl`（6例），无需重新prepare，也不读取60条新样本或dev/test。当前本机目录已更新，无需git pull。另一个Git checkout需先更新代码并准备该输入。

先只检查计划，不调用API或写数据：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
powershell -NoProfile -File scripts/local_facts_v2.ps1 -DryRun
```

首次应显示cases=6、modes为flat_v2/bound_v2、total_requests=12、new_requests_this_run_at_most=12、api_calls=0、writes=0。然后运行：

```powershell
powershell -NoProfile -File scripts/local_facts_v2.ps1
```

脚本先dry-run，再隐藏输入Key（已有DEEPSEEK_API_KEY环境变量时直接使用），最后生成比较报告和审阅队列。最多12个新HTTP请求，每次输出上限4096 tokens，输出预算合计最多49152 tokens，输入另计；实际费用未知。固定curl、无自动网络重试，不使用本机GPU。

输出目录为`data/facts_e2_v2/regression_run`，包含manifest、records、raw、comparison以及review_queue。review_queue共6×8=48项，每项并列两组结果和原文；未提及、相同判断、失败输出也保留，review_state初始为空。它是待审清单，不是48个人工标签。重新report会重建该清单，审阅意见应另存文件。

脚本没有使用旧6个单条件预期打分，避免漏掉新增错误和规范分歧。对下一轮所有条件进行模型辅助审阅，正式准确率仍需要独立人工核验。未发起新的joint请求，以集中检验flat/bound差异和减少调用。

## 出错时

出现API或结构拒绝时，facts-run保存进度并返回退出码3；包装脚本仍生成可读报告，然后返回3并要求停止。鉴权/HTTP故障只记录状态码，curl故障只记录退出码/类别；不记录密钥、错误正文或stderr。旧日志中缺失的诊断无法补回。

`complete=true`仍只表示无pending，必须看failed_requests和statuses。无响应任务不自动重试；原命令再次运行仅处理尚未尝试项，未知任务继续保留。不要删除缓存来盲目重试；出现错误先反馈结果。没有独立重试失败项的命令，本轮也不授权自动重试。

## 跑完交回的结果与继续标准

本机运行后告诉助手查看以下文件即可：

1. `data/facts_e2_v2/regression_run/manifest.json`
2. `data/facts_e2_v2/regression_run/comparison.json`
3. `data/facts_e2_v2/regression_run/records.jsonl`
4. `data/facts_e2_v2/regression_run/review_queue.jsonl`

本机raw用于核验，不向Git上传。若有错误，附终端显示的脱敏错误类别/代码。

这是E2/E3预算内的第二轮规范试验。先要求本轮12任务具备可审阅响应，并检查全部48个条件位置中两组的状态和证据；特别检查明确反证、主体误归和追缴误归。回归集只能用于发现问题，不能证明泛化或H3成立。未解决的语义错误、明显新增错误或大范围弃答不放行60条扩标；届时缩小条件集合或放弃该绑定实现。通过开发审查后才另行给冻结60条上的比较命令，仍不直接启动训练。
