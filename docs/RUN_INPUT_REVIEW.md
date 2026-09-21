# 本机应用已复核的隔离清单

2026-09-21更新：用户已执行且v6快照验收通过，实际979 train / 112 dev、隔离73、待审153。以下命令保留为历史复现，无需重跑。当前转入[E4规则先导](E4_RULE_PILOT.md)，本轮完整软件测试102项通过；[审计报告](INPUT_AUDIT_V3_REPORT.md)保留当时的处置依据。

## 用户运行

本机代码已更新，无需在无`.git`的目录拉取。Python标准库即可，无API、无Key、无GPU。

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc apply-input-review --train data/cail_screened_v5/train.jsonl --dev data/cail_screened_v5/dev.jsonl --audit-dir data/input_audit_v3 --decisions configs/input_review_v3.json --output-dir data/cail_reviewed_v6
```

输入为原1047 train / 117 dev、刚完成的三份审计文件，以及已编写的模型辅助复核清单。哈希不匹配会停止，不要手动改哈希绕过；非空输出目录也拒绝覆盖。

## 输出与回传

新目录`D:\workspace\codes\COLING\EGC\data\cail_reviewed_v6\`包含：

| 文件 | 内容 |
|---|---|
| `candidate_train.jsonl` / `candidate_dev.jsonl` | 剔除隔离行后的候选，原行内容和划分不变 |
| `quarantine.jsonl` | 每条隔离原记录、主理由、重复关系及审阅者类型 |
| `pending_review.jsonl` | 候选中尚未裁决的自动风险，不自动清零 |
| `review_plan.json` | 实际执行的清单副本 |
| `manifest.json` | 来源/产物哈希、数量、限制及training_ready=false |

按只读清单核算预计隔离73行，留下979 train / 112 dev，其中153行自动风险仍待审；这些数量尚不是已执行结果。两罪名预计剩127 train / 18 dev，后续用于规则先导准备，未放行训练。

跑完告诉助手完成，助手读取manifest、隔离清单和pending_review，必要时核验候选。若报错，提供终端报错；保留原文件与新目录，不重复删数据或重跑旧E3。不要把上述含原文的数据输出提交Git。

## 接下来的研究工作

本步骤只把已知问题与候选分开，既不改标签，也不自动截断说理。原E3结果不变。助手下一步核验应用结果并准备两罪名的真实规则来源、适用版本和可观测条件，围绕同证据的concat/egc对照推进。其余罪名待审项继续保留，不让全12罪名清理成为先导的无限前置工作；服务器暂不启动训练。
