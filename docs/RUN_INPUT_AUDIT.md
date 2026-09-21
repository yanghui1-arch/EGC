# 下一步：本机全候选池离线审计

2026-09-21更新：用户已完成审计，见[实际报告](INPUT_AUDIT_V3_REPORT.md)。不必重跑以下历史命令；当前下一步是[应用已复核隔离清单](RUN_INPUT_REVIEW.md)。E3也已完成，不需要重新运行API。

## 命令

本机代码已更新，这个目录没有`.git`，无需本机git pull。仅用Python标准库，无API调用、无API Key、无GPU，也不下载数据。

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc audit-inputs --inputs data/cail_screened_v5/train.jsonl data/cail_screened_v5/dev.jsonl --focus-input data/facts_e2_v1/new.jsonl --output-dir data/input_audit_v3
```

输入为原候选池1047 train / 117 dev；focus是已完成E3的冻结60例，必须原样属于输入池。审计覆盖整个1164行候选池，focus只提供子集摘要，不限制检查范围。默认字符5-gram近重复阈值0.85；同时检查同一集合内及不同集合间，不把不同ID自动视为独立案件。

## 输出与回传

新增目录`D:\workspace\codes\COLING\EGC\data\input_audit_v3\`，含：

| 文件 | 用途 |
|---|---|
| `report.json` | 输入哈希/数量、风险行计数、近重复及跨划分数量、E3子集摘要 |
| `review_queue.jsonl` | 全部输入行、风险片段位置、近重复索引及空白复核字段 |
| `overlap_pairs.json` | 重复候选对、相似度、跨划分与标签差异标志 |

运行后告诉助手完成即可，助手读取本机这三个文件。若报错，提供终端报错文本。不要上传原始案情或这些输出到GitHub。

输出目录非空时程序拒绝覆盖，以免丢失复核意见。若已经完成，不必换目录重复跑，先分析已有结果。

## 解释边界

脚本仅标记复核候选，不修改或删除事实、标签、划分，不自动做训练放行。正则可能把合法前科刑期列为候选，文本相似也不自动等于同案；未标记不保证输入干净。复核重点是当前案件量刑信息/说理混入、目标主体/罪名范围，以及同案或近重复跨集合问题。

不改E3历史输入和结果，也不重新运行`facts-prepare`。下一步根据复核决定可追溯清理和成组去重，再交付新版本数据命令；这些清理动作本命令尚未执行。真实规则先导与GPU实验仍待后续准备。
