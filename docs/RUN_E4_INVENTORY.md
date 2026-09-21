# E4：一次零API的训练池规则线索盘点

状态：用户已完成真实127条盘点，助手验收及有界复核完成，见[实际结论](E4_INVENTORY_RESULT.md)。**不要重复运行下方历史命令。** 当前草案扩标/训练路线停止，下一步B0专用包待助手实现；106项为既往离线测试结果。

在本机PowerShell运行（本机EGC不是Git checkout，不在本机执行git pull）：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc.e4_inventory --train data/cail_reviewed_v6/candidate_train.jsonl --source-zip data/raw/CAIL2018_ALL_DATA.zip --rules configs/rules_e4_draft.json --expected-train-hash c13b5f4b1253cc4a6dabe95eaf5a8bf0b91600f47e875839348738b349ba9248 --output-dir data/e4_inventory_v1
```

输入均已在本机：v6 train候选、官方CAIL zip、E4规则草案。脚本仅读官方small train成员，不读取dev/test成员。不会解压或改写原始数据。

输出：

- `D:\workspace\codes\COLING\EGC\data\e4_inventory_v1\report.json`：输入/规则/待核表哈希、线索计数、目标姓名未匹配数；正常预期979条输入、选出诈骗73与抢劫54共127条，实际以运行结果为准。
- `D:\workspace\codes\COLING\EGC\data\e4_inventory_v1\review_cases.jsonl`：仅本机保留的完整案情、目标人物来源、关键词跨度、空白条件与适用版本待核表。**不可提交GitHub。**

预计API请求0、模型调用0；只做CPU读取与匹配。线索数不代表条件成立数，`reviewed_state=null`表示尚未审查，并非unknown。源目标姓名匹配也不代表事件绑定正确；所有判断须结合全文。输出目录必须不存在，发生错误保留报错，不覆盖旧目录或盲目重复运行。

运行完把终端输出贴回，或告诉助手完成，助手即可从本机读取上述两个文件。若报错，回传完整报错，不需要重跑旧清洗、E3标注或服务器smoke。

下一步由助手复核最多12个新增train候选，更新科研主记录并决定缩窄规则、准备小试或明确当前任务不适配；不会看到关键词就直接扩大付费标注。
