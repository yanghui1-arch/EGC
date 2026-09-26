# Qwen2.5-7B：训练完成，生成结束异常，方法比较不可用

2026-09-26只读验收用户回传的`runs/results_qwen25_7b_seed42.zip`。该包证明此前目录冲突之后，任务最终完成；不推断用户如何处理旧目录。

## 核验事实

- 结果包SHA256：`7ab5605cb91f824405ec0c082ba046a995b550fe8d7a27b4095443edfd392378`。
- 服务端代码：`58880be85068ade3719fd3611e3e94cae594a0cd`。
- 输入包SHA256：`55e6eb8c69e5301f0efaa14c43fa96daeacbe1fbd6132c2bc270795008149622`；4757 train / 528 dev，与1.7B同包。
- 本地基座`/mnt/yanghui/models/Qwen/Qwen2.5-7B`，7615616512参数；每组40370176可训练参数，LoRA rank64/alpha128/dropout0.05，仅q/k/v/o。
- 三组均894步、3epochs、seed42、最终epoch保存。七个子进程均exit_code=0，合计33441.97秒（9.29小时）；完成标记为dev_only，test未运行。
- 26个数据成员的校验和、源包/manifest、训练输入hash、预测ID/prompt/generation指纹、模型/adapter路径和推理预算核验通过；四组原始指标重算一致。训练损失下降并不能保证生成有效。

## 原始评估（保留不改）

| 方法 | 有效输出 | 覆盖率 | 有效子集MAE/月 | 达到2048 token上限 |
|---|---:|---:|---:|---:|
| train逐罪名中位数 | 528/528 | 100% | 17.0019 | 不适用 |
| frozen | 522/528 | 98.86% | 21.0000 | 4 |
| direct | 0/528 | 0% | 不可计算 | 528 |
| flat | 0/528 | 0% | 不可计算 | 484 |
| bound | 0/528 | 0% | 不可计算 | 447 |

frozen有效子集RMSE为33.8341月，full指标仍为空。三组SFT的full及valid-only指标均为空，共同有效集合为空，不能比较H4。不能把空值说成0误差，也不能将本次失败当作绑定效果已被否定。

对全部1584条SFT输出用标准库JSONDecoder做只读边界检查：每条开头均存在一个满足既有reasoning/sentence_months结构要求的完整JSON，随后都有额外内容。该检查仅定位故障；未截取JSON用于评分、未修改预测或原评估器，也未核验理由语义正确。direct/flat/bound分别504/389/267条的第一个尾随字符为“𬭤”，另有5/98/204条为“䏡”。定向查看各组首两条，可见随后续写案情、角色文字或乱码。1584条中1459条达到长度上限，另外125条虽停止仍包含额外内容。

## 已知配置与待确认原因

返回配置显示基座EOS为151643（endoftext），训练/预检/推理均使用同一checkpoint模板hash，推理已明确加入151645（im_end）停止条件。因此不能把问题简单归为忘加stop token。LoRA没有训练lm_head或embedding；此前适配仅验证ChatML token ID与模板一致，没有检查冻结输出层能否区分这些结束标记，检查不充分。

目前较具体的待证假设是：Base权重中的im_end输出向量与某些普通token相同或难以区分，注意力LoRA无法改变输出向量之间的相等关系。若输出层两行完全相同，则对任意隐藏状态其数学logit相同；但服务器实际权重尚未检查，不能宣称已找到根因。训练标记、tokenizer保存及推理引擎差异仍需排查。

检索线索：[另一Qwen2.5-1.5B实验作者的模型卡](https://huggingface.co/miscusi/adaption-market-analyst-qwen2.5-1.5b)报告冻结输出层时im_end终止失败；[Qwen早期官方训练说明](https://github.com/QwenLM/Qwen/blob/main/README.md#finetuning)也区分Base LoRA与聊天模型的embedding/output训练。这些涉及不同checkpoint，只是诊断线索，不能代替本次7B本地权重测量。

## 下一步：只读CPU诊断，暂不重训

已实现`egc.diagnose_qwen_stop`：在服务器读取基座及三个adapter的tokenizer、检查词表/模板一致性，并用safetensors仅读取lm_head与embedding中结束标记和两个已观察异常字符对应的少量行，比较是否完全相同。不加载完整模型、不推理、不调用API、不修改权重，仅写新的诊断JSON；需要现有transformers/torch/safetensors。

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
python -m egc.diagnose_qwen_stop \
  --run-dir runs/learned_v2_screened_qwen25_7b_seed42 \
  --output runs/qwen25_stop_diagnostic.json
```

用户回传`runs/qwen25_stop_diagnostic.json`。若运行目录已改名，使用保存本次execution.json的实际目录。保留原adapter和全部结果。根据诊断决定修正训练结束标记/输出层或做少量跨引擎对照；不盲目重跑三组、不增加token上限、不把异常字符写成针对本批结果的停止规则。正式test继续冻结；H4仍未获支持，1.7B的负面绑定证据保留。

本次软件修复仅让`learned_audit`在共同有效集合为空时返回空指标与明确不可比较原因，原评估器、模型训练和推理代码未改。13项相关离线测试通过，包括模拟正/零覆盖审计及向量相等诊断；真实结果只读审计通过。诊断入口帮助检查通过，服务器权重诊断待用户执行，不能把合成测试称为实际根因验证。
