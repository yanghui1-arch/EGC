# Qwen2.5-7B：训练完成，生成结束异常，方法比较不可用

**2026-09-27最新进展：CPU和两例GPU探针均已完成，下方旧命令无需再执行。** 两例Transformers生成同样失败，当前下一步为[32步原生EOS修复小试](RUN_QWEN25.md)，真实修复效果待验证。

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

## 2026-09-27：CPU实测与下一步

用户回传`runs/qwen25_stop_diagnostic.json`，SHA256为`98d726a3fd3cde96907d649eecfec13af935d66730c1b8e7806e2b771d350de9`。run提交、模型路径、三个adapter配置与原结果包一致。

- 三个adapter的词表和模板均与基座一致，EOS均为151643；不支持tokenizer保存后发生错位的解释。
- `𬭤`为token123352，`䏡`为122500。其lm_head行与im_end151645行均**不完全相同**，最大逐元素绝对差分别0.004241943359375和0.01910400390625。此前针对这两个异常字符的“完全相同输出行”猜测未获支持。
- im_start151644与im_end151645的lm_head行最大差为0.00006103515625，embedding最大差为2.350988701644575e-37。这里只证明数值差异很小；没有隐藏状态/logit测量，不能据此断言模型无法区分或证明根因。未扫描整个词表，亦未验证vLLM实现。

实现了同一诊断入口的`--generate`模式：从原direct dev任务文件固定取前两条，不按参考刑期选样；核验原预测指纹、模型/adapter配置、权重文件清单及渲染prompt hash。用Transformers+PEFT加载已有direct adapter，bf16、greedy、相同2048输出上限、EOS及im_end停止条件，生成两次。另在原vLLM输出的首个JSON边界做两次前向计算，记录停止token及异常字符的logit排名。只读取推理任务与原预测，不读取刑期参考、不计算MAE、不替换原始结果、不修改磁盘权重。

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m egc.diagnose_qwen_stop \
  --generate \
  --run-dir runs/learned_v2_screened_qwen25_7b_seed42 \
  --output runs/qwen25_generation_probe.json
```

将`runs/qwen25_generation_probe.json`回传本机runs目录。生成使用GPU 0；CPU诊断无需重做。该检查存在引擎数值和批大小差异，2例结果只用于定位：若Transformers也续写，优先修正训练终止协议/输出层；若能正常终止，进一步检查vLLM LoRA路径，暂不重训。任一分支均不能凭2例宣称全dev已经修复。

相关14项离线测试通过，涵盖原审计、词表/格式适配、向量比较和JSON边界定位；GPU探针尚待服务器实际运行。API调用方式参考[Transformers生成文档](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)与[PEFT已训练adapter加载文档](https://huggingface.co/docs/peft/main/en/package_reference/peft_model#from_pretrained)。本轮无新增训练或方法效果，H4仍未获支持。

## 2026-09-27：GPU实测与有界修复实验

用户回传`runs/qwen25_generation_probe.json`，SHA256为`e529d158a78abeaabee077c3857e88747ec42997e03a594b875f19702c7e4a90`。选中ID及嵌入的原vLLM记录与原结果包direct前两条完全一致；运行提交58880be、模型/adapter配置和渲染prompt身份相符。Transformers 5.17.0、PEFT 0.20.0，用原adapter、bf16、greedy、2048上限，停止列表151643/151645。

| 案例ID后缀 | Transformers结果 | 原JSON边界im_end排名 | 原生EOS排名 |
|---|---|---:|---:|
| 522ea8d6008aa0df1b63 | length，2048，JSON后继续生成 | 131 | 12676 |
| c4a3456f9381f3ec2b59 | length，2048，JSON后继续生成 | 149 | 13024 |

第一例边界最高logit为11.625（包含`:UIButtonType`），im_end为10.375；第二例最高为异常字符𬭤的11.75，im_end为10.4375。两引擎尾随文本不完全相同，但均未正常终止。该有限对照支持模型侧终止学习失败，不能只解释为vLLM忽略了停止设置，也不证明输出层在数学上无法学会im_end。原始1584条失败仍保留，未补算刑期分数。

本轮修复是待验证的工程选择：Qwen2 Base本地config和tokenizer均以151643为EOS时，仅将训练序列最后assistant的`<|im_end|>`及其尾随换行替换为`<|endoftext|>`。保留全部输入ChatML、JSON正文、attention LoRA范围、学习率等超参；不解冻输出层、不把乱码设为停止符、不截取JSON追分。其他模型和以im_end为EOS的Qwen2 Instruct维持原终止序列。预检和训练共用分词函数，新协议`explicit-no-thinking-qwen2-native-eos-v3`进入身份记录；旧实验不得续跑混用。还在训练开始前核验真实TRL collator没有把EOS当PAD屏蔽。

实现`egc.qwen_eos_smoke`复用现有解包、训练、推理、指纹验证和结果收集函数。服务器从基座新训direct 32个优化步，随后按既定dev顺序每罪名首条共12条生成；选样仅读取给定罪名，不看参考刑期。仍用原完整训练包4757/528、seed42、lr1e-5、rank64、max8192和2048生成上限。这里的32步只是低成本软件/终止行为检查，不是充分收敛实验；失败不能直接否定原生EOS路线。12/12完整JSON并正常停止才放行新协议三组完整dev复核，通过亦不代表全dev成功或刑期准确率提升。正式test冻结，H4无新支持证据。

142项离线测试通过，包括新协议的输入保留/EOS监督、非Base格式保留、模拟32步调度及成功/截断门槛、结果包排除权重。没有在本机运行真实训练、模型推理或API；实际小试待用户按RUN_QWEN25执行并回传结果包。
