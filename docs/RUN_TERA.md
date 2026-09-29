# GPU 1：TERA模块先导实验

**2026-09-29当前下一步：seed42已验收，固定配置补seed43/44。** 详见[结果报告](TERA_SEED42_RESULT.md)。无需再次恢复seed42。新两轮使用同一数据包、模型、四组对照与训练/解码设置；有改善信号，但尚未确认最强基线优势和机制有效性。

在服务器GPU 1顺序执行两轮（真实作业由用户运行）：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
for seed in 43 44; do
  CUDA_VISIBLE_DEVICES=1 python -m egc.tera_server run \
    --archive data/learned_v2_screened/experiment.zip \
    --output "runs/tera_qwen25_7b_seed${seed}_$(date +%Y%m%d_%H%M%S)" \
    --model /mnt/yanghui/models/Qwen/Qwen2.5-7B \
    --seed "$seed" --epochs 3 --max-length 8192 || break
done
```

每轮先32步/12例小试，再从基座新训direct/generic/tera_noaux/tera并评528 dev；seed只改变训练随机性，数据划分不变。新目录不加resume，不续训seed42。任一轮失败即停止，保留权重和日志并回传失败包。

每轮最后会打印`Return experiment evidence:`路径。将两个结果ZIP分别命名为`tera_seed43_results.zip`、`tera_seed44_results.zip`放回本机`D:\workspace\codes\COLING\EGC\runs`；改ZIP外部文件名不改变内部校验。不要覆盖seed42结果，不额外调用DeepSeek，不运行test。

下面保留seed42恢复与历史诊断说明；当前执行以上新种子命令。原论文同配置复现、60例标注审阅、去目标/路由干预仍未完成，不能将多seed运行等同于这些工作。

---

**2026-09-28当前操作：诊断已验收，恢复缺失实验组。** 详见[缓存验收](TERA_CACHE_RESULT.md)。已保存组的BF16差异在FP32下显著下降，新增统一精度对照验收及组级恢复；14项相关CPU检查通过，真实恢复仍待运行。direct/generic权重和预测可在核验后复用；tera_noaux无导出需重训，完整tera尚未开始。无需API或本机处理，不要重新运行整套server_tera.sh。

服务器执行：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=1 python -m egc.tera_server run \
  --archive data/learned_v2_screened/experiment.zip \
  --output runs/tera_qwen25_7b_seed42_20260928_181851_23453 \
  --model /mnt/yanghui/models/Qwen/Qwen2.5-7B \
  --seed 42 --epochs 3 --max-length 8192 --resume
```

顺序：核对源包和准备数据 → 重新加载smoke/direct/generic做同一缓存验收并保留其原预测 → 保存旧失败目录 → 从基座重训tera_noaux、新训tera，各3epochs并生成528 dev → 汇总四组。新版继续在缓存检查前导出权重，失败也保留；不会恢复旧noaux已经丢失的权重。BF16解码、LoRA和训练预算不变；FP32只用于短缓存检查，退出前恢复原dtype，不写回权重。发现已有部分预测或不完整导出会停下，不静默覆盖。

结束或失败时将终端最后`Return experiment evidence:`后提示的**新results_<时间>.zip**放到本机`D:\workspace\codes\COLING\EGC\runs`。旧ZIP和失败日志保留。direct/generic当前均527/528有效，完整MAE不可报告；恢复不替换这条失败预测。没有TERA提升结论，正式test不运行。

---

以下是历史诊断及完整新建实验说明，已被上面的恢复步骤取代。

**2026-09-28最新：暂不重跑整套训练，先诊断训练后缓存检查。** 用户日志显示`tera_noaux`的3epochs训练已返回，随后缓存检查报`logits=0.3125, memory=0.0`。旧门槛为最大绝对差0.25；日志没有argmax、主干误差或精度对照，不能确定是BF16数值差异还是缓存/模块错误。旧代码在该检查后才保存artifact，且save_strategy=no，因此失败组可能没有可恢复权重；先检查文件清单，不承诺恢复。此前的smoke/direct/generic按流程已走过，实际文件和结果仍需核验。

现已将权重导出移动到缓存检查之前，completion.json分别记录training_complete与validation_status。缓存检查失败时保留artifact和校验和，但complete仍为false，不进入生成评分；此修复无法追溯恢复已经退出进程中的旧权重。检查报告新增相同权重/相同续接token下的主干logit差、模块增量差、argmax及margin、最大误差token和RMS。**没有放宽0.25门槛，也没有更改模型/训练/解码设置。**

当前仅执行下面的只读GPU诊断，复用现有run目录：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=1 python -m egc.tera_server diagnose-cache \
  --run-dir runs/tera_qwen25_7b_seed42_20260928_181851_23453
```

诊断先核验已保存artifact校验和、训练身份和dev prompt指纹，缺少导出权重的组明确标为no_exported_checkpoint_to_probe。对至多5个已有模型，各选固定前三条dev输入、各4个续接位置，比较缓存与整段重算；另对各模型第一条输入做同权重FP32对照（关闭TF32、仅在内存转精度，用后释放，不保存转换后的权重）。单个模型依次加载，FP32对照会增加显存；若失败则记录错误并保留已得诊断。没有训练、API、金标输入、指标评分或原预测改写。其他组的诊断不能直接证明未保存的tera_noaux权重为何失败。

输出`原run目录/cache_diagnostic_<时间>.metrics.json`及新`results_<时间>.zip`，均不覆盖旧结果。请将终端最后提示的新ZIP放到本机EGC/runs并告知文件名。先用这份证据决定数值门槛/实现是否需要修改，再给出复用已完成组的恢复步骤；**当前没有新增自动续训命令**。11项TERA离线检查在Transformers5.17下通过，覆盖失败先保存、阈值不放宽、主干/模块分解与只读固定样本诊断；真实GPU诊断待用户运行。以下完整训练命令保留作原流程说明，当前不要用它整套重启。

**2026-09-28启动错误已修复：** 首轮服务器在Trainer.train启用gradient checkpointing时因包装方法不接受every_n_layers而退出，尚无训练步。现原样转发包括every_n_layers/offload在内的kwargs到底层；无需降级Transformers或关闭checkpoint。8项相关合成检查已在Transformers5.17.0/PEFT0.20.0/Accelerate1.15.0及CPU Torch2.6通过，包括实际Trainer训练和四组透传回归；真实A100兼容性待重试。用户日志说明准备阶段已完成，不能把它当训练完成。保留runs/tera_qwen25_7b_seed42_20260928_180425_23415；拉取后执行下方原命令即可自动生成新目录。torch_dtype弃用警告不导致此次退出。接口依据[Transformers5.17官方签名](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/model#transformers.PreTrainedModel.gradient_checkpointing_enable)。

2026-09-28：代码已实现，149项离线检查通过；没有真实TERA训练/benchmark结果。创新假设仍见[研究主记录](RESEARCH_PLAN.md)与[结构及文献](MODULE_RESEARCH_2025_2026.md)。本次只新增独立入口，未修改正在GPU 0运行的H4代码。

## 在服务器运行

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=1 bash scripts/server_tera.sh
```

请在单独的终端/tmux会话执行，保留GPU 0原任务。脚本默认物理GPU 1，只允许单个可见GPU；进程内显示cuda:0属于CUDA重编号。无需DeepSeek Key，也无需本机重洗数据。脚本使用既有`data/learned_v2_screened/experiment.zip`，预期SHA256为`55e6eb8c69e5301f0efaa14c43fa96daeacbe1fbd6132c2bc270795008149622`；包实际hash记录在execution.json，4757 train/528 dev以输入包为准。

脚本为每次调用生成新目录：`runs/tera_qwen25_7b_seed42_<时间戳>_<PID>/`。不要用旧H4目录或旧adapter。模型固定`/mnt/yanghui/models/Qwen/Qwen2.5-7B`，未量化Qwen2系列；实际>=7B采用LoRA，<7B走主干全参。其他架构暂不支持。依赖沿用torch、transformers、peft、accelerate、tensorboard，新增实验不使用vLLM或TRL的训练入口。

执行顺序：

1. 在用户启动的服务器进程中校验、解包既有私有数据；复用bound训练标注作稀疏辅助标签。案情按中文句末标点/换行分段，超过160字符再分块。这只是位置划分，不按法律关键词决定标签。唯一引用且完整落于一个块的关系可用于监督；同块冲突、跨块或重复引用跳过辅助标签，所有案件保留SFT。缺关系/非原文引用直接报错，不填默认值。准备结果只在私有data目录，0次API。
2. 从基座做完整TERA **32步短训**，对固定每罪名首条共12个dev输入检查终止。训练时检查主干LoRA及模块有有限非零梯度；训练后检查缓存一致性；保存并重新加载后核对logits。要求12/12正常停止且完整JSON，否则停止，不启动完整四组。
3. 短训通过后自动从基座分别新训四组，各3epochs、seed42、batch1×累积16、lr1e-5、max8192，随后生成全部528 dev。组内顺序运行，整套GPU 1实验可以与GPU 0的H4并行。**不续用32步权重，不运行正式test。**
4. 计算严格完整输出覆盖率、MAE、RMSE、分罪名误差、月份分布，以及满足全覆盖时的配对bootstrap。无效输出不补默认值、不截取前半JSON追分。最后自动打包结果。

只想执行短训检查时可加`--smoke-only`；这不是默认命令，不自动继续四组。中途故障会保存failure.json及日志并尽量生成results.zip。第一版只导出每组最终模型，不提供中途checkpoint续训；已完成组和故障目录应保留，先返回证据，不盲目重跑整套。

## 四组的作用

| 组 | 结构 | 归属辅助损失 |
|---|---|---|
| direct | 相同Transformers训练/解码入口，普通LoRA | 无 |
| generic | 普通三头案情记忆模块＋并行归属头；归属预测不参与读取 | 0.1 |
| tera_noaux | TERA三路归属加权记忆 | 0 |
| tera | 完整TERA | 0.1 |

全部使用同一direct训练completion、原始刑期、输入模板与原生EOS监督，输出不增加证据字段。TERA在最终隐藏层和lm_head之间加入残差；记忆/目标条件严格来自prompt，首个允许修改的位置为P−1。弱标签只进入辅助loss，推理接口没有教师证据、参考理由或刑期字段。

Qwen2.5-7B维度3584/rank128：TERA实际新增1,957,640参数，generic新增2,023,176，相差3.35%，满足预定±5%；参数相近不表示FLOPs相等。记录总参数、训练参数、峰值显存、时间和生成token。主干LoRA保持q/k/v/o、rank64/alpha128/dropout0.05。所有新组用逐案例平均completion CE，线性lr衰减、无warmup/weight decay、最终checkpoint；不要把该自定义训练器与旧TRL的token加权/采样差异当作模块收益，因此新增同后端direct对照，而不是直接拿GPU 0分数相减归因。

推理为Transformers单请求贪心、最多2048新token，EOS与im_end停止；请求记忆显式传递，不存在全局last_memory。第一版拒绝batch>1而不是悄悄共享缓存。vLLM插件尚未实现，不能把memory.pt当普通LoRA加载。真实bf16缓存检查要求最大绝对logit差<=0.25且贪心token相同；保存重载top8 ID相同且logit差<=0.01。超限直接回传诊断，不自行放宽以获得分数。

## 返回什么

终端最后会显示`Return experiment evidence: .../results.zip`。无论完成或失败，将该ZIP复制到本机`D:\workspace\codes\COLING\EGC\runs\`，建议命名`tera_qwen25_7b_seed42_results.zip`，然后告知文件名。

包内有execution/completion/failure记录、各组run_manifest/completion、token_budget（稀疏标签覆盖及分罪名统计）、预测/指纹、dev指标和子进程日志。包不包含训练原文、教师缓存或模型权重；adapter和memory.pt留在服务器各组artifact目录。TensorBoard文件留在对应训练组目录。若强制结束导致没有ZIP，可执行：

```bash
python -m egc.learned_server collect --run-dir runs/tera_qwen25_7b_seed42_实际时间戳_实际PID
```

收到结果后先检查数据/模型身份、梯度、重载/缓存、标签覆盖、结束率和月份分布，再比较TERA对generic/direct/tera_noaux。60例训练语义审阅、主体机制评估、三seed稳定性及同协议原论文复现仍未完成；当前自动四组只是有界先导，3%相对MAE门槛不是发表保证。正式LAIC/PCCD/CAIL测试继续冻结。

## 本次软件验证范围

149项离线检查通过，其中7项新增检查覆盖字符与JSON转义对齐、稀疏冲突标签、答案隔离、零残差、梯度、缓存/无缓存、保存重载、真实PEFT层、交错请求、Trainer累积梯度及调度失败门槛。模型测试只使用随机初始化的微型Qwen2和合成字符串，不下载或运行预训练模型，不处理真实数据或调用API。当地检查环境torch2.6.0 CPU/transformers5.3.0，隔离目录补充accelerate1.15.0/peft0.20.0；用户服务器torch2.13/transformers5.17的真实GPU兼容性仍以小试为准。

Trainer接口依据[Transformers官方说明](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)，LoRA使用[PEFT官方接口](https://huggingface.co/docs/peft/en/developer_guides/lora)。原设计的vLLM适配范围留在模块研究文档，不属于本次已实现内容。
