# Learned Evidence v1：先把训练对照跑起来

**2026-09-23运行状态：先恢复已有失败，暂不扩标或训练。** 用户要求模型缺字段等问题重试，不用默认值补齐。已实现有界重试与本机密钥显示，126项离线测试通过。先停止旧进程，然后在本机运行：

```powershell
cd D:\workspace\codes\COLING\EGC
python -m egc.learned annotate --retry-failed --failed-only --max-attempts 3 --limit 200 --workers 1 --ask-key --show-key
```

隐藏粘贴密钥后，程序将完整密钥直接显示在本机交互控制台，核对后输入y再发请求。不会通过程序stdout/stderr输出或写入缓存；输出重定向时拒绝show-key。终端画面会显示密钥，不要把该画面贴回聊天。401/403停止派发，与JSON/引用失败分别记录。

每案最多3次总尝试（旧缓存初次调用计一次），本次最多200个新HTTP请求；成功keep/review跳过，不发送新案例。每次缺字段/JSON/原文校验失败，都把问题反馈给模型并重生成完整JSON。**没有代码补默认值，也不将无效主体自动改成null/unknown。** 到达尝试上限仍失败则隔离；超出预算的未完项可在后续显式续跑。保留`annotations/attempts/<请求指纹>/`中的原响应及重试请求，`000.json`为旧失败记录；不删除旧缓存。若进程中断导致某次发送结果不明，会标记deferred，不擅自重新收费。

结果仍写`data/learned_v1/annotations/summary.json`。完成后告诉助手检查该文件和attempts；真实重试尚未执行。原诊断见[此文档](LEARNED_FIRST_RUN_DIAGNOSIS.md)，下面大批量入口暂作后续说明。

2026-09-22。代码已实现，120项离线测试通过，包括合成源zip、模拟API缓存/预算、三组数据/标签核对、模拟完整服务器编排与续跑、失败打包和指标计算；尚未运行这轮真实抽样、Flash或GPU实验。研究结论仍以 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 为准。

## 原论文实际用了什么

本地 `2026.acl-long.1093.pdf` 第5页§3.1 Implementation、第6页表1/2：LegalChainReasoner 分别搭载 Llama-3.2-3B、Qwen-2.5-7B、DeepSeek-R1-Distill-Qwen-7B、Lawyer-Llama-13B-V2、DeepSeek-R1-Distill-Qwen-32B。均为LoRA：rank64、alpha128、dropout0.2，q/k/v/o投影，bf16，学习率约1e-5，A100 80GB。不是只用了一个7B模型。

Qwen-2.5-7B + LegalChainReasoner 在论文 LAIC 的 MAE/RMSE 为15.61/26.72月，在PCCD为32.39/47.34月。论文在LAIC训练，CAIL仅作未微调测试；本项目拟在CAIL训练，模型也不同，不能直接把表格作差当作方法增益。

当前作者公开版本缺完整训练集和全12罪名链，见[上游审计](UPSTREAM_AUDIT.md)。本轮是原创的证据蒸馏对照管线，没有实现原论文的神经ChainAwareEncoding，因此不是原文严格复现。

## 新候选方法和实验边界

假设H4：让学生模型学习“原文证据→主体归属→事实依据→刑期”，比普通SFT或无主体字段的证据生成更少误归、且有更小刑期误差。训练标签中的证据和摘要来自教师，刑期只取CAIL原始标签。推理时学生自己生成所有中间内容，既不调用教师也不读取参考证据。

| 组 | 监督输出 | 作用 |
|---|---|---|
| train_median | 按给定罪名统计的train刑期中位数 | 零训练统计基线，非关键词判罪 |
| frozen | 事实依据＋刑期 | 同模型未微调基线 |
| direct | 教师事实摘要＋真实刑期 | 常规SFT对照 |
| flat | 原文证据列表＋相同摘要＋真实刑期 | 证据蒸馏 |
| bound | 同一批原文证据及主体/归属/事件类别＋相同摘要＋真实刑期 | 待检验候选 |

三组SFT保留完全相同样本、输入、教师摘要、金标准月份、初始化模型、epochs和优化器设置。flat直接从bound同一次标注中移除元字段；没有单独给bound更多API调用。不把其他罪名映射或关键词分支用于判断罪名、量刑情节或月份。已知罪名是原任务输入，合法性格式/原文引用核验不是法律推断规则。

**本轮不能单独证实结构创新。** bound增加字段及监督token，尚未做同长度/同监督量控制；标准SFT本身和使用更强教师也不是充分创新。必须记录token预算，在有信号后补预算控制、至少3个种子和盲审，再决定是否值得写论文。此前E3绑定抽取未证明优于flat，保留该负面结果；H4检验的是学生训练效果，不是宣称E3已成功。

## 训练数据及清洗

- 来源：[官方CAIL2018](https://github.com/china-ai-law-challenge/CAIL2018)，现有`data/raw/CAIL2018_ALL_DATA.zip`，SHA256固定核对；只读取`final_all_data/exercise_contest/data_train.json`。
- 12罪名每类最多500，共最多6000；从train中按固定seed42分train/dev约9:1。实际样本数可能不足上限，按manifest报告。
- 只取原始单罪名、单目标meta人物、正整数有限刑期；事实最多2400字符，完整保留，不截断。标签冲突的重复原文整组排除。
- 排除LAIC/PCCD/CAIL三个冻结测试输入，以及既有v6 train/dev/隔离样本的精确及5-gram包含率≥0.85近重复；同轮样本亦去重。这个词面检查不能证明不存在语义/案件重叠。
- Flash仅看到facts、给定charge、target_person，不见刑期、裁判理由、法条标签；判定keep/review并提供原文证据。输入污染/主体范围可疑的review、响应失败和无效引用都对三组共同隔离。**不改写事实，不重新标罪名或刑期，不把教师摘要叫作法院理由。**
- 官方当前API名称是`deepseek-flash`，对应DeepSeek-V4.1-Flash（[2026-09-22核对](https://api-docs.deepseek.com/quick_start/pricing/)）。开启JSON，关闭thinking，每调用最多2048输出token，默认并发4，每次最多6000个新请求。保留响应模型版本、usage及缓存。
- 6000是单次运行的总新HTTP调用上限，包含输出纠正重试，不按实验臂重复。每案默认总计3次；每次输出上限2048。网络交付不确定不会自动连发。预算上界为6000×2048输出token，加输入token；实际费用以账户账单为准。
- 教师keep不是真正准确率。`teacher_review.jsonl`提供固定随机60例审查材料；可以完成探索训练后一起复核，不能未经审阅宣称清洗正确或机制成立。

## 本机运行（用户）

在PowerShell执行，沿用本机已能运行EGC的Python：

```powershell
cd D:\workspace\codes\COLING\EGC
powershell -ExecutionPolicy Bypass -File scripts\local_learned.ps1
```

脚本顺序是pool→annotate→prepare；已有pool时复用，API缓存可续跑。运行时隐藏输入密钥，不写文件、不上GitHub。上限6000新请求，默认每罪名500例，不是又一次20条试标。

如需先看请求数，分步运行：

```powershell
python -m egc.learned pool
python -m egc.learned annotate --dry-run
python -m egc.learned annotate --workers 4 --limit 6000 --ask-key
python -m egc.learned prepare
```

两种方式任选，**不要对已经生成的pool重复执行pool命令**。脚本不是自动删除旧产物的工具。若出现`uncertain_request_no_automatic_retry`，先看summary；确定放弃中断请求时执行以下命令，将其记失败而不重发，再prepare：

```powershell
python -m egc.learned annotate --resolve-pending-as-failed --ask-key
python -m egc.learned prepare
```

旧失败缓存只有传入--retry-failed才会重新请求，且受累计尝试上限约束；已收到的无效响应在当次运行会进行有界纠正。连续两批最终全部失败停止派发，401/403立即停止新派发（在途调用可能已发送）。缓存保留安全HTTP分类/状态码。任一划分保留不足一半或整个罪名消失时，prepare拒绝打包。

产物：

- `data/learned_v1/pool/manifest.json`：真实来源/计数/划分指纹。
- `data/learned_v1/annotations/summary.json`及cache：真实调用/筛选状态。
- `data/learned_v1/ready/manifest.json`、`excluded.jsonl`、`teacher_review.jsonl`：共同样本及质量审阅。
- **`data/learned_v1/ready/experiment.zip`**：私下传到服务器`/mnt/yanghui/EGC/data/learned_v1/experiment.zip`。有训练数据，不能放GitHub。

如果本机清洗大部分被拒绝或有源文件校验失败，先把上述manifest/summary位置告诉助手，不继续模型训练。

## 服务器一次跑完整dev实验（用户）

首轮默认现有Qwen3-1.7B，实际参数<7B全参，以降低三次训练资源成本。最大8192 tokens，batch1×累积16，3epochs，lr1e-5，seed42，bf16混合精度/FP32主参数与梯度。每组保留1个训练checkpoint和最终模型；预留约100GB以上磁盘，实际以存储和依赖实现为准。GPU显存不能仅据参数量保证，超长先在token预检停止，不静默截断。完整4B训练内存需求更高，首轮不要随意同时塞两项到一张卡。

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server run \
  --archive data/learned_v1/experiment.zip \
  --output runs/learned_v1_qwen17_seed42
```

脚本串行完成token预检→frozen dev推理→direct训练及dev推理→flat训练及dev推理→bound训练及dev推理→统计基线/MAE/RMSE/覆盖率/分罪名/配对bootstrap→结果包。各组从相同原模型重新初始化，子进程退出释放显存。所有推理显式bf16，避免全参导出dtype影响自动选择精度。固定最终epoch，不用各组不可比的教师序列dev loss选模型，也不看test选checkpoint。

无须额外连接服务器或安装本机GPU环境。`run`只允许Linux；正式训练仍由已有server.train按实际参数自动选择全参或LoRA。如后续运行Qwen3-8B，增加`--model /mnt/yanghui/models/Qwen/Qwen3-8B`，会用LoRA rank64/alpha128/dropout0.05；这与论文dropout0.2不同，不宣称相同配方。`Qwen2.5-7B`若是Base且没有chat_template，预检会停止，需要另定Base序列协议，不能冒充Instruct。

失败会尽量生成`failure.json`、日志和`results.zip`。同一代码commit/模型/预算/输入包续跑：

```bash
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server run \
  --archive data/learned_v1/experiment.zip \
  --output runs/learned_v1_qwen17_seed42 --resume
```

完成的训练跳过，推理按指纹续跑；未完成的训练仅从已有有效checkpoint继续。首个checkpoint之前失败则明确停止，需检查原因，不能假装续跑成功。保留所有原输出。若进程被系统强杀没有结果包，可执行：

```bash
python -m egc.learned_server collect --run-dir runs/learned_v1_qwen17_seed42
```

把`runs/learned_v1_qwen17_seed42/results.zip`（续跑时可能为带时间戳版本，以终端输出为准）下载到本机`data/learned_v1/`，告诉助手即可。包含预测、指标、日志和训练元数据；不含模型权重。

## Benchmark及后续判定

首轮只看新CAIL dev，用于开发诊断。冻结论文test输入：test0=LAIC1200、test1=PCCD100、test2=CAIL120；这轮不会自动推理这些test。CAIL训练后，CAIL测试不能叫跨来源泛化；LAIC/PCCD可检验跨来源迁移，但目标人字段缺失以及输入长度/标签口径差异必须报告。

预定主指标MAE月数，辅指标RMSE、有效输出覆盖率、各罪名误差和配对案件bootstrap；不把失败当0月。原文引用可定位率只测结构忠实性，不能替代主体/法律语义盲审。理由只有诊断性字符ROUGE，原论文ROUGE/BLEU/BERTScore/Legal-QA尚未复现，不声称超越完整CJOG。

bound相对flat的dev MAE至少降3%、覆盖率不降，只作为继续投入门槛；还须看种子方差、长刑期误差和主体误归盲审。若只有SFT胜frozen，不算候选方法创新成立；若bound不胜flat则停掉绑定复杂化。首次单seed无论结果高低均不能宣布论文成功。

冻结最终配置/预算控制和至少3个种子后，已实现的测试入口为：

```bash
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server test --run-dir runs/learned_v1_qwen17_seed42
```

**当前先不要运行test命令。** 后续记录冻结决定后对各seed一次测试；需要原论文编码器同数据/同基座重跑才能作方法归因。这项复现、同token消融、盲审及官方生成指标仍待完成，不能将当前探索管线写成已完成的论文实验。
