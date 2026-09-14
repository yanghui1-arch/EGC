# CAIL训练数据与DeepSeek Flash辅助标注

## 当前方案

采用“原始刑期/罪名标签 + 机器证据标注 + 明确标记的合成分析”，不再等待LAIC原生训练集。保留LAIC、PCCD测试集；CAIL训练后，原CAIL测试只能作为同来源补充测试，不能称为跨来源泛化。

同一模型的四个方法必须使用相同的训练案件、同一份目标文本和原始刑期。EGC相对concat的增量才反映组合机制；DeepSeek提供了额外教师能力，需要单独报告。与第二篇论文直接比较分数时存在训练数据不同的混杂因素，应重跑统一数据条件下的基线。

## 数据来源与选择

选择 [CAIL2018官方数据](https://github.com/china-ai-law-challenge/CAIL2018)。其[字段说明](https://github.com/china-ai-law-challenge/CAIL)提供fact、accusation、relevant_articles、term_of_imprisonment，适合当前给定罪名的刑期任务。通用法律问答和法律咨询文本与本任务不够匹配。

还检查了 [CaseGen](https://huggingface.co/datasets/CSHaitao/CaseGen) 和 [CCVG](https://huggingface.co/datasets/TIM0927/CCVG)：前者定位为500案例的生成benchmark，不用于填充本项目训练池；后者数据卡说明原训练集正在等待去标识化，目前只有训练例子与测试数据。因此本轮使用官方可获取且带判决结果标签的CAIL。

官方压缩包约939MiB，下载到本机忽略目录。ZIP中的README确认 `final_all_data/exercise_contest/data_train.json` 为练习赛训练集，含154,592行。没有把官方test/valid转换成训练集。

- 官方URL：`https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip`
- 文件大小：984,551,626字节。
- 本地文件SHA256：`3c05dfdade742f8b0d5e782d174475e7769448a5f407bfb7f14f0aed72d61d4a`。
- 所用成员CRC32：`f014a340`；解压后247,240,777字节。ZIP读取时校验成员CRC。

当前有效池为本机 `data/cail_pilot_v3`：12罪名各100条，1080 train / 120 dev。它是小规模机制验证池，不是CAIL完整训练集。

过滤多罪名/被告不明确、死刑/无期、非正整数月、量刑建议和可能的判决结果文本；事实相同而标签冲突的组不入池。同义罪名不自动猜测，只规范化CAIL原有方括号、分隔符与末尾“罪”。

本次全源筛选统计：34,117条多/缺失罪名；1,288条非正或不合法月数；39条特殊/缺失刑期类型；578条疑似判决/量刑建议文本；去除301条相同事实重复记录、4条标签冲突组中的记录。以上过滤按顺序计数，不表示相互独立的错误率。部分合法既往判决描述也可能被保守过滤。

填充1200条候选池期间，排除5条与现有三套测试集重叠的候选；train/dev词面近重复检查未发现额外重叠。检查用NFKC/去空白后的精确匹配及字符5-gram包含率≥0.85，仍不能排除同案改写和语义重复。

```powershell
python -m egc build-cail --archive data/raw/CAIL2018_ALL_DATA.zip --member final_all_data/exercise_contest/data_train.json --benchmarks data/processed/laic_test.jsonl data/processed/pccd_test.jsonl data/processed/cail_test.jsonl --output-dir data/cail_new_pool --per-charge 100 --seed 42
```

`--per-charge 1000` 可建立更大的候选池，最终数量受合法样本量限制。输出目录必须新建，便于固定实验版本。

## Flash API与标注边界

核对 [DeepSeek官方模型说明](https://api-docs.deepseek.com/quick_start/pricing/)：`deepseek-flash` 当前对应DeepSeek-V4.1-Flash。该名称是可变化的别名，因此保存每次返回的model、response_id、created、system_fingerprint，不宣称它永久固定一个权重版本。

本轮[关闭thinking](https://api-docs.deepseek.com/guides/thinking_mode/)，输出预算4096 tokens。一次请求同时生成八类事实条件和带原文引用的简短分析。API请求只包含案情、给定罪名、标注profile，不包含原始刑期、法条标签或参考理由。原始数值刑期在本机保留，绝不采用教师预测的月份。

`configs/evidence_profile.json` 是事实条件的研究标注规范，不是真实法律规则。主动到案不直接等同自首，前科不直接等同累犯，部分未得逞不直接等同全案未遂。当前合成分析偏向事实归纳，缺少经审核的法条前提，不能冒充原始法院说理或直接作为完整法理论证真值。

建议先20条规范试验，修订后200条，再决定是否扩展到1200或更大。20条中出现的错误要进入抽查清单，而不是依靠模型自称“已审核”。助手的抽查也属于模型辅助审阅，不是法律专家人工金标准。

## 本机运行

从当前训练池抽样，使用新输出路径保留各版本结果：

```powershell
python -m egc pilot --input data/cail_pilot_v3/train.jsonl --output data/cail_pilot_v3/pilot200.jsonl --n 200 --seed 42
python -m egc distill --input data/cail_pilot_v3/pilot200.jsonl --output-dir data/flash_pilot200_v2 --model deepseek-flash --limit 20 --dry-run
python -m egc distill --input data/cail_pilot_v3/pilot200.jsonl --output-dir data/flash_pilot200_v2 --model deepseek-flash --limit 20 --transport curl --ask-key
```

`--ask-key` 为终端隐藏输入，不写入文件。也支持预先设置 `DEEPSEEK_API_KEY`。Windows本次urllib连接曾连续中断，切换系统curl后完成20条；curl凭据通过进程stdin传入，不放在命令行参数或临时文件里。其他环境默认urllib。

每次最多新增20条；缓存已完成JSON响应，结构不合格也保留响应，不自动重付费。网络异常记录为 `.error.json` 并停止当前批次；后续同命令跳过该错误记录。网络结果不明确的请求可能已被计费，本工具不能确认。修改提示/profile/模型时必须新建输出目录。

输出：`accepted.jsonl` 为结构合格草稿（仍未语义审核），`rejected.json` 为被拦截或请求异常项，`raw/` 为响应缓存，`manifest.json` 固定版本和进度。原始刑期、教师身份与合成文本来源均可追溯，prepare进一步记录synthetic_opinions数量。

## 首轮20条实际结果

实验输入来自较早的 `cail_pilot_v2/pilot20.jsonl`，规范为 `fact-circumstances-v1`，提示为 `source-only-synthetic-analysis-v2`，结果在 `data/flash_pilot20_curl`。此轮是规范审计样本，**并非当前v3训练池已经完成标注**。

- 20次请求收到完整JSON响应；19条通过结构与原文跨度校验，1条因引文在原文中多次出现被拦截。
- 19条共152个条件判断：supported 19、refuted 12、unknown 121。这是模型输出分布，不是标注准确率。
- API报告输入23,476 tokens（其中缓存命中2,432）、输出15,334 tokens。按当时峰值价格并假设全部输入未命中，估计约0.0254美元；实际账单、早期探测和连接失败请求不包含在这个估计内。
- 抽查发现：第三方或单位还款被归到个人；“传唤到案”被过度否定；多次行为的部分结果被概括全案；部分陈述遗漏“指控”归属。另发现一条输入含量刑建议。
- 据此已收紧profile到v2，训练池v3新增量刑建议过滤。当前八条件仍是案例级，事件/主体绑定是接下来必须验证的改进。尚未重跑v2规范的200条标注，也未进行真实数据训练。

下一步：以新的规范试标并抽查200条（应加入主动到案、返还主体、多个事件的定向样本），补充可溯源的法律前提与规则，再冻结四组公平对照。不能把“19/20格式通过”报告成95%的法律标注准确率。
