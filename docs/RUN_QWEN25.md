# 首轮改用Qwen2.5-7B

**2026-09-24更新：** 用户先完成旧命令的Qwen3-1.7B全参实验，隔离包已实际生成并核验4757/528；[结果](QWEN17_DEV_RESULT.md)显示flat先导信号、bound无增益证据。下方Qwen2.5命令作为下一轮同数据复核，直接复用服务器已有包，不再运行本机隔离。使用不同run目录，勿覆盖1.7B结果。

2026-09-23，按用户要求，将learned_server首轮默认模型改为：

`/mnt/yanghui/models/Qwen/Qwen2.5-7B`

原论文LegalChainReasoner §3.1列有Qwen-2.5-7B；此前对PDF第5页的核对记录见RESEARCH_PLAN.md。不能仅据服务器目录名认定它与论文的精确checkpoint/revision一致，也不能把Base称为Instruct。

## 运行

沿用同一份隔离后数据包，不重新调用Flash。尚未生成新包时，先按LEARNED_V2_RESULT.md在本机执行`python -m egc.learned_screen`，再将data/learned_v2_screened/experiment.zip私传服务器同名目录。

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server run \
  --model /mnt/yanghui/models/Qwen/Qwen2.5-7B \
  --archive data/learned_v2_screened/experiment.zip \
  --output runs/learned_v2_screened_qwen25_7b_seed42
```

Qwen2.5-7B预期按实际参数数目走LoRA；代码仍执行用户的<7B全参、>=7B LoRA规则。默认rank64、alpha128、dropout0.05、q/k/v/o，3epochs、lr1e-5、batch1×累积16、max8192、seed42，最终epoch导出。frozen/direct/flat/bound都用同一本地基座，先只评dev。输出为该目录results.zip；不要对旧Qwen3任务使用resume或混合比较分数。

## 格式适配和可复现信息

2026-09-23查阅[官方Base tokenizer配置](https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer_config.json)及[官方Instruct配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json)：二者当前均提供chat_template；Base的eos_token为endoftext，Instruct为im_end。服务器本地文件未读取，不能由线上版本替代本地核验。

新configure_chat在预检、训练、推理统一执行：优先使用checkpoint自带模板；若qwen2缺模板，核对im_start/im_end为既有单token 151644/151645后使用简单文本ChatML。它仅规定序列格式，不改变模型权重或把Base转换成Instruct。Qwen2推理显式追加im_end停止token，并保留引擎原EOS行为，避免Base仅等endoftext而继续输出下一轮对话。其他已有模板的模型保留原序列和停止策略。

template hash、模板来源及stop_token_ids记录在token_budget.json、训练manifest及推理settings中；训练身份包含模板协议，推理fingerprint包含停止设置。分词协议升级，需要新run目录。缺不受支持的模板或Qwen2特殊token不匹配时清晰报错，不下载模型、不修改服务器源模型目录。

138项离线测试通过，包括原全参/LoRA路径、新默认基座、原生模板保留、无模板回退及错误token拒绝、模拟vLLM停止配置/指纹。没有运行真实服务器GPU，也没有在本机加载模型；真实tokenizer和vLLM兼容性需用户首轮预检与运行确认。

使用论文出现过的模型能减少基座差异，但当前CAIL训练集、数据选择、监督设计、LoRA dropout仍不同；原论文dropout0.2，本轮0.05。原ChainAware编码器未复现，仍不能将分数直接与论文表格相减作方法增益。
