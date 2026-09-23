# Learned Context v2：本机标注与服务器实验

**2026-09-23用户已完成标注。当前执行[验收报告中的零API隔离及服务器命令](LEARNED_V2_RESULT.md)。不要重跑下方标注步骤，也不要把旧ready包用于本轮主对照。** 原包5057/556核验通过，但抽样发现量刑建议漏筛；新增保守隔离预计4757/528，真实计数以用户执行结果为准。

2026-09-23已实现，132项离线测试通过；真实v2标注及训练尚未运行。

## 本机运行

停止旧版标注进程后，在已更新的本机工程运行：

```powershell
cd D:\workspace\codes\COLING\EGC
powershell -ExecutionPolicy Bypass -File scripts\local_learned.ps1
```

脚本复用`data/learned_v1/pool`的6000候选及原划分；标注写入`data/learned_v2/annotations`，成功完成后生成`data/learned_v2/ready/experiment.zip`。旧v1缓存不动、不自动迁移；旧76案也按新提示重新调用，以保持新轮协议一致。输入仍只有facts/charge/target_person，不给教师真实刑期，真实刑期保留用于SFT。

默认并发4，每次最多6000个新HTTP请求（包括纠正重试），每案累计最多3次，每调用最多2048输出token。密钥隐藏输入；确需本机显示核对时加`-ShowKey`，不要回传含密钥画面。没有程序默认值填充。引用须来自原文；跨句恢复主体、重复出现的引用、略长的引用/摘要不再硬拒绝。keep仅代表结构和来源合格的机器标注。

6000是调用预算，不保证一次完成6000案。如果摘要中`stop_reason=request_budget`，可重跑同一条脚本接续未完成案例；已成功的v2缓存不会重复调用。旧失败默认保留，要恢复v2失败才加`-RetryFailed`，仍不超过每案累计3次。请求连续8案失败、校验连续20案拒绝或401/403会停发，此时先回传summary。中断请求不自动重发，不删除pending或缓存。

若prepare报未完成标注，先看summary的not_yet_scheduled和deferred，不把它当训练问题。已有ready包时脚本不覆盖；若之后改变标注，必须显式使用新output重新prepare，不能把旧包当新结果。

完成后告诉助手，检查以下本机产物即可，不要上传到GitHub：

- `data/learned_v2/annotations/summary.json`：调用数/状态/停止原因。
- `data/learned_v2/ready/manifest.json`：三组共用的真实样本数量及版本。
- `data/learned_v2/ready/teacher_review.jsonl`：固定随机60条待语义复核材料。
- `data/learned_v2/ready/experiment.zip`：私传服务器的数据包。

## 服务器运行

将包私下传到`/mnt/yanghui/EGC/data/learned_v2/experiment.zip`，然后执行：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server run \
  --archive data/learned_v2/experiment.zip \
  --output runs/learned_v2_qwen17_seed42
```

默认`/mnt/yanghui/models/Qwen/Qwen3-1.7B`，全参训练。按同案例、同输入、同摘要/真实刑期比较direct/flat/bound；flat/bound共享同次教师的同一组引用，bound增加上下文主体字段。串行完成token预检、frozen dev、三次SFT及dev预测、统计基线/MAE/RMSE/覆盖率/结果包。实际参数<7B全参，>=7B LoRA。

首轮3epochs、lr1e-5、batch1×累积16、max8192、seed42、最终epoch导出；推理显式bf16。全参使用FP32主参数与梯度及bf16 autocast。预留约100GB以上磁盘，实际取决于运行环境。超长样本预检停止、不静默截断。同一代码/模型/数据/预算的中断任务在上述命令末尾加`--resume`。回传运行目录的`results.zip`，失败时保留failure.json及日志。默认只跑dev，不在测试集选配置。

## 判定

H4仍是“学习证据与上下文主体归属是否改善刑期预测”的假设。主要比较bound对flat，同步报告MAE、RMSE、覆盖率及监督token量。格式通过率不是语义准确率。需要抽样复核、预算控制和至少3seed；不能把普通SFT对frozen的提升或使用不同数据/基座的结果，直接称为超越原论文或可发表。原ChainAware编码器和官方生成指标仍未复现。
