# Context v2标注验收及输入风险隔离

**模型更新：用户指定首轮Qwen2.5-7B/LoRA，当前服务器命令见[RUN_QWEN25.md](RUN_QWEN25.md)。下方1.7B是此前方案，数据验收事实及本机零API隔离步骤保持有效。**

2026-09-23，用户完成本机脚本后，助手只读核验现有产物；未运行真实API、重新打包或GPU任务。

## 实际产物

6000候选全部有终态：keep 5613（93.55%）、review 381、failed 6。最终一次summary的57请求是续跑：53 keep、4 review，不是全部请求数。逐案attempts_used合计6057；5949案1次、45案2次、6案3次。5994份成功响应metadata模型名为deepseek-flash；这没有额外证明具体后台权重版本。无未完成attempt marker。

6个最终失败均为quote_not_in_source，3次后隔离。无需追求6000全通过，也无需重做已有标注。

原实验包：data/learned_v2/ready/experiment.zip，16193755 bytes，SHA256：

`7067084c792feb4d87809c48c5eedf25761b85f8d3d87eac41bb7211b8f3ea24`

实际保留5057 train / 556 dev、12罪名齐全。只读核验通过：6000缓存ID/输入及请求哈希；keep/review原响应JSON与存储body一致；所有包成员校验和；包内/包外manifest一致；三组案例及顺序、教师摘要、原始月份一致；flat/bound共享原引用；dev及test任务prompt/hash与参考行一致；test参考哈希与manifest一致；60条审阅队列与缓存/输入相符。未重跑来源抽样、去重或API。

## 有界抽样发现

从固定随机60条队列中按出现顺序每罪名取首条，共12条，索引0/1/2/3/4/6/9/13/16/29/33/39。模型辅助语义查看，不是独立人工金标准，不能计算标注准确率或将12条比例推广全体。

- 上下文主语恢复已能通过；引用较长不再自动失败。
- 两条抽样输入含本案数字量刑建议，仍被教师keep：索引4及13。即使边界部分匿名化，剩余年份上界仍提供预测线索。索引13还把建议片段列入证据。这是输入风险，与引用长度限制无关。
- 索引3/6的公司主体被标为target，但输入目标是自然人；索引1/13中查获事件的subject有时是目标人物，索引39的后果则给null/uncertain。语义角色和关联关系边界不一致，不能声称H4主体监督已准确。保留原响应，不程序改标签。
- 摘要还可能保留检方认定及结论性陈述；本次局部隔离不解决所有说理污染。

## 最小后续处理

已实现`python -m egc.learned_screen`：零API，仅对原包train/dev事实按句段查找“判处/量刑/判决如下/缓刑”与数字年月共现。全案保守隔离，对三组执行相同mask；不剪原文、不改罪名/月数/标注、不重划分、不使用测试结果。测试文件保持字节不变。

只读扫描实际原包得到300 train / 28 dev风险候选（合计328，保留集的约5.84%）；**不是328条都被证实泄漏**。部分命中为前科、法条年份或累犯年限；该小轮选择保守隔离并保留明细，避免付费重标和逐条无限修订。不能把这个源数据筛查解释成关键词推罪名或量刑规则，也不能把未命中当成无污染认证。

预计新包4757 train / 528 dev、12罪名仍齐全，实际以用户运行后manifest为准。原包/缓存不覆盖；新包记录来源SHA、策略、隔离清单hash与选择偏差。三组正式比较应全部使用新快照，不与旧包结果混用。

134项离线测试通过，新增测试验证候选不等于语义标签、源包不变、三组整案一致过滤、原标签保留、dev/test不受跨组错配、服务器解包校验可用。真实隔离打包尚未运行，服务器训练仍未开始。H4无benchmark新证据。

## 用户命令

本机只运行一次（无需密钥）：

```powershell
cd D:\workspace\codes\COLING\EGC
python -m egc.learned_screen
```

输入data/learned_v2/ready/experiment.zip；输出data/learned_v2_screened/experiment.zip、manifest.json、screening.json、quarantine.jsonl。已有输出时停止，不覆盖。将新的experiment.zip私传服务器`/mnt/yanghui/EGC/data/learned_v2_screened/experiment.zip`，然后：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 python -m egc.learned_server run \
  --archive data/learned_v2_screened/experiment.zip \
  --output runs/learned_v2_screened_qwen17_seed42
```

首轮Qwen3-1.7B全参，frozen及direct/flat/bound、同原始刑期、同划分，只评dev；其余预算不变，test后置。新目录不使用旧run的resume。用户运行完成后回传该run目录results.zip；失败保留日志/failure.json。本机screening.json用于确认真实计数。此轮是存在已知机器标签噪声的探索性实验；正式论证仍需语义审阅、token控制、多seed及原论文公平基线。
