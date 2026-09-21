# B0：本机打包，服务器冻结模型推理

状态：专用打包、服务器编排、结果打包及本机评估入口已实现，112项离线测试通过。真实18条dev任务包尚未生成，36个模型生成尚未运行；本轮助手只使用虚构数据做软件检查，没有在本机启动数据作业/API/模型。Git Bash在开发机不可用，shell包装脚本尚未执行；Python编排已用虚构子进程端到端测试。

## 本次测什么

使用v6既有两罪名dev：诈骗10＋抢劫8，共18案，固定Qwen3-4B。两组各18个任务，原输入与显式目标人物输入，共36次生成。两组共享同一系统指令、案情、罪名和解码参数，仅目标字段有区别。目标来自官方原记录`meta.criminals`，不来自参考刑期、法条或教师推测。

协议为`configs/b0_protocol.json`：上下文16384、输出上限1024、batch2、TP1、temperature0、seed42、显存利用率0.85；复用已经跑通的关闭thinking渲染方式。只推理，不加载adapter、不训练、教师API请求0。不截断或静默跳过超长案例；报错后检查日志，再决定是否更改两个组共同的新协议。

目标可用定义在看输出前固定为：源meta只有一个非空姓名，且姓名能在案情中逐字找到。它不是已证明事件归属正确，短姓名/匿名化仍可能歧义。不可用时target组使用与original组完全相同的输入，保留两个任务及不可用掩码；不猜名字、不删案。主表报告全部18案，可用子集另报。共同系统指令比历史base多了目标字段说明，因此B0 original不是历史base运行的原样重放。

本实验衡量基础模型误差和额外对象信息的影响，不能验证H1/H2，也不是新方法或benchmark提升。18条dev用于探索；不读取LAIC/PCCD/CAIL正式test来调参。

## 1. 用户在本机生成私有数据包

PowerShell：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc.b0 prepare --dev data/cail_reviewed_v6/candidate_dev.jsonl --source-zip data/raw/CAIL2018_ALL_DATA.zip --output-dir data/b0_v1
```

已存在的输入为v6 dev候选和官方CAIL zip。程序只选既有dev中的两罪名18条，读取其官方train成员来源行核验案情，**不重新划分数据**。输入整体digest和112行总数、罪名数量必须匹配冻结协议。输出目录必须不存在。

输出到`D:\workspace\codes\COLING\EGC\data\b0_v1\`：

- `b0_jobs.zip`：需要传给服务器的包，只包含两组任务、协议清单、可用掩码及校验和；没有参考刑期/opinion/法条标签，也不包含官方原始zip。
- `references.local.jsonl`、`source_provenance.local.json`、`local_manifest.json`：留在本机，用于后续评估与追溯，不随服务器包传输。
- 两组`*.jobs.jsonl`及清单：本机副本，用于对照返回结果。

案情原文本身仍可能含说理线索；字段隔离不等于已解决全部输入污染。所有这些私有数据文件均不提交GitHub。

终端应报告cases=18、generations=36、实际target_available及压缩包SHA256。若不符合或报错，回传终端输出，不改协议来绕过检查。

## 2. 私下传输一个文件

用你惯用的传输方式，将：

`D:\workspace\codes\COLING\EGC\data\b0_v1\b0_jobs.zip`

传到服务器：

`/mnt/yanghui/EGC/b0_jobs.zip`

不需要重新传模型或官方CAIL大压缩包。服务器代码由下一步Git拉取。

## 3. 用户在服务器运行

激活你已经验证通过的vllm环境后：

```bash
cd /mnt/yanghui/EGC
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash scripts/server_b0.sh /mnt/yanghui/EGC/b0_jobs.zip
```

模型固定为`/mnt/yanghui/models/Qwen/Qwen3-4B`，占用一张A100；无需PEFT。两个独立推理进程依次运行，第二组开始前第一组释放显存；因此会加载模型两次，日志中的wall time包含加载时间，不能作为方法速度对比。每案保留输入/输出token数、停止原因、生成原文、prompt指纹和模型运行清单。

脚本新建`/mnt/yanghui/EGC/runs/b0_<时间>_<进程号>/`，并打印回传路径。它先校验私有包，使用文件名白名单读取，无路径解压风险。模型推理失败会停止，保留已产生结果；不自动改参数、补推理或重试。不要因输出JSON格式失败而自行重跑，失败也是诊断结果。

无需shell包装的等价入口（选择其一，不能两者都跑）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m egc.b0 run --archive /mnt/yanghui/EGC/b0_jobs.zip --run-dir runs/b0_manual_v1
```

该入口目录也必须全新；它不自动将终端复制到run.log，故默认推荐上方bash脚本。不同命令不是不同实验方案。

## 4. 回传结果

正常结束回传`runs/b0_<时间>_<进程号>/b0_results.zip`，并保留旁边的`runs/b0_<时间>_<进程号>.log`。结果包含两组完整生成、各自推理manifest、输入协议、可用掩码、执行信息及格式完成摘要。`generation_complete=true`只代表36条生成记录齐全，不保证JSON有效、法律准确或预测良好。

若失败，回传终端堆栈/旁边的`.log`及已生成的`execution.json`；不要删除失败目录。助手无法直接读取服务器文件，需把结果传回本机或贴日志。

建议将正常结果包放回本机：

`D:\workspace\codes\COLING\EGC\data\b0_v1\b0_results.zip`

然后由用户在本机运行评估：

```powershell
Set-Location D:\workspace\codes\COLING\EGC
python -m egc.b0 evaluate --prepared-dir data/b0_v1 --results-zip data/b0_v1/b0_results.zip --output data/b0_v1/metrics.json
```

评估再次核验两组任务指纹、模型/权重清单、解码参数及原参考哈希；缺少生成记录会报错，格式无效或length截断会记失败，不算0月。输出`metrics.json`包含全18案、预先规定的目标可用子集和逐罪名的覆盖率、MAE/RMSE及逐案误差。任一组覆盖不全时主表完整集MAE和差值为null，保留valid-only结果作诊断，不能据此宣称整体提升。无独立理由金标准，不计算理由正确率。

完成后告诉助手；保留本机`b0_results.zip`和`metrics.json`供读取。助手将核验结果、审阅真实错误并更新科研主记录。首次B0不看分数自动启动SFT，也不以小样本均值差作为创新成功。
