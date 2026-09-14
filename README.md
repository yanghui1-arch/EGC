# EGC: Evidence-Grounded Chain Composition

面向 LegalChainReasoner 的研究工程：把案情证据映射到法律条件，区分 supported / refuted / unknown，再组合基础规则、修正规则和例外，研究能否改善裁判理由生成与刑期预测。

当前是**提示层面的机制验证版本**，包含本机数据管线、DeepSeek 辅助标注、服务器全参/LoRA SFT、vLLM 推理和评估入口。训练规则：实际总参数量小于7B使用全参，7B及以上使用LoRA。尚未运行真实模型实验，尚未证明 benchmark 提升；尚未实现原论文神经链编码器和后续偏好训练。JurisMA 仅作为结构化事实和检查思路的参考，不作为本方案的核心依赖。

## 现在先做什么

1. 在服务器运行下面的 `doctor`，确认模型的完整目录、GPU 显存及 Python 包版本。
2. 补齐 LAIC 训练数据；从训练集按案件分组划出独立 dev，检查与三套 test 的重叠。
3. 从 train/dev 抽取 200 条做人工证据审计；先完成诈骗、抢劫规则及相关修正规则的审核，用小规模实验验证机制。
4. 同一模型比较 base、rules、concat、egc，再决定是否投入可训练链编码器。所有规则和超参数在 dev 上确定后再跑 test。

完整实验设计见 [研究计划](docs/RESEARCH_PLAN.md)，已发现的数据与复现问题见 [上游审计](docs/UPSTREAM_AUDIT.md)。

## 服务器第一步

当前用户已回传环境：2×A100 80GB，已有Qwen3-4B和Qwen3-8B；PEFT由用户自行安装。更新代码后，先运行四步Qwen3-4B全参软件小样例：

```bash
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash scripts/server_smoke.sh 4b
```

它使用仓库中的虚构样例，无需LAIC训练集；训练后自动加载完整checkpoint推理。换 `8b` 则验证Qwen3-8B LoRA路径。实际GPU兼容性尚待该步骤验证。完整环境采集命令保留如下。

首次拉取：

```bash
git clone https://github.com/yanghui1-arch/EGC.git
cd EGC
python -m egc doctor --gpu --output runs/doctor.json
```

已有 checkout 则在 EGC 目录 `git pull --ff-only`，再运行 `doctor`。请回传 `runs/doctor.json`；它只收集模型配置摘要、包版本和 GPU 信息，不读取模型权重或 API Key。不需要先安装或升级整套环境。

所有 `train` / `infer` 命令只在服务器执行。模型必须传 `/mnt/yanghui/models/Qwen` 下含 `config.json` 的**具体目录**，不能直接填父目录。详细命令见 [服务器操作](docs/SERVER.md)。

## 本机数据处理

在仓库根目录使用 Python 3.10+；预处理和单元测试只用标准库：

```powershell
python -m unittest discover -s tests -v
python scripts/smoke.py
python -m egc fetch-upstream
python -m egc normalize --input data/raw/upstream/data/LAIC/test_data.json --output data/processed/laic_test.jsonl --dataset laic --split test
python -m egc normalize --input data/raw/upstream/data/PCCD/test_data.json --output data/processed/pccd_test.jsonl --dataset pccd --split test
python -m egc normalize --input data/raw/upstream/data/CAIL/test_data.json --output data/processed/cail_test.jsonl --dataset cail --split test
python -m egc prepare --input data/processed/laic_test.jsonl --variant base --output-dir data/prepared/base/laic_test
```

`fetch-upstream` 固定作者仓库 revision 并校验 Git blob 哈希。上游内容只下载到忽略目录；本仓库不重新发布其数据或代码。获取失败可单独传输对应原文件。`prepare` 的输出目录必须为空，避免实验混用。

拿到训练文件后，按其真实字段适配再 normalize；当前适配器支持作者公开格式 `filename / justice / caseCause / opinion / judge`，其中 `judge` 必须是整数月，绝不猜测字符串刑期。

```powershell
python -m egc normalize --input data/raw/laic_train.json --output data/processed/laic_train_all.jsonl --dataset laic --split train
python -m egc split --input data/processed/laic_train_all.jsonl --output-dir data/splits --seed 42
python -m egc overlap --inputs data/splits/train.jsonl data/splits/dev.jsonl data/processed/laic_test.jsonl data/processed/pccd_test.jsonl data/processed/cail_test.jsonl --output data/processed/overlap.json
python -m egc pilot --input data/splits/dev.jsonl --output data/review/pilot.jsonl --n 200
```

不要把 test 改名充当 train/dev。精确重复和同源案件检查不能代替近重复人工审查。

## 数据与凭据

Git 只管理代码、配置、文档和虚构测试样例。`data/`、`outputs/`、`runs/`、权重、`.env` 均忽略。使用 `python scripts/pack_server.py` 将已经准备好的 base 推理任务打包到本机 `outputs/base_jobs.zip`，手动传到服务器仓库根目录解压；包内只有输入任务、manifest 和校验清单，不包含评估标签。预测文件传回本机再评估。

DeepSeek 只做可选的条件标注，密钥通过环境变量 `DEEPSEEK_API_KEY` 读取，模型 ID 通过 `--model` 明确指定。`.env.example` 仅展示变量名，程序不会自动加载 `.env`。标注流程和规则审核见 [规则与标注](docs/RULES.md)。当前软件测试使用 mock，未发起真实 API 请求。
