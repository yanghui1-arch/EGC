# EGC: Evidence-Grounded Chain Composition

**科研主记录：**[创新点、实验计划与进度](docs/RESEARCH_PLAN.md)。每次项目操作前查看，发生变化后更新；具体约定见[AGENTS.md](AGENTS.md)。历史实验报告保留原始结果。

**当前阶段：**[E4当前规则链方案已停止扩标/训练准备](docs/E4_INVENTORY_RESULT.md)。[B0本机打包→服务器推理→本机评估](docs/RUN_B0.md)已实现，112项离线测试通过。下一步用户在本机打包，再把私有包传到服务器、Git拉取代码，固定Qwen3-4B跑既有18条dev的两组共36次生成；零教师API、不训练。真实包/推理尚未运行，无benchmark提升。

面向 LegalChainReasoner 的研究工程：把案情证据映射到法律条件，区分 supported / refuted / unknown，再组合基础规则、修正规则和例外，研究能否改善裁判理由生成与刑期预测。

当前是**提示层面的机制验证版本**，包含本机数据管线、DeepSeek 辅助标注、服务器全参/LoRA SFT、vLLM 推理和评估入口。训练规则：实际总参数量小于7B使用全参，7B及以上使用LoRA。Qwen3-4B全参和Qwen3-8B LoRA已在用户服务器完成虚构小样例的训练、保存、加载和推理；真实法律数据上的训练和benchmark提升尚未验证。尚未实现原论文神经链编码器和后续偏好训练。

## 现在先做什么

1. 已确认服务器两张A100 80GB，并跑通4B全参、8B LoRA小样例，无需重复安装环境。
2. 改用CAIL官方训练数据，已建立12罪名1200条候选池（1080 train / 120 dev），不再依赖LAIC原生训练集。
3. 已完成修订规范后的200条Flash试标：197条结构合格、3条拦截；随机审阅30条仍发现漏标和反证错误。完整候选池另隔离36条输入，现保留1047 train / 117 dev。当前不直接扩大联合标注提示，先拆分事实抽取与分析生成，见[200条审计报告](docs/PILOT_200_REPORT.md)。
4. E3、全池审计及v6快照验收已完成；暂选flat，进入两罪名E4部分规则与可观测性准备，再用同一模型比较 base、rules、concat、egc。其余罪名待审项不阻止规则准备，但全池不能称为干净训练集。所有规则和超参数在 dev 上确定后再跑 test。

数据来源、实际试标结果与本机命令见 [CAIL与Flash标注](docs/DATA_AND_FLASH.md)。完整实验设计见 [研究计划](docs/RESEARCH_PLAN.md)，已发现的数据与复现问题见 [上游审计](docs/UPSTREAM_AUDIT.md)。

## 服务器第一步

当前用户已回传环境：2×A100 80GB，已有Qwen3-4B和Qwen3-8B；PEFT由用户自行安装。更新代码后，先运行四步Qwen3-4B全参软件小样例：

```bash
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash scripts/server_smoke.sh 4b
```

它使用仓库中的虚构样例；训练后自动加载完整checkpoint推理。换 `8b` 则验证Qwen3-8B LoRA路径。两条路径已在本项目用户服务器跑通，以下命令供新环境复现。

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

当前训练路线请使用 [build-cail / distill](docs/DATA_AND_FLASH.md)。以下仅保留将来拿到作者格式训练文件时的备选入口：`filename / justice / caseCause / opinion / judge`，其中 `judge` 必须是整数月，绝不猜测字符串刑期。

```powershell
python -m egc normalize --input data/raw/laic_train.json --output data/processed/laic_train_all.jsonl --dataset laic --split train
python -m egc split --input data/processed/laic_train_all.jsonl --output-dir data/splits --seed 42
python -m egc overlap --inputs data/splits/train.jsonl data/splits/dev.jsonl data/processed/laic_test.jsonl data/processed/pccd_test.jsonl data/processed/cail_test.jsonl --output data/processed/overlap.json
python -m egc pilot --input data/splits/dev.jsonl --output data/review/pilot.jsonl --n 200
```

不要把 test 改名充当 train/dev。精确重复和同源案件检查不能代替近重复人工审查。

## 数据与凭据

Git 只管理代码、配置、文档和虚构测试样例。`data/`、`outputs/`、`runs/`、权重、`.env` 均忽略。使用 `python scripts/pack_server.py` 将已经准备好的 base 推理任务打包到本机 `outputs/base_jobs.zip`，手动传到服务器仓库根目录解压；包内只有输入任务、manifest 和校验清单，不包含评估标签。预测文件传回本机再评估。

DeepSeek用于条件标注和合成分析，密钥通过环境变量 `DEEPSEEK_API_KEY` 或 `distill --ask-key` 隐藏输入读取，模型为 `deepseek-flash`。`.env.example` 仅展示变量名，程序不会自动加载 `.env`。软件测试使用mock；另已完成真实API试标，数据和响应只在本机忽略目录。规则审核见 [规则与标注](docs/RULES.md)。
