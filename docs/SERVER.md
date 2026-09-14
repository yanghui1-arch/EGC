# 服务器操作

本机只做数据处理。下面模型相关命令均由用户在服务器执行；开发端没有连接服务器或运行模型。先从仓库根目录运行：

```bash
python -m egc doctor --gpu --output runs/doctor.json
```

已收到环境：Python3.12.14、torch2.13.0、transformers5.17.0、TRL1.13.0、datasets5.0.1、accelerate1.15.0，vLLM为0.28.1预发布构建，GPU为两张A100 80GB。PEFT由用户自行安装；本工程不会自动安装或升级环境。版本信息只能证明包存在，训练/推理组合仍需GPU小样例验证。当前仅适配本地因果语言模型的标准接口，MoE或特殊架构需额外验证。

## 已确定的训练规则

| 模型 | 模式 | 导出目录 | 推理加载方式 |
|---|---|---|---|
| Qwen3-1.7B、Qwen3-4B | 全参 | RUN/model | `--model RUN/model` |
| Qwen3-8B | LoRA | RUN/adapter | `--model 原模型 --adapter RUN/adapter` |

`--training-mode auto` 是默认值，依据加载后模型的实际总参数量（含embedding，不按目录名猜测）选择：小于7,000,000,000全参，达到或超过则LoRA。显式指定 `full` 或 `lora` 可用于核对，若不符合该规则则停止。manifest记录总参数量、可训练参数量及实际模式。全参分支不创建adapter，并检查全部参数均可训练；应用代码仅在LoRA分支导入PEFT。

先使用非Base版Qwen3-4B和Qwen3-8B现有聊天模型。同一模型的四个方法保持训练模式一致。原始Base版可能没有chat template；Qwen3.5及其他ConditionalGeneration架构不是当前入口已验证的目标。FP8等量化权重的训练会明确拒绝，当前没有实现该训练路径。

## 最小软件及模型检查

在仓库根目录更新并运行自动小样例：

```bash
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 bash scripts/server_smoke.sh 4b
```

它创建独立的带时间戳目录，用Qwen3-4B单卡全参训练4步，每2步dev验证，再加载 `model/` 做两条虚构输入的推理。初始长度1024、batch1、梯度累积1；这是兼容性检查，不是正式训练配置。请回传脚本末尾路径中的 `completion.json`、`run_manifest.json` 和 `predictions.jsonl`，若失败则回传报错堆栈。

PEFT安装好后可用 `CUDA_VISIBLE_DEVICES=1 bash scripts/server_smoke.sh 8b` 验证LoRA分支。先顺序跑通，两张卡之后可分别跑独立实验；不要求现在使用分布式或新装DeepSpeed。全参checkpoint包含全部权重和训练状态，实际显存及磁盘消耗需以运行检查为准。

全参分支使用FP32主权重和梯度，由Trainer按 `--precision` 做BF16/FP16混合精度计算；A100默认BF16。这样没有把Adam训练的全部状态压成低精度来节省显存。4B先以1024长度检查，确认峰值显存后再增加正式实验长度。

软件检查不读模型：

```bash
python -m unittest discover -s tests -v
python scripts/smoke.py --output-dir data/smoke
```

下面保留Qwen3-8B LoRA的分步示例；Qwen3-4B全参请使用上面的自动脚本，或将训练输出后的推理 `--model` 改为对应 `RUN/model`：

```bash
MODEL=/mnt/yanghui/models/Qwen/Qwen3-8B
python -m egc infer --model "$MODEL" --jobs data/smoke/base/test/jobs.jsonl --output runs/smoke_base.jsonl --max-model-len 2048 --max-new-tokens 256 --batch-size 1
python -m egc train --model "$MODEL" --train data/smoke/base/train/sft.jsonl --dev data/smoke/base/dev/sft.jsonl --output runs/smoke_sft --max-length 2048 --max-steps 4 --eval-steps 2 --grad-accum 1
python -m egc infer --model "$MODEL" --adapter runs/smoke_sft/adapter --jobs data/smoke/base/test/jobs.jsonl --output runs/smoke_adapter.jsonl --max-model-len 2048 --max-new-tokens 256 --batch-size 1
```

这是虚构数据的软件与GPU兼容性检查，不评估法律任务。命令默认单卡，BF16；是否能装下取决于实际模型和显存。缺BF16支持时可明确用 `--precision fp16`。Qwen思考模板尽量统一关闭 thinking；若服务器TRL接口不支持相同模板参数，程序会提示后停止，需依据doctor调整。

## 跑已经准备好的 base

把本机 `outputs/base_jobs.zip` 传到服务器 EGC 根目录，先解压：

```bash
python -m zipfile -e base_jobs.zip .
python -m egc infer --model "$MODEL" --jobs data/prepared/base/laic_test/jobs.jsonl --output runs/base/laic_test.jsonl
python -m egc infer --model "$MODEL" --jobs data/prepared/base/pccd_test/jobs.jsonl --output runs/base/pccd_test.jsonl
python -m egc infer --model "$MODEL" --jobs data/prepared/base/cail_test/jobs.jsonl --output runs/base/cail_test.jsonl
```

这能建立冻结模型的基础结果，但**不应先查看测试分数再据此选择 EGC 规则和超参数**。可先只跑软件检查，等 dev 流程与方法冻结后统一执行正式 test。

推理默认TP=1、上下文8192、输出1024、temperature=0；实际配置应在dev冻结且四种方法一致。多卡推理使用 `--tensor-parallel N`，N按可见GPU设置，不自动假定卡数。超长输入会报错，不能只对某个方法截断或丢弃样本。

同一参数再次执行会跳过已完成ID；变更模型、输入、权重文件或解码设置需新的输出文件。只记录了权重文件大小/mtime和配置以检查续跑身份，未做大权重内容哈希，不等于完整模型版本证明。

## 正式训练与对照

待训练集、dev划分和规则审核完成，在本机分别 `prepare` 四种变体的 train/dev/test。每个变体使用相同案件集合。将对应的 `sft.jsonl` 和 `jobs.jsonl` 私下传到服务器，不提交Git。训练标签只在train/dev SFT文件中；test任务没有标签。

```bash
python -m egc train --model "$MODEL" --train data/prepared/base/train/sft.jsonl --dev data/prepared/base/dev/sft.jsonl --output runs/base_s42 --seed 42
python -m egc infer --model "$MODEL" --adapter runs/base_s42/adapter --jobs data/prepared/base/test/jobs.jsonl --output runs/base_s42/predictions.jsonl
tensorboard --logdir runs
```

分别替换 base 为 rules、concat、egc，并用42、43、44至少三个训练种子。7B及以上默认LoRA rank16；小于7B全参时rank参数不参与模型构建。batch1、梯度累积16、3epochs、学习率1e-5只是初始设置，不是已调好的配置。全参与LoRA应分别在dev选择学习率，再在各自模型组内保持一致。未启用4bit量化。

checkpoint每100个更新保存和验证；若小数据训练总步数不足100，减小 `--eval-steps`，确保实际发生验证。以dev loss选择checkpoint，不使用test。恢复训练时在完整原命令上增加 `--resume runs/base_s42/checkpoint-N`，配置与数据必须保持一致。报超长时会生成 `overlength.json`，调整经dev确认的预算后使用新输出目录。

## 本机评估

把服务器预测及其 `.manifest.json` 传回本机；保留同一运行目录和模型信息。例：

```powershell
python -m egc evaluate --reference data/processed/laic_test.jsonl --predictions runs/base/laic_test.jsonl --output outputs/base_laic_metrics.json
python -m egc compare --reference data/processed/laic_test.jsonl --baseline runs/concat/laic_test.jsonl --candidate runs/egc/laic_test.jsonl --output outputs/egc_vs_concat.json
```

LAIC内部有重复组，`compare` 的独立案件bootstrap不能直接作为严格显著性证据。主表保留原benchmark全部行，正式区间需要预先固定成组采样实现，或使用固定去重子集并同时报告其样本数；不要为了显著性临时删样本。当前未实现成组bootstrap。
