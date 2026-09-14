# 上游资源与复现审计

核对版本：[LegalChainReasoner c66b3d1](https://github.com/Statistical-NLP-Lab/LegalChainReasoner/tree/c66b3d14d2b6e1c9bb6552b774494c64a08e79b3)。文件路径、大小和 Git blob SHA 记录于 `configs/upstream.json`。

## 本机数据核查

| 数据 | 行数 | 缺少理由 | 零月标签 | 精确重复事实组 | 启发式待审查行 |
|---|---:|---:|---:|---:|---:|
| LAIC test | 1200 | 4 | 0 | 2 | 705 |
| PCCD test | 100 | 0 | 0 | 0 | 39 |
| CAIL test | 120 | 120 | 5 | 0 | 0 |

上表来自当前脚本的规范化与审计输出。审查标记基于参考理由关键词未在输入中逐字出现、输入中出现判决措辞等启发式。705不是“705个错误样本”，0也不是“全部没有问题”；同义改写、主体归属、时间和语义都需人工复核。

LAIC 12罪名各100，CAIL各10；PCCD分布不均衡，例如诈骗21、受贿26。CAIL 的“未知”理由被规范化为缺失，不用这个占位文本评测生成理由。5条零月标签暂保留并标记，不擅自解释为缓刑、无罪或删除。三套测试集之间未发现标准化后精确相同事实或同数据源案件标识的重叠；近重复尚未检查。

规范化文件内容摘要（本项目稳定 JSON SHA256，不是上游字节SHA）：

- LAIC: `2ddffea08c8e45aaad7bd02ecb23c2bac4171ddc3abb9d23c1ad83236a62fd00`
- PCCD: `b733a67433a17142536703097e604c38956f0c32b84d95e449d5ba72d0d31db8`
- CAIL: `3352f8cd986c19bc27febc4a04a8b9150ea0e62eec551ca3fc3087f6816e4c0b`

## 资源缺口

该公开 revision 提供三套 test 和诈骗/抢劫的链，未提供论文使用的完整训练集和全部12罪名链。诈骗链9条、抢劫链32条。没有在所检查树中发现 LICENSE 文件；本仓库不复制发布上游源码和数据，仅提供固定版本的本地下载入口及原创适配代码。

## 作者代码中影响复现的项目

检查对象为固定版本的 [LegalChainReasoner.py](https://github.com/Statistical-NLP-Lab/LegalChainReasoner/blob/c66b3d14d2b6e1c9bb6552b774494c64a08e79b3/LegalChainReasoner.py)。

- 模型初始化包含 Llama 路径及数据路径占位，不能直接等同于用户服务器的 Qwen 实验。
- 可见实现中用测试集 RMSE/ROUGE 选择 checkpoint 和提前停止；本工程必须改用独立 dev，不能沿用测试选择流程。
- 可见处理有案情512字符、理由400字符截断。直接改成保留完整输入会改变实验协议，因此本工程结果应与同输入新跑的基线比较，不能只和论文表格作差。
- 批量路径中部分逻辑取首个罪名，需要在复现时验证异罪名批次；公式与门控实现的对应也需逐项确认。

这些是对公开实现的复现风险观察，不据此推断论文报告实验一定采用相同细节。当前实现不是对上述源码的直接移植。

## 依赖接口

服务器代码参考 [TRL SFT](https://huggingface.co/docs/trl/sft_trainer)、[TRL chat templates](https://huggingface.co/docs/trl/chat_templates)、[vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora.html)；标注接口参考 [DeepSeek Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)。最终以用户服务器 `doctor` 的版本和小规模实际运行验证为准，当前没有锁定或验证GPU依赖组合。
