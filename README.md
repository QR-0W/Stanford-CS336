# Stanford CS336: Language Modeling from Scratch

Stanford CS336 课程的个人实现与学习笔记。从零构建大语言模型全流程：
BPE Tokenizer → Transformer 架构 → 分布式训练 → Scaling Laws → 数据工程。

[官方课程仓库](https://github.com/stanford-cs336)

## 硬件配置

| 组件 | 配置 |
|---|---|
| CPU | AMD Ryzen Threadripper 9960X (24核/48线程, 5.49GHz) |
| 内存 | 251 GB DDR5 |
| GPU | 3× NVIDIA GeForce RTX 5090 (32GB × 3) |
| 存储 | 938GB NVMe + 1.9TB HDD |
| CUDA | 13.0 |

## 作业概览

| # | 作业 | 状态 | 笔记 |
|---|---|---|---|
| 1 | [Basics](./assignment1-basics/) — BPE Tokenizer, Transformer, 训练流程 | ✅ | [notes](./docs/assignment1-notes.md) |
| 2 | [Systems](./assignment2-systems/) — DDP, Flash Attention, Sharded Optimizer | ✅ | [notes](./docs/assignment2-notes.md) |
| 3 | [Scaling](./assignment3-scaling/) — IsoFLOPs, Chinchilla, Power-Law Fitting | ✅ | [notes](./docs/assignment3-notes.md) |
| 4 | [Data](./assignment4-data/) — WET 过滤, 去重, PII, 质量分类, 模型训练 | ✅ | [notes](./docs/assignment4-notes.md) |
| 5 | Alignment — SFT, DPO, GRPO | ⏳ | — |

### Assignment 1: Basics

从零实现 GPT-2 风格的语言模型，包含完整训练和文本生成。

| 模块 | 文件 |
|---|---|
| BPE 分词器 | `cs336_basics/tokenizer.py` |
| Transformer (RMSNorm, RoPE, Attention, SwiGLU) | `cs336_basics/transformer.py` |
| 优化器 (SGD, AdamW, Cosine LR) | `cs336_basics/optimizer.py` |
| 文本生成 (temperature, top-p) | `cs336_basics/decoding.py` |
| 训练脚本 | `cs336_basics/train.py` |

### Assignment 2: Systems

分布式训练系统优化：从 naive DDP 到 bucketed overlap，手写 Flash Attention 2 和 Sharded Optimizer。

| 模块 | 文件 |
|---|---|
| Bucketed Overlap DDP | `cs336_systems/ddp_bucketed.py` |
| Flash Attention 2 (PyTorch + Triton) | `cs336_systems/flash_attention.py` |
| Sharded Optimizer (ZeRO Stage 1) | `cs336_systems/sharded_optimizer.py` |
| 四维并行 memory/communication accounting | `cs336_systems/optimizer_state_sharding_accounting.py` |
| 分布式通信基准 (NCCL vs Gloo) | `cs336_systems/distributed_communication_single_node.py` |

### Assignment 3: Scaling

IsoFLOPs 曲线分析与 Chinchilla 计算最优配置外推。基于本地合成 API 的缩放定律实验。

| 模块 | 文件 |
|---|---|
| IsoFLOPs 分析与 Power-Law 拟合 | `cs336_scaling/isoflops.py` |
| Scaling API 客户端 | `cs336_scaling/api.py` |
| 本地合成训练 API | `cs336_scaling/local_api.py` |
| 实验查询计划构建 | `cs336_scaling/scaling_plan.py` |
| Chinchilla 最优分析 | `scripts/chinchilla_isoflops.py` |

### Assignment 4: Data

Common Crawl 数据清洗流水线：从原始 WET 到 GPT-2 训练二进制。

| 模块 | 文件 |
|---|---|
| HTML 文本抽取 | `cs336_data/extraction.py` |
| 语言识别 (fastText LID) | `cs336_data/language.py` |
| PII 脱敏 (邮箱/电话/IP) | `cs336_data/pii.py` |
| NSFW / Toxic 分类 | `cs336_data/harmful.py` |
| Gopher 质量规则 + 质量分类器 | `cs336_data/quality.py` |
| 精确 + MinHash/LSH 去重 | `cs336_data/deduplication.py` |
| 端到端 WET 过滤流水线 | `scripts/filter_data.py` |
| GPT-2 Tokenization | `scripts/tokenize_data.py` |

## 许可证

仅用于教育学习目的。请遵守 Stanford 的学术诚信政策。
