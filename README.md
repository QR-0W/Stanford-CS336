<div align="center">

<img src="https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/pytorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/cuda-13.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
<img src="https://img.shields.io/badge/license-educational-lightgrey?style=for-the-badge" alt="License"/>

</div>

<br>

<div align="center">

# CS336: Language Modeling from Scratch

**Stanford CS336 · 从零构建大语言模型**

</div>

<br>

<div align="center">
  <table><tr><td>
    <strong>BPE Tokenizer</strong> → <strong>Transformer</strong> → <strong>分布式训练</strong> → <strong>Scaling Laws</strong> → <strong>数据工程</strong>
  </td></tr></table>
</div>

<br>

## 作业进度

> **已完成 4/5** &nbsp;&nbsp; ████████████████████░░░░ &nbsp;&nbsp; **80%**

| # | 作业 | 状态 | 说明 |
|:---:|---|---|---|
| 1 | **Basics** | ✅ | BPE Tokenizer · Transformer (RMSNorm, RoPE, SwiGLU) · 完整训练流程 |
| 2 | **Systems** | ✅ | DDP (Bucketed Overlap) · Flash Attention 2 · Sharded Optimizer |
| 3 | **Scaling** | ✅ | IsoFLOPs 分析 · Chinchilla 最优外推 · Power-Law 拟合 |
| 4 | **Data** | ✅ | WET 清洗流水线 · 去重 · PII · 质量分类 · 模型训练 |
| 5 | **Alignment** | ⏳ | SFT · DPO · GRPO (待完成) |

> 📝 [浏览全部学习笔记](./docs/) &nbsp;|&nbsp; 📦 [官方课程仓库](https://github.com/stanford-cs336)

<br>

## 硬件

<div align="center">

| CPU | GPU | 内存 | 存储 |
|:---:|:---:|:---:|:---:|
| AMD Threadripper 9960X | 3× RTX 5090 32GB | 251 GB DDR5 | 938GB + 1.9TB |
| 24核 / 48线程 @ 5.49GHz | CUDA 13.0 | | NVMe + HDD |

</div>

<br>

## 仓库结构

```
.
├── assignment1-basics/     BPE 分词器 · Transformer 实现 · 训练
├── assignment2-systems/    分布式 DDP · Flash Attention · 优化器分片
├── assignment3-scaling/    IsoFLOPs · Chinchilla · Scaling Law 拟合
├── assignment4-data/       CC WET 清洗 · 去重 · PII · 质量过滤
└── docs/                   所有作业笔记
```

<br>

## 关键实现

<details open>
<summary><strong>Assignment 1 — Basics</strong></summary>

| 模块 | 源码 |
|---|---|
| BPE 分词器 (GPT-2 风格) | `cs336_basics/tokenizer.py` |
| Transformer (RMSNorm, RoPE, Multi-Head Attention, SwiGLU) | `cs336_basics/transformer.py` |
| 优化器 (SGD, AdamW, Cosine LR) | `cs336_basics/optimizer.py` |
| 文本生成 (temperature, top-p) | `cs336_basics/decoding.py` |

</details>

<details open>
<summary><strong>Assignment 2 — Systems</strong></summary>

| 模块 | 源码 |
|---|---|
| Bucketed Overlap DDP (梯度分桶 + 通信重叠) | `cs336_systems/ddp_bucketed.py` |
| Flash Attention 2 (PyTorch + Triton) | `cs336_systems/flash_attention.py` |
| Sharded Optimizer (ZeRO Stage 1) | `cs336_systems/sharded_optimizer.py` |
| 四维并行 Accounting (DP/FSDP/TP/PP) | `cs336_systems/optimizer_state_sharding_accounting.py` |

</details>

<details open>
<summary><strong>Assignment 3 — Scaling</strong></summary>

| 模块 | 源码 |
|---|---|
| IsoFLOPs 曲线分析 + Power-Law 拟合 | `cs336_scaling/isoflops.py` |
| 本地合成训练 API (Chinchilla 风格 surrogate) | `cs336_scaling/local_api.py` |
| 实验查询计划构建 | `cs336_scaling/scaling_plan.py` |
| Chinchilla 最优计算外推 | `scripts/chinchilla_isoflops.py` |

</details>

<details open>
<summary><strong>Assignment 4 — Data</strong></summary>

| 模块 | 源码 |
|---|---|
| HTML 文本抽取 (Resiliparse + 多编码回退) | `cs336_data/extraction.py` |
| fastText 语言识别 | `cs336_data/language.py` |
| PII 脱敏 (邮箱 / 电话 / IPv4) | `cs336_data/pii.py` |
| NSFW + Toxic 有害内容分类 | `cs336_data/harmful.py` |
| Gopher 规则 + wiki-vs-CC 质量分类器 | `cs336_data/quality.py` |
| 行级精确去重 + MinHash/LSH 文档级去重 | `cs336_data/deduplication.py` |
| 端到端 WET 过滤流水线 | `scripts/filter_data.py` |
| GPT-2 Tokenization → `np.uint16` 训练二进制 | `scripts/tokenize_data.py` |

</details>

<br>

## 快速开始

```bash
git clone https://github.com/QR-0W/Stanford-CS336.git
cd assignment4-data
python -m pytest -v          # 21 tests
python scripts/filter_data.py --input data/CC*.warc.wet.gz --output-dir data/out
```

<br>

<div align="center">

**仅用于教育学习目的 · 请遵守 Stanford 学术诚信政策**

</div>
