<div id="top"></div>

<!--
*** Stanford CS336: Language Modeling from Scratch
*** 课程作业与学习笔记
-->

<!-- 项目 SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- 项目 LOGO -->
<br />

<div align="center">

<h3 align="center">Stanford CS336: Language Modeling from Scratch</h3>

  <p align="center">
    从零开始构建大语言模型 - 课程作业实现与学习笔记
    <br />
    本仓库包含 Stanford CS336 课程的所有作业实现、实验笔记和个人理解。课程涵盖从 Tokenizer 到 Transformer 架构，从分布式训练到模型对齐的完整 LLM 开发流程。
    <br />
    <a href="https://github.com/QR-0W/Stanford-CS336/tree/main/docs"><strong>浏览文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/stanford-cs336">官方课程仓库</a>
    ·
    <a href="https://github.com/QR-0W/Stanford-CS336/issues">反馈 Bug</a>

  </p>

</div>

<!-- 目录 -->

<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于本项目">关于本项目</a>
      <ul>
        <li><a href="#技术栈">技术栈</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始">开始</a>
      <ul>
        <li><a href="#依赖">依赖</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#作业概览">作业概览</a></li>
    <li><a href="#学习笔记">学习笔记</a></li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系我">联系我</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>

<!-- 关于本项目 -->

## 关于本项目

本项目是 Stanford CS336 课程的个人学习仓库，记录了从零开始构建大语言模型的完整过程。

**课程核心内容：**

- 🔤 **Tokenization**: 实现 BPE (Byte-Pair Encoding) 分词器
- 🧠 **Transformer Architecture**: 从头实现 Transformer（RMSNorm, RoPE, Multi-Head Attention, SwiGLU）
- ⚡ **Systems**: 分布式训练（DDP, Sharded Optimizer, Flash Attention）
- 📊 **Scaling Laws**: 研究模型规模与性能的关系
- 🗂️ **Data Processing**: 数据清洗、去重、质量过滤
- 🎯 **Alignment**: SFT, DPO, GRPO 等对齐技术

### 技术栈

- [Python 3.10+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [UV Package Manager](https://github.com/astral-sh/uv)
- [Transformers](https://huggingface.co/docs/transformers/)
- [NumPy](https://numpy.org/)

### 硬件配置信息

| 组件     | 配置                                                |
| -------- | --------------------------------------------------- |
| **CPU**  | AMD Ryzen Threadripper 9960X (24核/48线程, 5.49GHz) |
| **内存** | 251 GB DDR5                                         |
| **GPU**  | 3× NVIDIA GeForce RTX 5090 (32GB × 3 = 96GB)        |
| **存储** | 938GB NVMe + 1.9TB HDD                              |
| **CUDA** | 13.0                                                |

<!-- 开始 -->

## 开始

以下是在本地配置和运行项目的指南。

### 依赖

- Python 3.10+
- UV (推荐) 或 pip
- CUDA 13.0

### 安装

1. 克隆本仓库

```bash
git clone https://github.com/QR-0W/Stanford-CS336.git
cd Stanford-CS336
```

2. 安装 UV 包管理器（推荐）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. 进入具体作业目录并下载数据

```bash
cd assignment1-basics
./download_data.sh
```

4. 运行测试

```bash
uv run pytest
```

<!-- 作业概览 -->

## 作业概览

### Assignment 1: Basics

**状态**: ✅ 已完成

**主要任务**:

- [x] 环境配置
- [x] BPE Tokenizer 实现
- [x] Transformer 组件（RMSNorm, RoPE, Attention）
- [x] 完整语言模型训练

**学习笔记**: [Assignment 1 笔记](./docs/assignment1-notes.md)

---

### Assignment 2: Systems

**状态**: ✅ 已完成

**主要任务**:

- [x] 单节点分布式通信基准测试 (NCCL vs Gloo)
- [x] Naive DDP 实现与基准测试
- [x] DDP with Overlap (Individual Parameters)
- [x] Bucketed DDP (Gradient Bucketing + Communication Overlap)
- [x] Flash Attention 2 (PyTorch + Triton 实现)
- [x] Sharded Optimizer (ZeRO Stage 1 优化器状态分片)
- [x] 混合精度训练 (BF16) 内存与速度分析
- [x] 四维并行性 (DP/FSDP/TP/PP) 通信与内存 accounting

**关键文件**:
- ``cs336_systems/ddp_bucketed.py`` — Bucketed overlap DDP 实现
- ``cs336_systems/sharded_optimizer.py`` — 优化器状态分片
- ``cs336_systems/flash_attention.py`` — Flash Attention 2
- ``cs336_systems/optimizer_state_sharding_accounting.py`` — 四维并行内存/通信计算

**学习笔记**: [Assignment 2 笔记](./docs/assignment2-notes.md)

---

### Assignment 3: Scaling

**状态**: ⏳ 待开始

**主要任务**:

- [ ] Scaling Laws 实验
- [ ] 模型大小与性能关系研究

---

### Assignment 4: Data

**状态**: ✅ 已完成

**主要任务**:

- [x] HTML 文本抽取 (Resiliparse, 多编码回退)
- [x] 语言识别 (fastText LID, ``lid.176.ftz``)
- [x] PII 脱敏 (邮箱/电话/IPv4 正则替换)
- [x] 有害内容检测 (Dolma NSFW / Toxic Speech fastText 模型)
- [x] Gopher 质量规则 (词数、词长、省略号比例、字母比例)
- [x] 质量分类器 (fastText wiki-vs-CC, Wikipedia references → CC negatives)
- [x] 行级精确去重 (global line-frequency exact match)
- [x] MinHash/LSH 文档级近似去重 (word n-gram + Union-Find 聚类)
- [x] 端到端 WET 过滤流水线 (``filter_data.py``)
- [x] 过滤数据人工检查 (``inspect_filtered_data.py``)
- [x] GPT-2 tokenization + 二进制序列化 (``tokenize_data.py``)
- [x] GPT-2 small-shaped 模型训练 (2000 step smoke run, ``train_model``)

**关键文件**:
- ``cs336_data/deduplication.py`` — 精确 + MinHash/LSH 去重
- ``cs336_data/quality.py`` — Gopher 规则 + wiki-vs-CC 分类器
- ``cs336_data/harmful.py`` — NSFW / toxic 分类
- ``cs336_data/pii.py`` — 邮箱/电话/IP 脱敏
- ``cs336_data/language.py`` — fastText 语言识别
- ``cs336_data/extraction.py`` — HTML → 纯文本
- ``scripts/filter_data.py`` — 端到端 WET 过滤
- ``scripts/tokenize_data.py`` — GPT-2 tokenization
- ``scripts/train_quality_classifier.py`` — fastText 质量分类器训练

**学习笔记**: [Assignment 4 笔记](./docs/assignment4-notes.md)

---

### Assignment 5: Alignment

**状态**: ⏳ 待开始

**主要任务**:

- [ ] Supervised Fine-Tuning (SFT)
- [ ] Direct Preference Optimization (DPO)
- [ ] Group Relative Policy Optimization (GRPO)

<!-- 学习笔记 -->

## 学习笔记

### BPE Tokenizer 实现要点

**核心思想**: 从字节级别开始，反复合并高频相邻 token 对

**实现步骤**:

1. 初始化 256 个字节 token (0-255)
2. 统计所有相邻 token 对的频率
3. 合并频率最高的 token 对
4. 重复步骤 2-3 直到达到目标词表大小

**性能优化**:

- 使用分块处理避免内存溢出
- 增量更新频率统计
- 多进程并行化

更多笔记请查看各作业目录下的 `notes.md` 文件。

<!-- 路线图 -->

## 路线图

- [x] 仓库初始化
- [x] 下载所有作业代码
- [x] 配置开发环境
- [x] 完成 Assignment 1: Basics
  - [x] BPE Tokenizer
  - [x] Transformer 实现
  - [x] 训练流程
- [x] 完成 Assignment 2: Systems
- [ ] 完成 Assignment 3: Scaling
- [x] 完成 Assignment 4: Data
- [ ] 完成 Assignment 5: Alignment

到 [open issues](https://github.com/QR-0W/Stanford-CS336/issues) 页查看所有计划功能和已知问题。

<!-- 贡献 -->

## 贡献

这是一个个人学习项目，但欢迎任何建议和讨论！

如果你发现了 bug 或有改进建议：

1. Fork 本项目
2. 创建你的 Feature 分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到该分支 (`git push origin feature/AmazingFeature`)
5. 创建一个 Pull Request

<!-- 许可证 -->

## 许可证

本项目仅用于教育学习目的。请遵守 Stanford 的学术诚信政策。

<!-- 联系我 -->

## 联系我

项目链接: [https://github.com/QR-0W/Stanford-CS336](https://github.com/QR-0W/Stanford-CS336)

<!-- 致谢 -->

## 致谢

- [Stanford CS336 Official Repository](https://github.com/stanford-cs336)
- [othneildrew README Template](https://github.com/othneildrew/Best-README-Template)
- [UV Package Manager](https://github.com/astral-sh/uv)

<!-- MARKDOWN 链接 & 图片 -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/QR-0W/Stanford-CS336?style=for-the-badge
[contributors-url]: https://github.com/QR-0W/Stanford-CS336/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/QR-0W/Stanford-CS336?style=for-the-badge
[forks-url]: https://github.com/QR-0W/Stanford-CS336/network/members
[stars-shield]: https://img.shields.io/github/stars/QR-0W/Stanford-CS336?style=for-the-badge
[stars-url]: https://github.com/QR-0W/Stanford-CS336/stargazers
[issues-shield]: https://img.shields.io/github/issues/QR-0W/Stanford-CS336?style=for-the-badge
[issues-url]: https://github.com/QR-0W/Stanford-CS336/issues
[license-shield]: https://img.shields.io/github/license/QR-0W/Stanford-CS336?style=for-the-badge
[license-url]: https://github.com/QR-0W/Stanford-CS336/blob/main/LICENSE
