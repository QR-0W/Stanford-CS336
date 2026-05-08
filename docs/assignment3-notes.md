[TOC]

# Assignment 3: Scaling Laws

本作业的核心问题是：在固定训练计算预算 `C` 下，应该如何在模型大小 `N` 和训练 token 数 `D` 之间分配计算量，使最终训练 loss 尽可能低。

常用近似为：

```text
C ≈ 6ND
```

其中 `N` 是模型参数量，`D` 是训练 token 数。Assignment 3 分成两部分：

- `chinchilla_isoflops`：使用作业提供的 synthetic IsoFLOPs 数据，复现 Chinchilla 风格的 scaling-law 拟合。
- `scaling_laws`：官方版本需要访问 Stanford Training API。由于我不是 Stanford 学生，无法访问官方 API，因此这里实现了一个公开课自学版的 local synthetic training API，用来复现实验设计、预算控制和 scaling law 拟合流程。

# 2 缩放定律回顾

### 2.1 从等FLOPs曲线得出的缩放定律

**问题：chinchilla_isoflops**

做法：

1. 读取 `assignment3-scaling/data/isoflops_curves.json`。
2. 按 `compute_budget` 分组。
3. 对每个固定计算预算 `C_i`，选择 `final_loss` 最低的运行作为该 IsoFLOPs 曲线上的最优点。
4. 得到 `⟨C_i, N_opt(C_i)⟩`。
5. 用 `D_opt(C_i) = C_i / (6N_opt(C_i))` 得到对应的最优训练 token 数。
6. 在 log-log 空间中拟合 power law：

```text
N_opt(C) = a_N C^alpha_N
D_opt(C) = a_D C^alpha_D
```

模型大小拟合结果：

```text
N_opt(C) = 1.163411 * C^0.468683
```

![image-20260507154047157](./assets/image-20260507154047157.png)

根据 IsoFLOPs scaling law 外推，`10^23` FLOPs 下的计算最优模型大小约为 `70B` 参数，`10^24` FLOPs 下约为 `206B` 参数。

数据集大小拟合结果：

```text
D_opt(C) = 0.143257 * C^0.531317
```

![image-20260507154101789](./assets/image-20260507154101789.png)

根据 IsoFLOPs scaling law 外推，`10^23` FLOPs 下的计算最优数据集大小约为 `238B` tokens，`10^24` FLOPs 下约为 `809B` tokens。

# 3 构建缩放定律

**问题：scaling_laws**

## 3.1 官方 API 与自学版替代方案

官方作业要求使用 Stanford Training API 查询真实训练结果。API 的输入包括：

- `d_model`
- `num_layers`
- `num_heads`
- `batch_size`
- `learning_rate`
- `train_flops`

API 返回对应配置训练后的 `final_loss`。官方 API 背后的训练运行使用了 §3.2 描述的 Transformer 结构，包括 absolute position embeddings、LayerNorm、GELU FFN、dropout、untied embeddings、AdamW、cosine learning-rate schedule、SlimPajama 数据集等。

但是这个 API 需要课程注册过的 SSH public key 和 Stanford 网络/VPN。作为公开课自学者，我无法访问该 API。因此本节采用一个 local synthetic training API 替代官方 API，用来练习同样的方法论：

- 如何规划查询预算。
- 如何选择待查询的模型大小和超参数。
- 如何构造 IsoFLOPs 曲线。
- 如何拟合 scaling law。
- 如何外推到 `1e19` FLOPs。

需要明确的是：这里的 local API **没有真实训练 Transformer**，也不代表 Stanford 官方隐藏训练服务的结果。它只是一个可解释的 synthetic surrogate。

## 3.2 Local Synthetic Training API

本地 API 使用作业给出的非 embedding 参数量估计：

```text
N ≈ 12 * num_layers * d_model^2
```

给定训练计算预算 `C` 后，训练 token 数估计为：

```text
D = C / (6N)
```

synthetic final loss 使用如下形式：

```text
L(N, D) = E + A / N^alpha + B / D^beta + hyperparameter_penalty
```

其中 `hyperparameter_penalty` 让 loss 对 learning rate、batch size 和 head dimension 有轻微依赖。这样做的目的不是模拟某个真实数据集的精确 loss，而是提供一个具有 Chinchilla 风格权衡关系的可复现实验环境。

对应实现：

- `assignment3-scaling/cs336_scaling/local_api.py`
- `assignment3-scaling/scripts/run_local_scaling_study.py`
- `assignment3-scaling/scripts/fit_scaling_laws.py`

## 3.3 查询预算与实验设计

作业规定 scaling-law 实验预算不能超过：

```text
2e18 FLOPs
```

本地自学版实验共查询 `75` 个配置，总预算为：

```text
1.983e18 FLOPs
```

预算分配如下：

| 阶段 | 配置数 | 预算 |
| --- | ---: | ---: |
| `pilot_hparams` | `40` | `4.0e16` FLOPs |
| `stage1_isoflops` | `23` | `3.03e17` FLOPs |
| `stage2_high_compute` | `12` | `1.64e18` FLOPs |
| 合计 | `75` | `1.983e18` FLOPs |

设计思路：

1. 先用低 FLOPs 的 `pilot_hparams` 测试 `batch_size ∈ {128, 256}` 和若干 learning rate。
2. 再用 `stage1_isoflops` 在多个 `train_flops` 上扫描模型大小，得到初步 IsoFLOPs 最优点。
3. 最后用 `stage2_high_compute` 在较高 FLOPs 下补充曲线，避免只靠低预算外推。

## 3.4 IsoFLOPs 最优点

每个计算预算下，选择 loss 最低的配置作为 `N_opt(C)`。

| `C` | `N_opt` | loss | layers | `d_model` | heads | batch | lr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e15` | `2.65M` | `3.868251` | `6` | `192` | `3` | `128` | `6e-4` |
| `3e15` | `2.65M` | `3.596996` | `6` | `192` | `3` | `128` | `6e-4` |
| `1e16` | `6.29M` | `3.366188` | `8` | `256` | `4` | `128` | `6e-4` |
| `3e16` | `6.29M` | `3.178757` | `8` | `256` | `4` | `128` | `6e-4` |
| `6e16` | `6.29M` | `3.084462` | `8` | `256` | `4` | `128` | `6e-4` |
| `1e17` | `17.69M` | `3.018695` | `10` | `384` | `6` | `128` | `6e-4` |
| `3e17` | `17.69M` | `2.890006` | `10` | `384` | `6` | `128` | `6e-4` |

## 3.5 拟合结果

模型大小 scaling law：

```text
N_opt(C) = 10.107419 * C^0.356427
```

![assignment3-local-model-size-fit](./assets/assignment3-local-model-size-fit.png)

loss scaling law：

```text
L_opt(C) = 2.256312 + 447.893350 * C^-0.162956
```

![assignment3-local-loss-fit](./assets/assignment3-local-loss-fit.png)

## 3.6 对 `1e19` FLOPs 的预测

连续 scaling law 外推得到：

```text
N_opt(1e19) ≈ 59.81M non-embedding parameters
predicted loss ≈ 2.615236
```

在 API 允许的离散超参数空间中，选择最接近该参数量的模型配置：

| 超参数 | 推荐值 |
| --- | ---: |
| `num_layers` | `22` |
| `d_model` | `476` |
| `num_heads` | `7` |
| estimated non-embedding params | `59.82M` |
| `batch_size` | `128` |
| `learning_rate` | `6e-4` |

一句话总结：

> 在本地 synthetic training API 上，我使用 `1.983e18` FLOPs 的实验预算构造 IsoFLOPs 曲线并拟合 scaling law，外推得到 `1e19` FLOPs 下的计算最优模型大小约为 `59.8M` non-embedding parameters，预测训练 loss 约为 `2.615`；对应推荐配置为 `22` layers、`d_model=476`、`7` heads、`batch_size=128`、`learning_rate=6e-4`。

## 3.7 复现命令

生成 `chinchilla_isoflops` 图和结果：

```bash
cd /mdata/wjx/CS336/assignment3-scaling
/mdata/wjx/miniconda3/bin/conda run -n coding python scripts/chinchilla_isoflops.py
```

运行本地自学版 scaling-law 实验：

```bash
cd /mdata/wjx/CS336/assignment3-scaling
/mdata/wjx/miniconda3/bin/conda run -n coding python scripts/run_local_scaling_study.py
/mdata/wjx/miniconda3/bin/conda run -n coding python scripts/fit_scaling_laws.py \
  --runs results/scaling_laws/local_runs.json \
  --output-dir results/scaling_laws/local_fit \
  --target-compute 1e19
```
