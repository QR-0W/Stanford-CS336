# 1.1.6 `memory_profiling`

## (a) Active Memory Timeline 与现象说明

本题已生成可在 https://pytorch.org/memory_viz 中加载的快照文件：

- 仅前向（2.7B, FP32）
  - `results/memory_profiling/isolated_ctx128_full/snapshots/size-2.7b_ctx-128_mode-forward_precision-full_fp32.pickle`
  - `results/memory_profiling/isolated_ctx256_full/snapshots/size-2.7b_ctx-256_mode-forward_precision-full_fp32.pickle`
  - `results/memory_profiling/isolated_ctx512_full/snapshots/size-2.7b_ctx-512_mode-forward_precision-full_fp32.pickle`
- 仅前向（2.7B, BF16 autocast）
  - `results/memory_profiling/isolated_ctx128_mixed/snapshots/size-2.7b_ctx-128_mode-forward_precision-mixed_bf16.pickle`
  - `results/memory_profiling/isolated_ctx256_mixed/snapshots/size-2.7b_ctx-256_mode-forward_precision-mixed_bf16.pickle`
  - `results/memory_profiling/isolated_ctx512_mixed/snapshots/size-2.7b_ctx-512_mode-forward_precision-mixed_bf16.pickle`

在参考超参数（2.7B, batch=4, AdamW）下，`forward+backward+optimizer step` 在 `context_length=128/256/512` 全部 OOM，详细记录见 `results/memory_profiling/problem_1_1_6/results.json`。

可交付回答（2-3 句）：
前向时间线整体呈现较高基线并随序列长度增加而上升，主要由大模型参数常驻显存与激活增长共同决定。对 2.7B 模型做完整训练步时，本机 31.36 GiB 显存无法在题目参考设置下完成运行，因此没有可用的 train-step 时间线图。结合小模型与机制分析，最大峰值通常出现在 backward 阶段，其后 optimizer step 会引入/访问优化器状态。

## (b) 峰值显存表（参考设置，2.7B, FP32）

下表使用 `torch.cuda.max_memory_allocated`（单位 MiB）：

| Context length | 前向峰值 (MiB) | 完整训练步峰值 (MiB) |
|---|---:|---:|
| 128 | 13234.02 | OOM |
| 256 | 13335.28 | OOM |
| 512 | 13750.54 | OOM |

前向数据来源：`results/memory_profiling/isolated_ctx*_full/results.json`。

## (c) 混合精度下的峰值显存（BF16 autocast）

在本实验环境中，BF16 autocast 的前向峰值显存高于 FP32 前向：

- ctx=128：FP32 `13234.02 MiB`，BF16 `19600.35 MiB`
- ctx=256：FP32 `13335.28 MiB`，BF16 `19650.28 MiB`
- ctx=512：FP32 `13750.54 MiB`，BF16 `19850.42 MiB`

完整训练步在 FP32 与 BF16 下均 OOM，因此 mixed precision 在该硬件和参考超参数下未能让 2.7B 的 train-step 变为可行。结论是：本机上 mixed precision 明显提升速度，但并未显著降低到可支撑完整训练步的显存水平。

## (d) 2.7B 残差流激活张量大小推导（FP32）

残差流激活张量形状为 `[B, T, d_model]`，单精度每元素 4 字节，因此：

`bytes = B * T * d_model * 4`

在参考设置中 `B=4`，2.7B 配置 `d_model=2560`，所以：

- `T=128`：`4*128*2560*4 = 5,242,880 bytes = 5.00 MiB`
- `T=256`：`10.00 MiB`
- `T=512`：`20.00 MiB`

因此本题最大上下文 `T=512` 时，单个残差流激活张量大小为 `20.00 MiB`。

## (e) 最大分配块大小与来源

使用带栈追踪的前向快照：

- `results/memory_profiling/problem_1_1_6_trace_py/snapshots/size-2.7b_ctx-128_mode-forward_precision-full_fp32.pickle`

可见最大活跃分配块为重复出现的 `100.0 MiB`。栈信息指向 `assignment1-basics/cs336_basics/transformer.py` 的模型构造路径（由 `cs336_systems/benchmarking_script.py` 中 `_build_model` 调用），说明这些最大块主要来自模型参数分配，而不是单步前向中的瞬时激活。
