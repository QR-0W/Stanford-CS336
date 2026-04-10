# 1 概述

## 1.1 性能分析和基准测试

### 1.1.3 端到端基准测试

**问题 (benchmarking_script)**

A：一个脚本

B：在 `warmup=5`、`measure=10`、`batch_size=4`、`context_length=256`、`float32` 下，前向平均耗时约为：small **19.15 ms**、medium **50.67 ms**、large **111.76 ms**；反向平均耗时约为：small **36.63 ms**、medium **103.52 ms**（large backward 以及更大模型在本机 32GB 显存下 OOM）。各成功配置的标准差整体较小（约 0.17ms–2.56ms），说明 warmup 后测量波动不大。

C：不做 warmup 时，耗时会明显偏高且波动更大：以 small 为例，前向为 **32.51±40.59 ms**、反向为 **40.40±11.72 ms**，主要是首轮包含了 CUDA 上下文初始化、内存分配器状态建立和内核路径初始化等一次性开销。做 1-2 次 warmup 后结果会改善（例如前向约 20.46/20.20 ms，反向约 36.23/37.23 ms），但仍可能和 5 次 warmup 有差异，因为缓存与执行路径不一定已完全进入稳定状态。warmup 足够后，测得时间更接近稳态吞吐。

###  1.1.4 Nsight Systems 分析器

**问题(nsys_profile)**

A：以 `medium, context_length=256` 为例，nsys 运行下 forward 的均值约为 **25.34 ms**（见 `results/nsys_profile/logs/size-medium_ctx-256_mode-forward.benchmark.json`）；不用 nsys 的 Python 计时约为 **24.74 ms**。两者非常接近（约 2-3% 差异），说明 end-to-end 基准和 nsys 观测基本一致，只存在小幅 profiler 开销。

B：forward 中累计时间最高的 kernel 是 CUTLASS GEMM：`cutlass_80_simt_sgemm_128x256_8x4_tn_align1`（见 `results/nsys_profile/stats/size-medium_ctx-256_mode-forward.stats.txt`）。在 `small/medium/large, ctx=256` 的 profile 中总调用次数分别为 **555/1095/1635**（对应 5 warmup + 10 measured 的总次数），换算单次 forward 约 **37/73/109** 次。在 `forward_backward` 中该 kernel 依然是累计时间第一（同样可见于对应 stats 文件）。

C：除了 GEMM，时间占比明显的还有 `elementwise_kernel` / `vectorized_elementwise_kernel`、`reduce_kernel`、以及 `exp/sigmoid` 等点算子和归约算子（见 `results/nsys_profile/stats/size-medium_ctx-256_mode-forward.stats.txt`）。这些 kernel 单次开销不大，但调用频繁且偏内存访问，累计起来占比不可忽略。

D：以 `medium, ctx=256` 为例，从 `cuda_gpu_kern_sum` 估算，矩阵乘法相关 kernel 占比在 forward-only 约 **81.6%**，在 full train step 约 **65.3%**。这说明训练时除了 GEMM，还增加了较多梯度与优化器相关的 pointwise/reduction 开销。

E：在 self-attention 路径中，softmax 相关 kernel（max/sum/exp/div）总耗时显著低于 matmul（GEMM），数量级上通常只是 GEMM 的小部分。趋势与 FLOPs 认知一致（matmul FLOPs 更大），但 runtime 差异并不完全等于 FLOPs 比例，因为 softmax 更偏 memory-bound，而 matmul 更能利用 Tensor Core 的高吞吐。

### 1.1.5 混合精度

**问题(mixed_precision_accumulation)**

FP32 + FP32 = `10.0001335`，FP16 + FP16 = `9.953125`，FP32 + FP16 = `10.0021362`，FP16 先转 FP32 再累加 = `10.0021362`。可以看出纯 FP16 累加误差最大，因为随着和变大，FP16 尾数精度不足，很多 `+0.01` 会被舍入吞掉；而用 FP32 做累加时精度明显更好。同时，若每一步的加数本身先以 FP16 表示（`0.01` 已量化），即使再转成 FP32 累加也无法恢复这部分量化误差，所以后两种结果基本一致。

**问题(benchmarking_mixed_precision)**

A：

- 模型参数（autocast 上下文内）: `torch.float32`
- `ToyModel.fc1` 输出: `torch.float16`
- `ToyModel.ln` 输出: `torch.float32`
- 模型 logits: `torch.float16`
- loss: `torch.float32`
- 梯度: `torch.float32`

B：LayerNorm 中最敏感的是均值/方差的归约与归一化计算（减均值、平方、求和、`rsqrt`），这些步骤对数值范围和舍入误差都更敏感。FP16 尾数较短且动态范围更窄，容易带来更明显的精度损失，因此通常在更高精度（FP32）里做这类累积/归约。BF16 的动态范围与 FP32 接近，溢出/下溢风险比 FP16 小很多，但尾数仍较短，所以实践中仍常把 LayerNorm 的关键计算保留在 FP32 以保证稳定性。

C：在 `batch_size=4, context_length=256, warmup=5, measure=10` 下，BF16 混合精度相对 FP32 在可运行模型上均明显加速：small 前向 `0.02177s→0.01222s (1.78x)`、反向 `0.03616s→0.02552s (1.42x)`；medium 前向 `1.86x`、反向 `1.69x`；large 前向 `1.71x`、反向 `1.57x`。总体趋势是模型变大后混合精度仍保持稳定收益，且前向提速通常略高于反向。`xl` 和 `2.7b` 在当前显存配置下出现 OOM（两种精度均有），已在结果中记录。

### 1.1.6 内存分析

**问题(memory_profiling)**

A：![image-20260410151650071](./assets/image-20260410151650071.png)

![image-20260410151749740](./assets/image-20260410151749740.png)

BF16 前向时间线整体呈现较高基线并随序列长度增加而上升，主要由大模型参数常驻显存与激活增长共同决定。对 2.7B 模型做完整训练步时，本机 31.36 GiB 显存无法在题目参考设置下完成运行，因此没有可用的 train-step 时间线图。

FP32 forward 的 active memory timeline 较平稳，不是因为立即 OOM，而是因为 2.7B 参数常驻显存占主导，前向激活增量在 ctx=128 下相对较小。

B：峰值显存表（参考设置，2.7B, FP32）

下表使用 `torch.cuda.max_memory_allocated`（单位 MiB）：

| Context length | 前向峰值 (MiB) | 完整训练步峰值 (MiB) |
| -------------- | -------------: | -------------------: |
| 128            |       13234.02 |                  OOM |
| 256            |       13335.28 |                  OOM |
| 512            |       13750.54 |                  OOM |

前向数据来源：`results/memory_profiling/isolated_ctx*_full/results.json`。

C：在本实验环境中，BF16 autocast 的前向峰值显存高于 FP32 前向：

- ctx=128：FP32 `13234.02 MiB`，BF16 `19600.35 MiB`
- ctx=256：FP32 `13335.28 MiB`，BF16 `19650.28 MiB`
- ctx=512：FP32 `13750.54 MiB`，BF16 `19850.42 MiB`

完整训练步在 FP32 与 BF16 下均 OOM，因此 mixed precision 在该硬件和参考超参数下未能让 2.7B 的 train-step 变为可行。结论是：本机上 mixed precision 明显提升速度，但并未显著降低到可支撑完整训练步的显存水平。

D：残差流激活张量形状为 `[B, T, d_model]`，单精度每元素 4 字节，因此：

`bytes = B * T * d_model * 4`

在参考设置中 `B=4`，2.7B 配置 `d_model=2560`，所以：

- `T=128`：`4*128*2560*4 = 5,242,880 bytes = 5.00 MiB`
- `T=256`：`10.00 MiB`
- `T=512`：`20.00 MiB`

因此本题最大上下文 `T=512` 时，单个残差流激活张量大小为 `20.00 MiB`。

E：使用带栈追踪的前向快照：

- `results/memory_profiling/problem_1_1_6_trace_py/snapshots/size-2.7b_ctx-128_mode-forward_precision-full_fp32.pickle`

可见最大活跃分配块为重复出现的 `100.0 MiB`。栈信息指向 `assignment1-basics/cs336_basics/transformer.py` 的模型构造路径（由 `cs336_systems/benchmarking_script.py` 中 `_build_model` 调用），说明这些最大块主要来自模型参数分配，而不是单步前向中的瞬时激活。

## 1.2 Flash Attention 2 优化注意力

### 1.2.1 Pytorch 注意力基准测试

**问题 (pytorch_attention)**

| d_model | seq_len | status | fwd_mean_ms | bwd_mean_ms | mem_before_bwd_mean_mib | mem_before_bwd_max_mib | peak_allocated_mib |
| ------: | ------: | ------ | ----------: | ----------: | ----------------------: | ---------------------: | -----------------: |
|      16 |     256 | ok     |       4.378 |       4.515 |                   21.46 |                  21.46 |              29.59 |
|      16 |    1024 | ok     |       3.944 |       4.863 |                   91.84 |                  91.84 |             220.34 |
|      16 |    4096 | oom    |           - |           - |                       - |                      - |                  - |
|      16 |    8192 | oom    |           - |           - |                       - |                      - |                  - |
|      16 |   16384 | oom    |           - |           - |                       - |                      - |                  - |
|      32 |     256 | ok     |       4.152 |       4.373 |                   22.09 |                  22.09 |              30.34 |
|      32 |    1024 | ok     |       4.381 |       1.956 |                   94.34 |                  94.34 |             223.34 |
|      32 |    4096 | ok     |      13.417 |      38.595 |                 1204.62 |                1204.62 |            3256.62 |
|      32 |    8192 | oom    |           - |           - |                       - |                      - |                  - |
|      32 |   16384 | oom    |           - |           - |                       - |                      - |                  - |
|      64 |     256 | ok     |       4.483 |       4.392 |                   23.34 |                  23.34 |              31.84 |
|      64 |    1024 | ok     |       3.994 |       4.902 |                   99.34 |                  99.34 |             229.34 |
|      64 |    4096 | oom    |           - |           - |                       - |                      - |                  - |
|      64 |    8192 | oom    |           - |           - |                       - |                      - |                  - |
|      64 |   16384 | oom    |           - |           - |                       - |                      - |                  - |
|     128 |     256 | ok     |       4.490 |       4.490 |                   25.84 |                  25.84 |              34.84 |
|     128 |    1024 | ok     |       4.743 |       4.750 |                  109.34 |                 109.34 |             241.34 |
|     128 |    4096 | ok     |      18.693 |      43.259 |                 1264.62 |                1264.62 |            3328.62 |
|     128 |    8192 | oom    |           - |           - |                       - |                      - |                  - |
|     128 |   16384 | oom    |           - |           - |                       - |                      - |                  - |

## 1.3 JIT编译注意力机制的基准测试

