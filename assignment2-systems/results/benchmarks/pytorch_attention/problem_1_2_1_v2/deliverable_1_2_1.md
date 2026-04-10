# 1.2.1 `pytorch_attention`

## 实验设置

- 实现脚本：`cs336_systems/pytorch_attention.py`
- 环境：`coding` conda env，CUDA GPU
- 固定 batch size：`8`
- 不使用 multi-head（输入形状为 `[B, T, d_model]`）
- 使用因果 mask（causal masking）
- 扫描网格：`d_model in [16, 32, 64, 128]`，`seq_len in [256, 1024, 4096, 8192, 16384]`
- 每个配置：warmup `10` 次，forward 计时 `100` 次，backward 计时 `100` 次；每次后 `torch.cuda.synchronize()`
- backward 前显存统计：`torch.cuda.memory_allocated()`（记录均值/最大值）

运行命令：

```bash
/mdata/wjx/miniconda3/bin/conda run -n coding \
python cs336_systems/pytorch_attention.py \
  --output-dir results/benchmarks/pytorch_attention/problem_1_2_1_v2
```

## 结果表

| d_model | seq_len | status | fwd_mean_ms | bwd_mean_ms | mem_before_bwd_mean_mib | mem_before_bwd_max_mib | peak_allocated_mib |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | 256 | ok | 4.378 | 4.515 | 21.46 | 21.46 | 29.59 |
| 16 | 1024 | ok | 3.944 | 4.863 | 91.84 | 91.84 | 220.34 |
| 16 | 4096 | oom | - | - | - | - | - |
| 16 | 8192 | oom | - | - | - | - | - |
| 16 | 16384 | oom | - | - | - | - | - |
| 32 | 256 | ok | 4.152 | 4.373 | 22.09 | 22.09 | 30.34 |
| 32 | 1024 | ok | 4.381 | 1.956 | 94.34 | 94.34 | 223.34 |
| 32 | 4096 | ok | 13.417 | 38.595 | 1204.62 | 1204.62 | 3256.62 |
| 32 | 8192 | oom | - | - | - | - | - |
| 32 | 16384 | oom | - | - | - | - | - |
| 64 | 256 | ok | 4.483 | 4.392 | 23.34 | 23.34 | 31.84 |
| 64 | 1024 | ok | 3.994 | 4.902 | 99.34 | 99.34 | 229.34 |
| 64 | 4096 | oom | - | - | - | - | - |
| 64 | 8192 | oom | - | - | - | - | - |
| 64 | 16384 | oom | - | - | - | - | - |
| 128 | 256 | ok | 4.490 | 4.490 | 25.84 | 25.84 | 34.84 |
| 128 | 1024 | ok | 4.743 | 4.750 | 109.34 | 109.34 | 241.34 |
| 128 | 4096 | ok | 18.693 | 43.259 | 1264.62 | 1264.62 | 3328.62 |
| 128 | 8192 | oom | - | - | - | - | - |
| 128 | 16384 | oom | - | - | - | - | - |

原始文件：

- `results/benchmarks/pytorch_attention/problem_1_2_1_v2/results.json`
- `results/benchmarks/pytorch_attention/problem_1_2_1_v2/table.md`

## OOM 配置与内存核算

本次运行中最小 OOM 配置是：`(d_model=16, seq_len=4096)`。

对该配置做一个朴素注意力（FP32）内存估算：

- `scores` 大小：`B*T*T*4 = 8*4096*4096*4 = 536,870,912 bytes = 512 MiB`
- `probs` 大小：同样约 `512 MiB`
- `Q,K,V` 合计：`3*B*T*d*4 = 6,291,456 bytes ≈ 6 MiB`
- 输出 `O`：`B*T*d*4 = 2,097,152 bytes ≈ 2 MiB`

仅 forward 关键中间量粗略就约 `512 + 512 + 6 + 2 = 1032 MiB`，而 backward 还会引入额外梯度与中间计算缓存，显存压力进一步上升，因此在可用显存不足时很容易 OOM。

## 简要结论（1-2 段）

随着序列长度增长，forward/backward 运行时间整体上升，且在可运行配置中 backward 通常慢于 forward。更关键的是，`memory_before_backward` 随 `seq_len` 增长显著上升，说明反向传播前需要保留的大量中间状态是主要内存瓶颈。这个趋势与朴素注意力中 `T x T` 矩阵（scores/probs）带来的二次复杂度一致。

从 OOM 边界也能看出问题核心：在较长序列（本次实验里从 4096 或 8192 开始，取决于 d_model）时，显存瓶颈首先出现，随后计算时间也急剧恶化。要消除这类内存成本，关键是避免显式物化完整 `T x T` 注意力矩阵并减少反向保存的中间激活，典型做法就是使用 tiled/fused 的 FlashAttention 类内核与重计算策略（recomputation/checkpointing）。
