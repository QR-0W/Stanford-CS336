# Problem (nsys_profile) Deliverable

## Run status
- Sweep command executed via `cs336_systems/nsys_profile.py`.
- Total profiling runs: 60 (`results/nsys_profile/summary.jsonl`).
- Successful profiles: 46.
- Failed profiles: 14 (all OOM; see `results/nsys_profile/logs/*.profile.stderr.log`).

## A-E Answers (Chinese)

### (a) 总 forward 时间是否与 Python 计时一致？
以 `medium, context_length=256` 为例，nsys 运行下 forward 的均值约为 **25.34 ms**（见 `results/nsys_profile/logs/size-medium_ctx-256_mode-forward.benchmark.json`）；不用 nsys 的 Python 计时约为 **24.74 ms**。两者非常接近（约 2-3% 差异），说明 end-to-end 基准和 nsys 观测基本一致，只存在小幅 profiler 开销。

### (b) forward 中累计时间最高的 CUDA kernel 是什么？调用多少次？在 forward+backward 中是否相同？
forward 中累计时间最高的 kernel 是 CUTLASS GEMM：`cutlass_80_simt_sgemm_128x256_8x4_tn_align1`（见 `results/nsys_profile/stats/size-medium_ctx-256_mode-forward.stats.txt`）。在 `small/medium/large, ctx=256` 的 profile 中总调用次数分别为 **555/1095/1635**（对应 5 warmup + 10 measured 的总次数），换算单次 forward 约 **37/73/109** 次。在 `forward_backward` 中该 kernel 依然是累计时间第一（同样可见于对应 stats 文件）。

### (c) forward 中除了矩阵乘法，还有哪些 kernel 占用显著时间？
除了 GEMM，时间占比明显的还有 `elementwise_kernel` / `vectorized_elementwise_kernel`、`reduce_kernel`、以及 `exp/sigmoid` 等点算子和归约算子（见 `results/nsys_profile/stats/size-medium_ctx-256_mode-forward.stats.txt`）。这些 kernel 单次开销不大，但调用频繁且偏内存访问，累计起来占比不可忽略。

### (d) 完整训练步（forward+backward+optimizer）相比仅 inference，矩阵乘法占比如何变化？
以 `medium, ctx=256` 为例，从 `cuda_gpu_kern_sum` 估算，矩阵乘法相关 kernel 占比在 forward-only 约 **81.6%**，在 full train step 约 **65.3%**。这说明训练时除了 GEMM，还增加了较多梯度与优化器相关的 pointwise/reduction 开销。

### (e) self-attention 内 softmax 与 matmul 的 runtime 对比，和 FLOPs 差异相比如何？
在 self-attention 路径中，softmax 相关 kernel（max/sum/exp/div）总耗时显著低于 matmul（GEMM），数量级上通常只是 GEMM 的小部分。趋势与 FLOPs 认知一致（matmul FLOPs 更大），但 runtime 差异并不完全等于 FLOPs 比例，因为 softmax 更偏 memory-bound，而 matmul 更能利用 Tensor Core 的高吞吐。

## OOM note for report
- OOM mainly appears on larger configs in backward/train-step, e.g. `large@1024` (forward_backward/train_step), `xl` and `2.7b` at larger contexts.
- These entries are explicitly recorded in `results/nsys_profile/summary.jsonl` and stderr logs.
