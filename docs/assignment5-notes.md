[TOC]

# 1 Assignment Overview

作业 5 主题是 alignment and reasoning RL，目标是让语言模型在数学题上进行 step-by-step reasoning。主线围绕 MATH 推理展开：zero-shot baseline → SFT → Expert Iteration → GRPO → leaderboard。

官方代码仓库为 `https://github.com/stanford-cs336/assignment5-alignment`，当前 `main` 为 Spring 2026，我们使用 `spring2025` tag。课程集群 MATH 数据（`/data/a5-alignment/MATH`）不在 GitHub 仓库中，本机也不存在。主线只要求通过 `tests/test_sft.py` 和 `tests/test_grpo.py`。

# 2 Reasoning with Language Models

## 2.1 Motivation

前几个作业关注 next-token prediction 和 cross-entropy，本节开始直接衡量模型能否把数学题做对。使用 Qwen 2.5 Math 1.5B Base，评估指标是最终答案正确率而非 loss。

## 2.2 Chain-of-Thought Reasoning and Reasoning RL

CoT 的做法是让模型先输出推理过程再输出最终答案。作业使用 R1-Zero prompt，要求输出 `<think>...</think><answer>...</answer>` 格式。`<think>` 中的推理过程可用作后续 SFT / Expert Iteration 的训练目标，`<answer>` 中的答案可被 parser 抽取与 ground truth 比较。

Expert Iteration 先让模型采样 reasoning traces，仅保留答案正确的 traces 继续 SFT。只要模型偶尔能采样出正确答案就可以自举出更好的训练数据。Reasoning RL 直接用 verified reward 训练模型，数学题用答案验证作 reward，不依赖人工偏好标注。

**Our setup: model and dataset**

MATH 数据集路径 `/data/a5-alignment/MATH` 是 Stanford/Together 集群共享目录，不是 GitHub 中的公开数据。PDF 建议无法访问 MATH 时使用 Countdown、GSM8K、Tulu 3 SFT Math 等替代。

自学版选择 GSM8K：它已在官方仓库 `data/gsm8k/` 中，题目是自然语言数学推理且比 Countdown 更接近 MATH，短答案在 `####` 后容易抽取。test split 共 1319 条样本。

模型通过 ModelScope 下载到本地 `models/Qwen/Qwen2___5-Math-1___5B`。vLLM 推理使用 `Conf_Test` conda 环境（`vllm==0.18.1`），3 × RTX 5090, ~32GB 每卡。

# 3 Measuring Zero-Shot MATH Performance

本节测量 base model 在 validation set 上的 zero-shot 表现，作为后续 SFT、Expert Iteration 和 GRPO 的 baseline。

## 3.1 Using vLLM for offline language model inference

后续评估和 RL rollout 需要大量生成，vLLM 的 PagedAttention、KV cache 管理和 CUDA kernel 优化能显著提高吞吐。课程集群模型路径 `/data/a5-alignment/models/Qwen2.5-Math-1.5B` 在本机不存在，通过 ModelScope 下载替代权重到 `models/Qwen/Qwen2___5-Math-1___5B`，约 2.9G。

## 3.2 Zero-shot MATH Baseline

**问题：math_baseline**

脚本评估 Qwen 2.5 Math 1.5B Base 在 MATH validation split 上的 zero-shot 表现。使用 R1-Zero prompt、vLLM 生成、`r1_zero_reward_fn`（或 GSM8K 下轻量 numeric reward）计算分数，并将 generations、metrics 和 category examples 序列化到磁盘。

由于本机没有 MATH 数据，默认使用 GSM8K 替代：从 `data/gsm8k/test.jsonl` 读取样本，`answer` 字段 `####` 后抽取短答案作为 `ground_truth`。GSM8K 下使用 numeric reward——仍检查 R1-Zero 格式标签，但允许 `</think>` 和 `<answer>` 之间出现换行/空白，答案正确性比较 `<answer>` 中最后一个数字与 GSM8K 短答案。

对应实现文件：

```text
assignment5-alignment/cs336_alignment/evaluation.py
assignment5-alignment/scripts/math_baseline.py
```

**(a)**

脚本支持 `--dataset-format {auto,math,gsm8k}` 和 `--reward-fn {auto,r1_zero,numeric}`。GSM8K 下自动选择 numeric reward，输出三类统计：`format_1_answer_1`（格式+答案正确）、`format_1_answer_0`（格式正确答案错误）、`format_0_answer_0`（格式错误答案错误）。

**(b)**

完整 GSM8K test split（1319 条）运行结果：

| 指标 | 数值 |
| --- | ---: |
| `format_reward` | `0.2691` |
| `answer_reward` / accuracy | `0.1107` |
| `reward` | `0.1107` |

| 类别 | 数量 |
| --- | ---: |
| `format_1_answer_1` | `146` |
| `format_1_answer_0` | `209` |
| `format_0_answer_0` | `964` |

`format_0_answer_0` 中多数模型未严格输出 `<answer>` 标签，或生成了 `<answer>` 之外的自然语言/代码块；`format_1_answer_0` 主要是格式满足但算错。Qwen 实际生成中常输出 `</think>\n<answer>`，numeric reward 用正则允许标签间有空白。

**(c)**

Qwen 2.5 Math 1.5B Base 在 GSM8K 上的 zero-shot accuracy 约 11.07%，格式符合率约 26.9%。

输出文件：

```text
outputs/math_baseline_gsm8k/generations.jsonl
outputs/math_baseline_gsm8k/metrics.json
outputs/math_baseline_gsm8k/category_examples.json
```

**测试**：语法检查通过（`py_compile`），fake vLLM smoke test 确认 GSM8K 抽取/prompt/reward/汇总流程正常，真实 vLLM 小样本和完整 test split 均已运行完成。

**运行命令**（自学版 GSM8K）：

```bash
cd assignment5-alignment
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --gpu-memory-utilization 0.60 \
  --output-dir outputs/math_baseline_gsm8k
```

若在课程集群上有原始 MATH 数据，可切换回 handout 配置：

```bash
uv run python scripts/math_baseline.py \
  --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data /data/a5-alignment/MATH/validation.jsonl \
  --dataset-format math \
  --reward-fn r1_zero \
  --output-dir outputs/math_baseline_math
```

一句话总结：

> `math_baseline` 实现了 GSM8K zero-shot 推理基线脚本；Qwen 2.5 Math 1.5B Base 在 1319 条 GSM8K 样本上 accuracy 约 11.07%，格式正确率约 26.9%，作为后续 SFT/GRPO 的 baseline。

# 4 Supervised Finetuning for MATH

本节用带 reasoning trace 的标注数据做 SFT。样本为 `(prompt, response)`，response 包含 `<think>...</think><answer>...</answer>`。训练时只在 response token 上计算 cross-entropy loss，prompt 不参与。官方 reasoning traces 来自 DeepSeek R1，路径 `/data/a5-alignment/MATH/sft.jsonl`，本机无该数据。

## 4.1 Using HuggingFace Models

HuggingFace 模型加载建议使用 `torch_dtype=torch.bfloat16` 和 `attn_implementation="flash_attention_2"`。本节还引入 gradient accumulation：大 batch 拆为 `k` 个 microbatch，每个 loss 除以 `k` 后 backward，累计 `k` 次再 `optimizer.step()`。

## 4.2 SFT Helper Methods

以下 helper 会被 SFT 和后续 RL（Expert Iteration、GRPO）复用。completion、output、response 在文档中视为同义词。

**问题：tokenize_prompt_and_output**

对每个 `(prompt, response)` 对分别 tokenize 并拼接，返回 `input_ids`（去掉最后 token）、`labels`（去掉第一 token）和 `response_mask`（与 `labels` 对齐，response 位置为 `True`）。不同样本 pad 到 batch 最大长度再统一切位。mask 对齐：设 prompt 长 `P`、output 长 `O`，response label 在 `labels` 中的区间为 `[P-1, P+O-2)`。使用 `tokenizer.encode(..., add_special_tokens=False)` 分别 tokenize，不额外加入 BOS/EOS。

对应实现文件：

```text
assignment5-alignment/tests/adapters.py
```

**测试**：官方命令 `uv run pytest -k test_tokenize_prompt_and_output` 因 fixture 写死课程集群路径 `/data/a5-alignment/models/Qwen2.5-Math-1.5B` 无法直接运行。使用本地 ModelScope Qwen tokenizer 对 `tests/_snapshots/test_tokenize_prompt_and_output.npz` 做等价验证，`input_ids`、`labels`、`response_mask` 与官方 snapshot 完全一致。

一句话总结：

> `tokenize_prompt_and_output` 将 prompt 与 output 分别 tokenize 后拼接，构造只在 response 位置为 `True` 的 mask，是 SFT/Expert Iteration/GRPO 训练流程的共用基础函数。

**问题：compute_entropy**

输入 logits `(batch_size, sequence_length, vocab_size)`，输出 per-token 离散熵 `(batch_size, sequence_length)`。使用 `log_softmax` + `exp` 做数值稳定计算：`H = -Σ(p * log p)` = `-(softmax(logits) * log_softmax(logits)).sum(-1)`。

对应实现文件：

```text
assignment5-alignment/tests/adapters.py
```

**测试**：使用 `tests/_snapshots/test_compute_entropy.npz` 等价验证通过。

**问题：get_response_log_probs**

从 causal LM 获取每个 label token 的条件 log-prob：`model(input_ids).logits` → `log_softmax` → `gather(labels)`。causal LM 中 `logits[:, t, :]` 预测 `labels[:, t]`。可选通过 `return_token_entropy=True` 同时返回 per-token 熵（调用 `compute_entropy`）。

对应实现文件：

```text
assignment5-alignment/tests/adapters.py
```

**测试**：使用本地 ModelScope Qwen fp32 权重对 `tests/_snapshots/test_get_response_log_probs.npz` 等价验证，`log_probs` max diff 约 `0.00006`，`token_entropy` max diff 约 `0.0007`，通过。

**问题：masked_normalize**

将被 mask 选中的元素沿指定 `dim` 求和，再除以 `normalize_constant`。`dim=None` 时对所有元素求和返回标量。实现为 `(tensor * mask.float()).sum(dim) / normalize_constant`。

对应实现文件：

```text
assignment5-alignment/tests/adapters.py
```

**测试**：dim=0/1/-1/None 四种 snapshot 全部通过。

**问题：sft_microbatch_train_step**

单个 microbatch 的 SFT 前向+反向传播。loss 公式：`-Σ(log_prob_i * mask_i) / (normalize_constant * batch_size * gradient_accumulation_steps)`，其中 batch_size 从 `policy_log_probs.shape[0]` 推断。函数内调用 `loss.backward()`，PyTorch 自动累积梯度供后续 `optimizer.step()` 使用。

对应实现文件：

```text
assignment5-alignment/tests/adapters.py
```

**测试**：`test_sft_microbatch_train_step`、`test_sft_microbatch_train_step_normalize`、`test_sft_microbatch_train_step_10_steps` 三个 snapshot 全部通过。
