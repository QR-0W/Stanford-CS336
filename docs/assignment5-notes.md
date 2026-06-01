[TOC]

# 1 Assignment Overview

作业 5 的主题是 alignment and reasoning RL。主线任务是让语言模型在数学题上进行 step-by-step reasoning，并用 verified reward 衡量最终答案是否正确。

本作业主要实现四件事：

- Zero-shot prompting baseline：评估 Qwen 2.5 Math 1.5B Base 在 MATH 上的基线表现。
- Supervised finetuning：用 DeepSeek R1 reasoning traces 做 SFT。
- Expert Iteration：模型自己采样 reasoning traces，只保留正确答案继续 SFT。
- GRPO：用 group-relative policy gradient 和 verified reward 做 reasoning RL。

官方仓库是 `https://github.com/stanford-cs336/assignment5-alignment`。当前 GitHub `main` 对应 Spring 2026，本地使用 `spring2025` tag。课程集群 MATH 数据在 `/data/a5-alignment/MATH`，本机没有该目录；因此自学版实验用仓库自带 GSM8K 数据替代 MATH，主要保证流程和测试能够跑通。

# 2 Reasoning with Language Models

## 2.1 Motivation

前几个作业主要用 next-token prediction 和 cross-entropy 训练/评估模型。本作业转向 downstream evaluation：数学推理是否做对。模型使用 Qwen 2.5 Math 1.5B Base，评估指标主要是答案正确率，而不是语言模型 loss。

这个设置和前面作业有两个变化：

- 模型不再使用自己从头训练的小模型，因为它们数学推理能力太弱；改用已经有数学预训练的 Qwen 2.5 Math 1.5B Base。
- 评估不再只看 cross-entropy，而是抽取最终答案，与 reference answer 做等价性判断。

## 2.2 Chain-of-Thought Reasoning and Reasoning RL

Chain-of-Thought 的核心是让模型先生成中间推理步骤，再输出最终答案。本作业使用 R1-Zero prompt，要求模型输出：

```text
<think> reasoning process here </think> <answer> answer here </answer>
```

这样做有两个好处：

- `<think>` 里的 reasoning trace 可以作为 SFT 或 Expert Iteration 的训练目标。
- `<answer>` 里的最终答案容易被 parser 抽取并和 ground truth 比较。

Expert Iteration 的思路是：模型先采样很多 reasoning traces，只保留答案正确的 traces，再用这些 traces 做 SFT。Reasoning RL 则直接用 verified reward 训练策略；数学题里 reward 可以来自答案验证，不需要人工偏好标注。

本机没有课程集群的 MATH 数据，所以自学版选择 GSM8K。GSM8K 比 MATH 简单，但仍是自然语言数学推理题，而且标准答案在 `####` 后，方便抽取 short answer。

# 3 Measuring Zero-Shot MATH Performance

本节先测量 base model 的 zero-shot 表现，作为后续 SFT、Expert Iteration 和 GRPO 的 baseline。

## 3.1 Using vLLM for offline language model inference

后续评估和 RL rollout 需要大量生成，所以使用 vLLM 做 batched offline inference。vLLM 的优势主要是 PagedAttention、KV cache 管理和优化后的 CUDA kernel，能提高吞吐并降低显存压力。

本地模型路径：

```text
models/Qwen/Qwen2___5-Math-1___5B
```

课程集群模型路径：

```text
/data/a5-alignment/models/Qwen2.5-Math-1.5B
```

## 3.2 Zero-shot MATH Baseline

**问题：math_baseline**

**(a)**

实现 zero-shot evaluation 脚本。脚本需要做五件事：读取验证集、用 R1-Zero prompt 格式化 prompt、调用 vLLM 生成、计算 reward/metrics、把 examples/generations/scores 序列化到磁盘。

本机没有 MATH，因此脚本同时支持 MATH 和 GSM8K：

- `--dataset-format {auto,math,gsm8k}` 控制数据格式。
- `--reward-fn {auto,r1_zero,numeric}` 控制 reward 函数。
- MATH 下使用 `r1_zero_reward_fn`。
- GSM8K 下使用轻量 numeric reward：仍检查 `<think>` / `<answer>` 格式，但答案正确性比较 `<answer>` 中最后一个数字与 GSM8K `####` 后的短答案。

实现位置：

```text
assignment5-alignment/cs336_alignment/evaluation.py
assignment5-alignment/scripts/math_baseline.py
```

**(b)**

在完整 GSM8K test split（1319 条）上的运行结果：

| 指标 | 数值 |
| --- | ---: |
| `format_reward` | `0.2691` |
| `answer_reward` / accuracy | `0.1107` |
| `reward` | `0.1107` |

三类 generation 数量：

| 类别 | 数量 |
| --- | ---: |
| `format_1_answer_1` | `146` |
| `format_1_answer_0` | `209` |
| `format_0_answer_0` | `964` |

`format_0_answer_0` 中多数是 base model 没有严格输出 `<answer>` 标签，或者在 `<answer>` 外继续输出自然语言/代码块。这主要是 base model 没有完全遵守 R1-Zero 格式，不是 parser 本身的问题。

`format_1_answer_0` 表示格式满足但答案错误。这类样本说明 parser 可以抽取答案，但模型推理或算术过程错了；它们反映的是 reasoning accuracy 问题。

Qwen 实际生成里常出现 `</think>\n<answer>`，所以 numeric reward 的正则允许标签之间有换行和空白。

**(c)**

Qwen 2.5 Math 1.5B Base 在 GSM8K zero-shot 上 accuracy 约为 `11.07%`，格式符合率约为 `26.9%`。这个结果作为后续 SFT/EI/GRPO 的本地 baseline。

输出文件：

```text
outputs/math_baseline_gsm8k/generations.jsonl
outputs/math_baseline_gsm8k/metrics.json
outputs/math_baseline_gsm8k/category_examples.json
```

运行命令：

```bash
cd assignment5-alignment
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --gpu-memory-utilization 0.60 \
  --output-dir outputs/math_baseline_gsm8k
```

如果在课程集群上有 MATH 数据，可以切回 handout 配置：

```bash
uv run python scripts/math_baseline.py \
  --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data /data/a5-alignment/MATH/validation.jsonl \
  --dataset-format math \
  --reward-fn r1_zero \
  --output-dir outputs/math_baseline_math
```

# 4 Supervised Finetuning for MATH

SFT 用带 reasoning trace 的 `(prompt, response)` 数据训练模型。训练时 response 包含 `<think>...</think><answer>...</answer>`，loss 只在 response tokens 上计算，prompt 不参与 loss。

官方 SFT 数据路径：

```text
/data/a5-alignment/MATH/sft.jsonl
```

本机没有这份数据，所以自学版用 GSM8K 构造 `(prompt, response)`：`question` 放进 R1-Zero prompt，`####` 前作为 reasoning trace，`####` 后作为 `<answer>`。

## 4.1 Using HuggingFace Models

训练时用 HuggingFace `AutoModelForCausalLM` 加载 Qwen 2.5 Math 1.5B。建议设置：

- `torch_dtype=torch.bfloat16`，节省显存。
- `attn_implementation="flash_attention_2"`，加速 attention。
- gradient accumulation，把大 batch 拆成多个 microbatch，累计梯度后再 `optimizer.step()`。

梯度累积时每个 microbatch 的 loss 要除以 `gradient_accumulation_steps`，这样累计后的梯度等价于平均大 batch 梯度。

## 4.2 SFT Helper Methods

**问题：tokenize_prompt_and_output**

对每个 `(prompt, output)` 分别 tokenize，再拼接成 `prompt + output`。返回三个张量：

- `input_ids`：拼接序列去掉最后一个 token。
- `labels`：拼接序列去掉第一个 token，相当于 next-token labels。
- `response_mask`：和 `labels` 对齐，只在 response token 的 label 位置为 `True`。

mask 对齐方式：设 prompt 长度为 `P`、output 长度为 `O`，则 response label 在 `labels` 中的区间是 `[P-1, P+O-2)`。

实现位置：

```text
assignment5-alignment/tests/adapters.py
```

测试情况：官方 fixture 写死课程集群模型路径，本地用 ModelScope Qwen tokenizer 对 snapshot 做等价验证，`input_ids`、`labels`、`response_mask` 与官方 snapshot 一致。

**问题：compute_entropy**

输入 logits 形状为 `(batch_size, sequence_length, vocab_size)`，输出每个 token 位置上的 next-token 熵，形状为 `(batch_size, sequence_length)`。

公式：

```text
H = -sum(p * log p)
```

实现上用 `log_softmax` 和 `exp`：

```text
log_probs = log_softmax(logits)
probs = exp(log_probs)
entropy = -(probs * log_probs).sum(-1)
```

这样比直接 `softmax` 后取 log 更稳定。snapshot 验证通过。

**问题：get_response_log_probs**

从 causal LM 获取每个 label token 的条件 log-prob：

```text
model(input_ids).logits -> log_softmax -> gather(labels)
```

在 causal LM 里，`logits[:, t, :]` 预测 `labels[:, t]`。可选参数 `return_token_entropy=True` 时，同时调用 `compute_entropy` 返回每个位置的 entropy。

测试情况：本地 Qwen fp32 权重与 snapshot 对齐，`log_probs` 最大差约 `0.00006`，`token_entropy` 最大差约 `0.0007`。

**问题：masked_normalize**

对 mask 选中的元素求和，再除以指定常数：

```text
(tensor * mask.float()).sum(dim) / normalize_constant
```

`dim=None` 时对所有元素求和并返回标量。`dim=0/1/-1/None` 四类 snapshot 均通过。

**问题：sft_microbatch_train_step**

单个 SFT microbatch 的 loss 是 response token 的负 log likelihood：

```text
loss = -sum(policy_log_probs * response_mask) / (normalize_constant * batch_size * gradient_accumulation_steps)
```

函数内部调用 `loss.backward()`，梯度会累积到模型参数上，外层循环负责在累计足够 microbatch 后执行 `optimizer.step()`。

测试情况：`test_sft_microbatch_train_step`、`test_sft_microbatch_train_step_normalize`、`test_sft_microbatch_train_step_10_steps` 对应 snapshot 通过。

**问题：log_generations**

训练过程中需要周期性查看模型真实生成，而不是只看 loss。`log_generations` 对给定 prompts 调用 vLLM 生成，并记录：

- prompt。
- response。
- ground truth。
- format reward / answer reward / total reward。
- response length。
- 可选 average token entropy。
- 正确/错误样本的平均 response length。

实现位置：

```text
assignment5-alignment/cs336_alignment/evaluation.py
```

测试情况：`py_compile` 通过；fake vLLM smoke test 验证生成记录、reward 分类、response length 和 entropy 汇总正常。

## 4.3 SFT Experiment

**问题：sft_experiment**

**1. 不同 SFT 数据规模实验**

官方要求在 `/data/a5-alignment/MATH/sft.jsonl` 上训练 Qwen 2.5 Math 1.5B Base，并比较 `{128, 256, 512, 1024, full}` 不同 unique examples 的 validation accuracy 曲线。

本地实现同时支持：

- 官方 `{"prompt": ..., "response": ...}` 的 MATH SFT 格式。
- 自学版 GSM8K 格式，自动构造 R1-Zero prompt 和 `<think>/<answer>` response。

训练设置包括 AdamW、cosine schedule、gradient accumulation、`max_grad_norm=1.0`、`lr=1e-5`、`batch_size=1`、`grad_accum=16`、`max_seq_len=1024`。每个 train size 都从同一个 base model 重新开始训练。

实现位置：

```text
assignment5-alignment/cs336_alignment/sft.py
assignment5-alignment/scripts/sft_experiment.py
```

**(a) SFT 训练结果**

每个 train size 训练 1 epoch，optimizer steps = `num_examples / grad_accum`：

| train_size | examples | opt_steps | final_train_loss | answer_accuracy | format_accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | — | — | — | 11.07% | 26.9% |
| 128 | 128 | 8 | 0.653 | 10.69% | 32.9% |
| 256 | 256 | 16 | 0.627 | 13.65% | 38.1% |
| 512 | 512 | 32 | 0.633 | 14.63% | 44.0% |
| 1024 | 1024 | 64 | 0.575 | 14.33% | 48.9% |
| **full** | **7473** | **467** | **0.430** | **38.7%** | **94.2%** |

趋势：

- `train_loss` 随数据量单调下降：0.65 (128) → 0.43 (7473)。
- `format_accuracy` 单调上升：32.9% → 94.2%。说明 SFT 主要让模型学会了 R1-Zero 格式规范。
- `answer_accuracy` 在小数据量下提升缓慢（11% → 14.6%），但在 `full` 下有显著跳跃至 38.7%。
- 这表明 SFT 需要大量数据才能泛化到未见过的数学题；小数据量主要学到格式，大数据量才能学到推理能力。

小数据量（size=512/1024）下 accuracy 有微小回落（14.6% → 14.3%），属于单次随机 seed 下的正常噪声。

输出位置：

```text
outputs/sft_experiment_gsm8k_4sizes/train_size_{128,256,512,1024}/
outputs/sft_experiment_full/train_size_full/
outputs/eval_sft_size_{128,256,512,1024}/
outputs/eval_sft_full/
```

完整实验命令：

```bash
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/sft_experiment.py \
  --train-sizes 128,256,512,1024 \
  --epochs 1 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --max-sequence-length 1024 \
  --save-model
```

vLLM 评估命令（对已训练的 checkpoint）：

```bash
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model outputs/sft_experiment_gsm8k_4sizes/train_size_{SIZE}/checkpoint \
  --data data/gsm8k/test.jsonl \
  --dataset-format gsm8k \
  --output-dir outputs/eval_sft_size_{SIZE}
```

**2. 过滤正确 reasoning traces 后再训练（`--filter-correct`）**

训练前用 reward 函数检查每条 SFT 样本的 response 是否答案正确，只保留正确的样本做 SFT。

GSM8K 自学版的 response 由标准解答构造，所有 response 答案都正确，因此过滤后保留 7473/7473 全部样本。`filter-correct` 与普通 full SFT 训练结果几乎一致：

| 实验 | train_loss | answer_accuracy | format_accuracy |
| --- | ---: | ---: | ---: |
| SFT full | 0.430 | 38.7% | 94.2% |
| SFT full + filter-correct | 0.429 | 38.7% | 95.0% |

两组的 loss 和 accuracy 在统计误差范围内完全一致，确认 GSM8K 的 SFT response 没有错误答案需要过滤。

输出位置：

```text
outputs/sft_experiment_filtered/train_size_full/
outputs/eval_sft_filtered/
```

# 5 Expert Iteration for MATH

Expert Iteration 不再依赖外部 reasoning traces，而是让当前模型自己生成。每一轮流程是：

```text
rollout -> reward/filter -> SFT on correct traces -> eval
```

如果 base model 偶尔能采样出正确答案，就可以把这些正确 traces 作为新的 SFT 数据，自举提升推理能力。

**问题：expert_iteration_experiment**

PDF 要求在 MATH train 上运行 `n_ei_steps=5`，变化 rollout 数 `G`、问题 batch size `Db ∈ {512,1024,2048}` 和 SFT epoch 数，并观察 validation accuracy 与 response entropy。

本地实现复用：

- `sft.py`：做 SFT update。
- `evaluation.py`：做 vLLM 生成和 reward 评估。
- `math_baseline.py` 的 prompt/reward/data 处理逻辑。

实现位置：

```text
assignment5-alignment/scripts/expert_iteration_experiment.py
```

**(a) 完整 EI 实验结果（GSM8K）**

参数：`n_ei_steps=5`，`G=4`，`question_batch_sizes=[512, 1024, 2048]`，`sft_epochs=1`，`lr=1e-5`，`warmup_ratio=0.03`。从 base model（zero-shot accuracy 11.07%）开始。

每步流程：对 batch_size 个问题各做 G=4 次 vLLM rollout → 过滤出答案正确的 response → 用这些 response 做 1 epoch SFT → 在 1319 条 GSM8K test 上评估。

| ei_step | batch_size | answer_accuracy | format_accuracy | 正确+格式对 | 格式对答案错 | 格式错 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | — | 11.07% | 26.9% | 146 | 209 | 964 |
| 1 | 512 | 16.83% | 44.0% | 222 | 359 | 738 |
| 2 | 1024 | 30.48% | 66.3% | 402 | 473 | 444 |
| 3 | 2048 | 41.39% | 83.3% | 546 | 553 | 220 |
| 4 | 2048 | 49.20% | 90.1% | 649 | 540 | 130 |
| **5** | **2048** | **49.66%** | **93.1%** | **655** | **573** | **91** |

关键观察：

1. **单调提升**：accuracy 从 16.8%（第 1 步）→ 49.7%（第 5 步），每个 EI step 都带来显著增益。
2. **SFT vs EI 对比**：纯 SFT full（38.7%）被 EI 5 步（49.7%）超越约 11 个百分点。迭代式 "用自己的正确 rollout 再训练" 比单次 SFT 更有效。
3. **format 快速收敛**：格式率从 44.0% → 93.1%，接近 SFT full 的 94.2%。
4. **第 1 步起点低**（16.8%）：因为 base model 只有 ~11% 准确率，G=4 时平均每题只有 ~0.5 个正确 response，可用 SFT 数据稀疏。
5. **收敛趋势**：第 3→4→5 步增益递减（41.4% → 49.2% → 49.7%），说明方法接近瓶颈；进一步提高可能需要更大的 G 或更强的 base model。

**(b) 与 SFT 的关系**

EI 从 base model 出发，5 步 EI 超越纯 SFT full（38.7% vs 49.7%）。这验证了文献中的结论：即使 base model 没有外部标注的 reasoning traces，只要能偶尔采样出正确答案，就可以通过 EI 自举。

GSM8K 的 EI smoke test 也已完成，用 `n_ei_steps=1, Db=2, G=1` 验证 rollout → filter → SFT → eval 流程跑通。

# 6 Primer on Policy Gradients

本节是 GRPO 前的 RL 背景，不是实现题为主。

## 6.1 Language Models as Policies

在 RL 视角里，语言模型就是一个 categorical policy：当前 prefix 是 state，下一个 token 是 action。

```text
a_t ~ pi_theta(. | s_t)
```

训练时需要两个基本操作：

- 从 policy 采样 token / response。
- 计算某个 token action 的 `log pi_theta(a_t | s_t)`。

## 6.2 Trajectories

一个 response 可以看作一条 trajectory：

```text
tau = (s_0, a_0, s_1, a_1, ..., s_T, a_T)
```

在语言模型里，环境转移是确定的：新 state 就是旧 prefix 拼上新 token。

## 6.3 Rewards and Return

数学推理任务通常只有 terminal reward：中间 token reward 为 0，生成结束后根据最终答案给 `0/1` reward。

本作业使用 undiscounted return，因为 response 有自然终止点（`</answer>` 或最大生成长度）。

## 6.4 Vanilla Policy Gradient

REINFORCE 的核心梯度：

```text
grad J(theta) = E_tau [ sum_t grad log pi_theta(a_t | s_t) * R(tau) ]
```

直觉是：高 reward trajectory 中的 token 会被提高概率，低 reward trajectory 中的 token 会被降低概率。

## 6.5 Policy Gradient Baselines

baseline 用来降低梯度估计方差。只要 baseline 只依赖 state，不依赖 action，就不会引入 bias。

实现时的 policy gradient loss 不是普通意义上的 validation loss；它只是为了让 `backward()` 产生对应梯度。RL 训练中真正应该报告的是 train/validation reward。

## 6.6 Off-Policy Policy Gradient

on-policy 每采样一批 rollouts 只做一次更新，理论干净但效率低。off-policy 用旧策略 `pi_theta_old` 采样的数据来更新当前策略 `pi_theta`，需要 importance ratio：

```text
pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

GRPO-Clip 里的 clipping 就是为了限制当前策略不要离旧策略太远。

# 7 Group Relative Policy Optimization

## 7.1 GRPO Algorithm

GRPO 的关键是对同一个问题采样 `G` 个 responses，用组内 reward 计算 advantage，而不是训练一个 value model。

给定同一个 question 的 rewards：

```text
r_1, r_2, ..., r_G
```

标准 GRPO advantage：

```text
A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)
```

Dr. GRPO 提出可以不除以组内标准差：

```text
A_i = r_i - mean(r_1, ..., r_G)
```

GRPO-Clip 的 per-token objective 和 PPO 类似，用 ratio 和 clipping 限制更新幅度。

## 7.2 Implementation

**问题：compute_group_normalized_rewards**

先对每个 rollout response 调用 reward function，得到 raw reward；再按 `group_size` 切成每题一组，在组内做归一化。

实现支持两种模式：

- `normalize_by_std=True`：`(reward - group_mean) / (group_std + advantage_eps)`。
- `normalize_by_std=False`：`reward - group_mean`。

返回值：

- `advantages`：组归一化后的 rewards。
- `raw_rewards`：未归一化 rewards。
- `metadata`：raw reward 的 mean/std/min/max。

实现位置：

```text
assignment5-alignment/tests/adapters.py
```

**问题：compute_naive_policy_gradient_loss**

朴素 policy gradient per-token loss：

```text
loss_t = -A * log p_theta(o_t | q, o_<t)
```

这里的 `A` 可以是 raw reward，也可以是已经归一化的 advantage。实现上利用 broadcasting，把 `(batch_size, 1)` 的 reward/advantage 扩展到 `(batch_size, sequence_length)`。

**问题：compute_grpo_clip_loss**

GRPO-Clip per-token loss：

```text
ratio = exp(policy_log_probs - old_log_probs)
loss = -min(advantages * ratio, advantages * clamp(ratio, 1-eps, 1+eps))
```

metadata 记录 `clip_fraction`，用于观察有多少 token 的 ratio 被 clipping 影响。

**问题：compute_policy_gradient_loss**

这是 policy gradient loss wrapper，根据 `loss_type` 分发到不同实现：

| `loss_type` | 使用的信号 | loss |
| --- | --- | --- |
| `no_baseline` | `raw_rewards` | naive PG |
| `reinforce_with_baseline` | `advantages` | naive PG |
| `grpo_clip` | `advantages` + `old_log_probs` | GRPO-Clip |

这个 wrapper 方便后续在实验里切换 baseline 和 clipping 设置。

**问题：masked_mean**

只对 mask 为 1 的元素求均值：

```text
(tensor * mask).sum(dim) / mask.sum(dim)
```

`dim=None` 时对所有被 mask 选中的元素求全局均值。这个函数主要用于把 per-token loss 聚合成 per-sample loss，或者统计 response token entropy。

**问题：grpo_microbatch_train_step**

单个 GRPO microbatch 的流程：

```text
policy_log_probs + rewards/advantages -> per-token PG loss
per-token loss + response_mask -> per-sample mean loss
batch mean / gradient_accumulation_steps -> backward
```

当前实现用 `masked_mean` 风格，即每条 response 内先除以 response length，再对 batch 求均值。

**问题：grpo_train_loop**

PDF 要求实现完整 GRPO train loop，包括 rollout、reward、advantage、policy update、定期 validation 和 rollout logging。目前：

- **helper 函数已全部实现**：`compute_group_normalized_rewards`、`compute_naive_policy_gradient_loss`、`compute_grpo_clip_loss`、`compute_policy_gradient_loss`、`masked_mean`、`grpo_microbatch_train_step` 共 6 个函数均通过 snapshot 测试。
- **GRPO train loop 脚本尚未完成**：`tests/adapters.py` 中的 `run_iterate_batches` 等函数也尚未实现（不影响 helper 测试）。

# 8 GRPO Experiments

本节是完整 GRPO train loop 之后的实验题。**本地所有 GRPO 实验尚未运行**，以下记录各问题的状态和预备知识。

**问题：grpo_learning_rate**（未完成）

需要从默认 GRPO 超参数出发 sweep learning rate，并报告多条 validation reward curves。目标是在 MATH 上达到至少 `25%` validation accuracy。当前现状：`compute_policy_gradient_loss` 支持 `no_baseline` / `reinforce_with_baseline` / `grpo_clip` 三种模式，`grpo_microbatch_train_step` 已实现。待 GRPO train loop 脚本完成后可跑。

**问题：grpo_baselines**（未完成）

需要比较 `no_baseline` 和 `reinforce_with_baseline`。实现上对应 `compute_policy_gradient_loss` 的两个分支；待 GRPO train loop 完成后可跑。

**问题：think_about_length_normalization**

两种长度归一化方式的区别（已阐述，不做实验）：

- `masked_mean`：每条 response 内按自己的长度求平均，短 response 的单个 token 梯度更大，能避免长 response 因 token 多而天然权重大。
- `masked_normalize`：对 response token 求和后除以固定常数，所有 token 的尺度一致，长 response 总梯度更大，更接近未按轨迹长度平均的 policy gradient。

如果希望每个样本贡献相近，`masked_mean` 更稳；如果认为更长的 reasoning trace 中每个 action 都应该贡献梯度，固定常数归一化更自然，但可能导致长 response 梯度更大、训练更不稳定。

**问题：grpo_length_normalization**（未完成）

需要端到端比较 `masked_mean` 和 `masked_normalize` 的 validation answer reward curves。

**问题：grpo_group_standard_deviation**（未完成）

需要比较 `use_std_normalization=True` 和 `False`。helper 已支持两种模式。

**问题：grpo_off_policy**（未完成）

需要实现 off-policy GRPO：对同一 rollout batch 做多个 epoch/多个 optimizer updates，并在 rollout 后缓存 `old_log_probs`。

**问题：grpo_off_policy_sweep**（未完成）

需要固定 `rollout_batch_size=256`，sweep `epochs_per_rollout_batch` 和 `train_batch_size`。

**问题：grpo_off_policy_clip_ablation**（未完成）

需要新增 `GRPO-No-Clip` loss，并和 GRPO-Clip 比较 entropy、response length、gradient norm 等指标。

**问题：grpo_prompt_ablation**（未完成）

需要比较 R1-Zero prompt 和 `question_only.prompt`，并切换到 `question_only_reward_fn`。

# 9 Leaderboard: GRPO on MATH

**问题：leaderboard**

Leaderboard 要求在 2 张 H100、4 小时训练预算内取得尽可能高的 MATH validation accuracy。评估约束是：

- validation accuracy 必须在完整 MATH validation set（约 5K）上取平均。
- validation 必须使用 R1-Zero prompt。
- vLLM generation 必须使用 `temperature=1.0`、`max_tokens=1024`。
- accuracy 必须用 starter code 里的 `r1_zero_reward_fn` 计算。

本机没有 MATH 数据和 H100 环境，因此未运行 leaderboard 实验。

# 10 Epilogue

本作业把前面语言模型训练的基础组件连接到 post-training：先用 zero-shot 建 baseline，再用 SFT 学 reasoning trace，然后用 EI 只依赖 verified reward 改善 reasoning。

## 完成状态总览

| Section | 内容 | 状态 |
| --- | --- | --- |
| 3.2 | Zero-shot MATH Baseline | ✅ GSM8K baseline accuracy 11.07% |
| 4.2 | SFT Helper Methods (6 个) | ✅ 全部 snapshot 测试通过 |
| 4.3 | SFT Experiment ({128,...,full}) | ✅ 完整训练 + vLLM 评估 |
| 4.3 | SFT + filter-correct | ✅ GSM8K 全量数据 filter=0 过滤 |
| 5 | Expert Iteration (5 steps) | ✅ accuracy 16.8% → 49.7% |
| 7.2 | GRPO Helper Methods (6 个) | ✅ 全部 snapshot 测试通过 |
| 7.2 | GRPO train loop 脚本 | ⏳ 未实现 |
| 8 | GRPO 实验 (全部) | ⏳ 未运行 |
| 9 | Leaderboard | ⏳ 未运行（需 MATH + H100） |

## 核心发现

1. **SFT 数据量是关键**：128 样本提升 format，7473 样本才显著提升 accuracy（11% → 38.7%）。
2. **EI 自举有效**：5 步 EI 将 accuracy 从 base 11% 提升至 49.7%，超越纯 SFT（38.7%）约 11 个百分点。
3. **format 先收敛**：SFT 和 EI 都观察到 format accuracy 先于 answer accuracy 收敛，说明模型先学会格式规范，再学会正确推理。
4. **GRPO 基础设施就绪**：所有 helper 函数已实现并通过 snapshot 测试，train loop 完成后即可运行 GRPO 实验。
