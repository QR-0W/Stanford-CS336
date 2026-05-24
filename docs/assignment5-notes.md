[TOC]

# 1 Assignment Overview

作业 5 的主题是 alignment and reasoning RL，目标是训练语言模型在数学题上进行 step-by-step reasoning。主 PDF 的必做部分围绕 MATH 数学推理展开，依次包括 zero-shot prompting baseline、SFT、Expert Iteration、GRPO，以及最后的 leaderboard 实验。

官方代码仓库是：

```text
https://github.com/stanford-cs336/assignment5-alignment
```

我查了官方 GitHub。当前 `main` 分支已经是 Spring 2026 版本，`spring2025` tag / release 是 2025 版本存档。我们本地的 `assignment5-alignment/` 对应 Spring 2025 handout。

GitHub 仓库可以获取到代码、tests、prompts、PDF handout、PDF supplement，以及 `data/alpaca_eval/`、`data/gsm8k/`、`data/mmlu/`、`data/simple_safety_tests/` 这些公开数据。主 PDF 必做 reasoning 部分用到的 MATH 数据不在 GitHub 仓库里，而是在课程集群共享路径里。

作业要求提交 `writeup.pdf` 和 `code.zip`。测试方面，主线只要求通过 `tests/test_sft.py` 和 `tests/test_grpo.py`；其他测试主要是 optional supplement 用的。

# 2 Reasoning with Language Models

## 2.1 Motivation

作业 5 的重点从普通语言模型训练转到 reasoning。前几个作业主要看 next-token prediction 和 cross-entropy，这里开始直接看模型能不能把数学题做对。

这部分使用的模型是：

```text
Qwen 2.5 Math 1.5B Base
```

原因是我们前面自己训练的小模型太弱，基本没有可观察的数学推理能力。评估数据按 handout 是 MATH 12K，指标不是 loss，而是最终答案是否正确。也就是说，作业 5 要研究的是“语言模型作为推理器”，后面的 SFT、Expert Iteration 和 GRPO 都围绕数学题的 verified reward 展开。

## 2.2 Chain-of-Thought Reasoning and Reasoning RL

Chain-of-Thought 的做法是让模型先写推理过程，再输出最终答案。作业使用 R1-Zero 风格 prompt，要求输出格式类似：

```text
<think> reasoning process here </think> <answer> answer here </answer>
```

这个格式有两个作用：`<think>` 里的推理过程可以作为 SFT / Expert Iteration 的训练目标，`<answer>` 里的最终答案可以被 parser 抽出来，与 ground truth 比较。

Expert Iteration 的流程是先让模型自己采样 reasoning traces，然后只保留答案正确的 traces，再拿这些正确 traces 继续 SFT。只要模型偶尔能采样出正确答案，就可以自举出更好的训练数据。

Reasoning RL 则直接用 verified reward 训练模型。数学题可以用答案验证做 reward，代码题可以用 unit tests 做 reward，这和 RLHF 不一样，不依赖人工偏好标注。这里的逻辑是先用 CoT 提供可训练的推理轨迹，再用 Expert Iteration 从模型自己的正确样本中学习，最后用 GRPO 进一步做 RL。

**Our setup: model and dataset**

handout 中的 MATH 路径是：

```text
/data/a5-alignment/MATH
```

这个路径是 Stanford / Together cluster 上预置的共享目录，不是 GitHub 仓库里的文件。当前本机检查结果是 `/data/a5-alignment` 不存在，所以不能按原路径直接跑 MATH。handout 也明确说明 MATH 数据集因为 copyright claim 不是公开可下载数据集；如果没有官方集群权限，就基本不能获取这份课程预置 MATH 数据。

**Tip for Open-Source Auditors: Alternative Datasets**

PDF 建议在无法访问 MATH 时使用开源数学推理数据集替代，包括 Countdown、GSM8K、Tulu 3 SFT Math，或者其他 math SFT dataset。如果替代数据没有短答案字段，可以用 Math-Verify 之类的 parser 从 ground-truth 解答里抽取短答案。

我认为当前自学版最合适的是先用 GSM8K。它已经在官方仓库的 `data/gsm8k/` 里，不需要额外找数据；题目也是自然语言数学推理，比 Countdown 更接近 MATH；每条样本都有标准解答，最终短答案在 `####` 后面，容易抽取。后续 zero-shot baseline、SFT 和 GRPO 都可以先围绕 GSM8K 跑通。

GSM8K 样本格式示例：

```json
{"question": "Natalia sold clips to 48 of her friends...", "answer": "...\n#### 72"}
```

后续需要把它转成评估脚本使用的短答案格式：

```json
{"question": "Natalia sold clips to 48 of her friends...", "ground_truth": "72"}
```

自学版暂时不需要全都用。主线先用 GSM8K，必要时再用 Countdown 做 GRPO sanity check；Tulu 3 SFT Math 可以等主流程跑通后再考虑。

# 3 Measuring Zero-Shot MATH Performance

本节先测量 base model 在 MATH validation set 上的 zero-shot 表现，作为后续 SFT、Expert Iteration 和 GRPO 的 baseline。除非特别说明，MATH 实验都使用 DeepSeek R1-Zero 风格 prompt，prompt 文件在：

```text
assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt
```

这个 prompt 会要求模型从 `Assistant: <think>` 后继续生成推理过程，之后关闭 `</think>`，再在 `<answer> ... </answer>` 中给出最终答案。使用 answer tags 的目的，是让 reward function 更容易解析最终答案，并且让 vLLM 可以在生成到 `</answer>` 时停止。

PDF 也提醒了一个 prompt mismatch：`r1_zero` prompt 不一定是 Qwen 2.5 Math 1.5B 的最佳 prompt，因为 Qwen 可能已经在 question-only 风格数据上预训练过。作业仍然选择 R1-Zero prompt，是因为它更方便观察短步数 RL 的改进，并且后面会要求和 `question_only.prompt` 做对比。

## 3.1 Using vLLM for offline language model inference

作业建议使用 vLLM 做 offline batched inference，而不是自己手写 generation。原因是后续评估和 RL rollout 都需要大量生成，vLLM 的 PagedAttention、KV cache 管理和 CUDA kernel 优化能显著提高吞吐。

handout 中的课程集群模型路径是：

```text
/data/a5-alignment/models/Qwen2.5-Math-1.5B
/data/a5-alignment/models/Llama-3.1-8B
/data/a5-alignment/models/Llama-3.3-70B-Instruct
```

这些是集群预下载模型路径，本机不存在。模型本身可以用 Hugging Face model id、ModelScope 或本地下载路径替代。当前 `Conf_Test` conda 环境里检测到了 vLLM：

```text
vllm == 0.18.1
python == /mdata/wjx/miniconda3/envs/Conf_Test/bin/python
```

因此后续实际跑 baseline 时应优先使用 `Conf_Test` 环境，而不是 `coding` 环境。

我检查了 ModelScope，`Qwen/Qwen2.5-Math-1.5B` 存在，并已下载到本地：

```text
assignment5-alignment/models/Qwen/Qwen2___5-Math-1___5B
```

实际权重目录大小约为 `2.9G`，`AutoTokenizer.from_pretrained(...)` 可以正常加载。

## 3.2 Zero-shot MATH Baseline

**问题：math_baseline**

本题要求写一个脚本评估 `Qwen 2.5 Math 1.5B Base` 在 MATH validation split 上的 zero-shot performance。脚本需要加载 `/data/a5-alignment/MATH/validation.jsonl`，用 R1-Zero prompt 格式化每道题，调用 vLLM 生成答案，使用 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` 计算 reward，并把 examples、model generations 和 evaluation scores 序列化到磁盘，供后续人工分析。

由于本机没有课程集群的 MATH 数据，本题实际采用 handout 推荐的开源替代数据集 GSM8K。脚本现在默认读取 `assignment5-alignment/data/gsm8k/test.jsonl`，并从 GSM8K `answer` 字段的 `####` 后抽取短答案作为 `ground_truth`。GSM8K 自学版默认使用轻量 numeric reward：仍然检查 R1-Zero 标签格式，但答案正确性只比较 `<answer>` 中最后一个数字和 GSM8K 短答案。

对应实现文件：

```text
assignment5-alignment/cs336_alignment/evaluation.py
assignment5-alignment/scripts/math_baseline.py
```

当前自学版默认配置：

```text
model = models/Qwen/Qwen2___5-Math-1___5B
data = data/gsm8k/test.jsonl
dataset_format = gsm8k
reward_fn = auto  # GSM8K 下自动使用 numeric reward
prompt = cs336_alignment/prompts/r1_zero.prompt
temperature = 1.0
top_p = 1.0
max_tokens = 1024
stop = </answer>
include_stop_str_in_output = True
```

这里要注意 `include_stop_str_in_output=True`。官方 MATH 的 `r1_zero_reward_fn` 会严格检查 response 中是否包含：

```text
</think> <answer>
</answer>
```

如果 vLLM 把 `</answer>` 这个 stop string 去掉，格式分可能会被错误打成 0。GSM8K 自学版的 numeric reward 沿用同样的标签语义，但允许 `</think>` 和 `<answer>` 之间出现换行或多个空格，因为 Qwen 在实际生成中经常输出 `</think>\n<answer>`。

脚本输出：

```text
outputs/math_baseline_gsm8k/generations.jsonl
outputs/math_baseline_gsm8k/metrics.json
outputs/math_baseline_gsm8k/category_examples.json
```

其中三类统计对应 handout 要求：

| 类别 | 含义 |
| --- | --- |
| `format_1_answer_1` | 格式正确且答案正确 |
| `format_1_answer_0` | 格式正确但答案错误 |
| `format_0_answer_0` | 格式错误且答案错误 |

完整运行命令：

```bash
cd /mdata/wjx/CS336/assignment5-alignment
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --gpu-memory-utilization 0.60 \
  --output-dir outputs/math_baseline_gsm8k
```

小规模 smoke test：

```bash
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --limit 16
```

如果以后在课程集群上有原始 MATH 数据，可以显式切回 handout 配置：

```bash
uv run python scripts/math_baseline.py \
  --model /data/a5-alignment/models/Qwen2.5-Math-1.5B \
  --data /data/a5-alignment/MATH/validation.jsonl \
  --dataset-format math \
  --reward-fn r1_zero \
  --output-dir outputs/math_baseline_math
```

当前验证：

```bash
/mdata/wjx/miniconda3/bin/conda run -n coding python -m py_compile \
  assignment5-alignment/cs336_alignment/evaluation.py \
  assignment5-alignment/scripts/math_baseline.py

/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py --help
```

结果：语法检查和 CLI help 均通过。此外，用 fake vLLM 做了无 GPU smoke test，确认 GSM8K 短答案抽取、prompt 构造、generation 评分、category count 汇总和 JSON/JSONL 输出流程都能跑通。

真实 vLLM 小样本：

```bash
/mdata/wjx/miniconda3/bin/conda run -n Conf_Test python scripts/math_baseline.py \
  --model models/Qwen/Qwen2___5-Math-1___5B \
  --limit 4 \
  --gpu-memory-utilization 0.40 \
  --output-dir outputs/math_baseline_gsm8k_smoke
```

下载 ModelScope 权重后，这一步可以正常完成。前 4 条的结果为：`format_1_answer_1=1`、`format_1_answer_0=1`、`format_0_answer_0=2`。

完整 GSM8K test split 已运行完成，共 `1319` 条样本：

| 指标 | 数值 |
| --- | ---: |
| `format_reward` | `0.2691` |
| `answer_reward` / accuracy | `0.1107` |
| `reward` | `0.1107` |

三类输出数量：

| 类别 | 数量 |
| --- | ---: |
| `format_1_answer_1` | `146` |
| `format_1_answer_0` | `209` |
| `format_0_answer_0` | `964` |

输出文件位于：

```text
assignment5-alignment/outputs/math_baseline_gsm8k/generations.jsonl
assignment5-alignment/outputs/math_baseline_gsm8k/metrics.json
assignment5-alignment/outputs/math_baseline_gsm8k/category_examples.json
```

从 `category_examples.json` 看，`format_0_answer_0` 中有不少是模型没有严格输出 `<answer>` 标签，或者生成了 `<answer>` 之外的自然语言 / 代码块 / 其他标签；`format_1_answer_0` 则主要是格式满足但算错或最终数字不等于 GSM8K 的短答案。由于这里使用的是 GSM8K 自学版 numeric reward，而不是官方 MATH parser，结果不能和 handout leaderboard 直接比较，但可以作为后续 SFT / GRPO 的 baseline。
