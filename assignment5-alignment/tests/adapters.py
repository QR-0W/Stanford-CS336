from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """分别 tokenize prompt 和 output，并构造只覆盖 response 标签位置的 mask。

    Args:
        prompt_strs: prompt 字符串列表。
        output_strs: output / response 字符串列表。
        tokenizer: 用于 tokenization 的 HuggingFace tokenizer。

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define either pad_token_id or eos_token_id")

    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]

    prompt_and_output_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids)
    ]
    max_sequence_length = max(len(token_ids) for token_ids in prompt_and_output_ids)

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []
    for prompt_ids, output_ids, token_ids in zip(
        prompt_token_ids, output_token_ids, prompt_and_output_ids
    ):
        padded_token_ids = token_ids + [pad_token_id] * (max_sequence_length - len(token_ids))

        # labels 是右移一位后的目标 token；response_mask 因此也要按 labels 的位置对齐。
        input_ids.append(padded_token_ids[:-1])
        labels.append(padded_token_ids[1:])
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        response_mask.append(
            [
                prompt_len - 1 <= label_position < prompt_len + output_len - 1
                for label_position in range(max_sequence_length - 1)
            ]
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """对每组 rollout 响应计算 raw reward 并按组归一化。

    两组归一化模式：
    - normalize_by_std=True（DeepSeekMath/DeepSeek-R1）：
      A(i) = (r(i) - mean_group) / (std_group + advantage_eps)，std 使用 ddof=1。
    - normalize_by_std=False（Dr. GRPO）：
      A(i) = r(i) - mean_group。

    Args:
        reward_fn: 接收 (response, ground_truth) 返回含 "reward" 键的 dict。
        rollout_responses: 长度 = n_prompts × group_size。
        repeated_ground_truths: 同长度，每题 ground truth 重复 group_size 次。
        group_size: 每题的 rollout 数。
        advantage_eps: 防止除零的小常数。
        normalize_by_std: 是否除以组内标准差。

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages: shape (rollout_batch_size,) 组归一化后的 advantage。
            raw_rewards: shape (rollout_batch_size,) 原始 reward。
            metadata: 均值/方差等统计量。
    """
    raw_rewards_list: list[float] = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        raw_rewards_list.append(float(scores["reward"]))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    total = len(raw_rewards_list)
    n_groups = total // group_size
    advantages = torch.empty_like(raw_rewards)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        group_mean = group_rewards.mean()
        if normalize_by_std:
            group_std = group_rewards.std(unbiased=True)
            advantages[start:end] = (group_rewards - group_mean) / (group_std + advantage_eps)
        else:
            advantages[start:end] = group_rewards - group_mean

    metadata = {
        "raw_reward_mean": float(raw_rewards.mean().item()),
        "raw_reward_std": float(raw_rewards.std(unbiased=True).item()),
        "raw_reward_min": float(raw_rewards.min().item()),
        "raw_reward_max": float(raw_rewards.max().item()),
    }
    return advantages, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算每个 token 位置上 next-token 预测的熵，即对词表维度求离散熵。

    Args:
        logits: 未归一化的 logits，shape (batch_size, sequence_length, vocab_size)。

    Returns:
        torch.Tensor shape (batch_size, sequence_length)，每个位置的信息熵。
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """获取给定 prefix 下每个 label token 的条件 log-prob，并可选返回 next-token 熵。

    causal LM 中 logits[:, t, :] 预测的是 token_{t+1}，即 labels[:, t]。
    因此对每个位置 t，从 logits[:, t, :] 中取 labels[:, t] 对应词表的 log-prob。

    Args:
        model: HuggingFace 模型（已放置在正确设备上）。
        input_ids: shape (batch_size, sequence_length)，拼接后的 prompt + response token。
        labels: shape (batch_size, sequence_length)，右移一位后的目标 token。
        return_token_entropy: 是否同时返回 per-token 熵。

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": shape (batch_size, sequence_length)，各位置的条件 log p(labels|prefix)。
            "token_entropy" (optional): shape (batch_size, sequence_length)，per-token 熵。
    """
    logits = model(input_ids).logits
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_size, sequence_length, _ = log_probs_all.shape

    # 展平 batch/sequence 维度后，每一行对应一个位置的 vocab 分布；labels 给出该行要取的真实 token id。
    flat_log_probs = rearrange(log_probs_all, "batch sequence vocab -> (batch sequence) vocab")
    flat_labels = rearrange(labels, "batch sequence -> (batch sequence)")
    flat_positions = torch.arange(flat_labels.numel(), device=flat_labels.device)
    flat_selected_log_probs = flat_log_probs[flat_positions, flat_labels]
    log_probs = rearrange(
        flat_selected_log_probs,
        "(batch sequence) -> batch sequence",
        batch=batch_size,
        sequence=sequence_length,
    )

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = run_compute_entropy(logits)
    return result


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """朴素策略梯度 loss：L_t = -A × log p_θ(o_t | q, o_<t)。

    raw_rewards_or_advantages (batch_size, 1) 会自动广播到 (batch_size, sequence_length)。

    Args:
        raw_rewards_or_advantages: shape (batch_size, 1)，reward 或 advantage。
        policy_log_probs: shape (batch_size, sequence_length)，per-token log-probs。

    Returns:
        torch.Tensor shape (batch_size, sequence_length)，per-token loss。
    """
    return -raw_rewards_or_advantages * policy_log_probs


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-Clip loss。

    ratio = exp(policy_log_probs - old_log_probs)
    L = -min(A × ratio, A × clip(ratio, 1-ε, 1+ε))

    Args:
        advantages: shape (batch_size, 1)，per-sample advantage。
        policy_log_probs: shape (batch_size, sequence_length)，当前策略 log-probs。
        old_log_probs: 同 shape，旧策略 log-probs。
        cliprange: 裁剪范围 ε。

    Returns:
        (per-token loss, metadata with clip fraction)
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss_unclipped = advantages * ratio
    loss_clipped = advantages * clipped_ratio
    loss = -torch.min(loss_unclipped, loss_clipped)
    clip_fraction = (loss_unclipped != loss_clipped).float().mean()
    return loss, {"clip_fraction": clip_fraction.detach()}


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """根据 loss_type 委托到对应的策略梯度 loss 函数。

    - "no_baseline": 使用 raw_rewards 的 naive PG loss。
    - "reinforce_with_baseline": 使用 advantages 的 naive PG loss。
    - "grpo_clip": 使用 advantages + old_log_probs + cliprange 的 GRPO-Clip loss。
    - "grpo_no_clip": 使用 advantages + old_log_probs 的 GRPO-No-Clip loss（无裁剪）。
    """
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_no_clip":
        return run_compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)
    raise ValueError(f"Unknown loss_type: {loss_type}")

def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """对被 mask 选中的元素沿指定维度求均值。

    Args:
        tensor: 输入张量。
        mask: 与 tensor 同 shape 的 bool 张量。
        dim: 沿哪个维度求均值；None 对所有被 mask 选中的元素求全局均值。

    Returns:
        torch.Tensor，mask=1 位置的均值。
    """
    masked = tensor * mask.float()
    count = mask.float().sum(dim=dim)
    return masked.sum(dim=dim) / count

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行一个 SFT microbatch 的前向+反向传播。

    SFT loss 是负对数似然：L = -Σ(log_prob_i * mask_i) / (norm_const * B * grad_accum_steps)，
    其中 B 是 batch_size。

    Args:
        policy_log_probs: shape (batch_size, sequence_length)，per-token log-probs。
        response_mask: 同 shape 的 bool 张量，response 位置为 True。
        gradient_accumulation_steps: 每 optimizer step 累计的 microbatch 数。
        normalize_constant: 归一化分母常数。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: 标量，调整梯度累积后的 microbatch loss。
            metadata: 额外统计字典。
    """
    masked_sum = (policy_log_probs * response_mask.float()).sum()
    batch_size = policy_log_probs.shape[0]
    loss = -masked_sum / (normalize_constant * batch_size * gradient_accumulation_steps)
    loss.backward()
    return loss, {}

    
def run_compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-No-Clip loss：L = -A × ratio（不做 clipping）。

    用于 Section 8.7 的 clip ablation 实验，对比有无 clipping 对训练稳定性
    和最终性能的影响。

    Args:
        advantages: shape (batch_size, 1)，per-sample advantage。
        policy_log_probs: shape (batch_size, sequence_length)，当前策略 log-probs。
        old_log_probs: 同 shape，旧策略 log-probs。

    Returns:
        (per-token loss, empty metadata dict)
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    loss = -advantages * ratio
    return loss, {}


def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """执行一个 GRPO microbatch 的前向+反向传播。

    先计算 per-token PG loss，再用 response_mask 只在 response 位置求和，
    除以 batch_size × gradient_accumulation_steps 后 backward。

    Args:
        policy_log_probs: shape (batch_size, sequence_length)，当前策略 log-probs。
        response_mask: 同 shape bool 张量。
        gradient_accumulation_steps: 梯度累积步数。
        loss_type: "no_baseline" / "reinforce_with_baseline" / "grpo_clip"。
        raw_rewards: loss_type="no_baseline" 时需提供。
        advantages: loss_type 含 baseline 时需提供。
        old_log_probs: loss_type="grpo_clip" 时需提供。
        cliprange: loss_type="grpo_clip" 时需提供。
    """
    loss_per_token, loss_metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    masked = loss_per_token * response_mask.float()
    per_sample_mean = masked.sum(dim=1) / response_mask.float().sum(dim=1).clamp_min(1)
    loss = per_sample_mean.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, loss_metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """沿指定维度对被 mask 选中的元素求和，并用常数归一化。

    Args:
        tensor: 输入张量。
        mask: 与 tensor 同 shape 的 bool 张量，mask=1 的位置参与求和。
        dim: 沿哪个维度求和；None 表示对所有元素求和。
        normalize_constant: 归一化分母。

    Returns:
        torch.Tensor，求和后除以 normalize_constant 的结果。
    """
    masked = tensor * mask.float()
    return masked.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """构造一个固定长度 packed SFT 数据集。

    读取 JSONL 中的 prompt/response 对，把所有样本的 token 序列拼接成一条长
    序列，然后切分成 seq_length 个 token 的段。每段用因果语言模型的常规方式
    构造 input_ids（前 seq_length-1 个 token）和 labels（后 seq_length-1 个
    token，右移一位）。
    """
    import json
    import random

    # 1. 读取 JSONL 文件中的所有 prompt/response 对。
    records: list[dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "response" not in record:
                raise ValueError(f"Each line must contain 'prompt' and 'response' keys, got: {list(record.keys())}")
            records.append(record)

    # 2. 编码每条文档。
    #    使用 Llama 3 instruction format 手动构造 token 序列：
    #    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    #    \n\n{prompt}<|eot_id|>
    #    <|start_header_id|>assistant<|end_header_id|>
    #    \n\n{response}<|eot_id|>
    #
    #    注意：prompt/response 文本和周围特殊 token 必须作为完整字符串一起 tokenize，
    #    以保证 BPE 分词在边界处正确。
    BOS_ID = tokenizer.bos_token_id
    if BOS_ID is None:
        raise ValueError("tokenizer must have bos_token_id set")

    def _encode_special(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    start_header = _encode_special("<|start_header_id|>")
    end_header = _encode_special("<|end_header_id|>")
    eot = _encode_special("<|eot_id|>")
    user_role = _encode_special("user")
    assistant_role = _encode_special("assistant")
    newline = _encode_special("\n\n")

    doc_tokens_list: list[list[int]] = []
    for record in records:
        prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
        response_ids = tokenizer.encode(record["response"], add_special_tokens=False)
        doc_tokens = (
            [BOS_ID]
            + start_header + user_role + end_header
            + newline
            + prompt_ids
            + eot
            + start_header + assistant_role + end_header
            + newline
            + response_ids
            + eot
        )
        doc_tokens_list.append(doc_tokens)

    # 3. 可选打乱文档顺序。
    if shuffle:
        rng = random.Random(42)
        rng.shuffle(doc_tokens_list)

    # 4. 把所有文档的 token 拼接成一条长序列。
    all_tokens: list[int] = []
    for doc_tokens in doc_tokens_list:
        all_tokens.extend(doc_tokens)

    # 5. 按 seq_length 切分成固定长度段，每段需要 seq_length+1 个 token 来
    #    构造 input_ids（前 seq_length 个）和 labels（后 seq_length 个，右移一位）。
    chunk_size = seq_length + 1
    num_chunks = len(all_tokens) // chunk_size
    if num_chunks == 0:
        raise ValueError(
            f"Not enough tokens ({len(all_tokens)}) to create at least one chunk "
            f"of size {chunk_size} (seq_length={seq_length})"
        )

    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []
    for i in range(num_chunks):
        chunk = all_tokens[i * chunk_size : (i + 1) * chunk_size]
        # input_ids 是去掉最后一个 token 的前缀；labels 是去掉第一个 token 的后缀，
        # 这样 labels[t] 就是模型在 input_ids[:t+1] 条件下应该预测的下一个 token。
        input_ids_list.append(chunk[:-1])
        labels_list.append(chunk[1:])

    # 6. 包装为简单的 PyTorch Dataset。
    class PackedSFTDataset(Dataset):
        def __init__(self, input_ids: list[list[int]], labels: list[list[int]]):
            self.input_ids = input_ids
            self.labels = labels

        def __len__(self) -> int:
            return len(self.input_ids)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return PackedSFTDataset(input_ids_list, labels_list)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """用 DataLoader 对数据集按 batch_size 分组，返回一个可迭代对象。

    遍历该迭代器一次等价于对数据集完成一个 epoch。每个 batch 是一个字典，
    包含 "input_ids" 和 "labels"，形状为 (batch_size, seq_length)（最后一
    个 batch 可能不足 batch_size）。
    """
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(42)

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """把同一个 batch 内的样本在 batch 维度上堆叠。"""
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
    )


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """从 MMLU 模型输出中抽取预测的选项字母 (A/B/C/D)。

    解析策略：
    1. 先尝试在输出中直接匹配 "A"、"B"、"C"、"D" 等独立字母（用单词边界限定）。
    2. 如果失败，尝试匹配选项文本（"A)"、"B)" 等）。
    3. 如果仍失败，返回 None。
    """
    import re

    # 方法1：在文本末尾附近查找 option letter（模型输出通常在最后给出选项）
    # 匹配类似 "answer is A" 或 "answer: B" 或单独的 "A." / "(A)" 等模式
    patterns = [
        # 匹配 "(A)"、"(B)" 等
        r'\(([A-D])\)',
        # 匹配 "A."、"B." 等独立选项
        r'\b([A-D])\.',
        # 匹配 "option A"、"answer A" 等
        r'(?:answer|option|choice)\s*(?:is|:)?\s*([A-D])\b',
        # 匹配 "I choose A" 等
        r'(?:choose|select|pick)\s*([A-D])\b',
        # 匹配行首单独的 "A"、"B"、"C"、"D"
        r'^([A-D])\s*$',
        # 最后匹配文本最后一个出现的独立 A/B/C/D（作为后备）
        r'\b([A-D])\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, model_output, re.IGNORECASE)
        if matches:
            # 取最后一个匹配的选项字母
            return matches[-1].upper()

    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """从 GSM8K 模型输出中抽取预测的数字答案。

    抽取规则：取输出中最后一个出现的数字（包括带小数点的数字和逗号分隔的
    大数字如 "70,000"）。
    """
    import re

    # 匹配所有可能的数字：含可选的负号、逗号分隔的千位、可选的小数部分。
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None
    # 返回最后一个数字，去除逗号并 strip
    return matches[-1].replace(",", "").strip()


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """计算单条偏好对的 DPO (Direct Preference Optimization) loss。

    DPO loss 公式：
        L = -log σ( β × (log π_θ(chosen|prompt) - log π_ref(chosen|prompt)
                         - log π_θ(rejected|prompt) + log π_ref(rejected|prompt)) )

    其中 σ 是 sigmoid，π_θ 和 π_ref 分别是当前策略模型和参考模型。
    log π(response|prompt) 是 model 在给定 prompt 条件下对 response token
    的平均条件 log-prob（只对 response 部分求和，不包括 prompt）。
    """
    import torch.nn.functional as F

    def _response_log_prob(
        model: torch.nn.Module,
        prompt_str: str,
        response_str: str,
    ) -> torch.Tensor:
        """计算模型在给定 prompt 下生成 response 的 log-prob。

        只对 response token 的 log-prob 求和，prompt 部分不参与计算。

        关键：prompt 和 response 必须合并编码（而非分别编码后拼接），因为 BPE
        分词在连接处的边界可能因上下文不同而产生不同的 token 序列。

        causal LM 中 logits[t] 预测 token[t+1]（即 labels[t]），因此第一个
        response token（R_0）的预测来自 logits[P-1] 位置，不是 logits[P]。
        """
        # 单独编码 prompt 仅用于确定 response 起始位置。
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        P = len(prompt_ids)

        # 合并编码 prompt+response，确保 BPE 分词在连接处正确。
        full_text = prompt_str + response_str
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # 构造 causal LM 的输入和标签：input_ids 去掉最后一个 token，
        # labels 去掉第一个 token（等于右移一位）。
        input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
        labels = torch.tensor([full_ids[1:]], dtype=torch.long)

        # 确保模型在 eval 模式下计算（关闭 dropout 等），保证确定性。
        was_training = model.training
        if was_training:
            model.eval()
        try:
            logits = model(input_ids=input_ids).logits
        finally:
            if was_training:
                model.train()

        log_probs_all = F.log_softmax(logits, dim=-1)

        # logits[0, P-1] 预测的是 token[P] = R_0（第一个 response token）。
        # 因此 response 的 log-prob 索引从 P-1 开始。
        response_start = P - 1
        response_log_probs = log_probs_all[0, response_start:].gather(
            dim=-1, index=labels[0, response_start:].unsqueeze(-1)
        ).squeeze(-1)

        return response_log_probs.sum()

    # DPO 核心计算
    chosen_log_pi = _response_log_prob(lm, prompt, response_chosen)
    chosen_log_ref = _response_log_prob(lm_ref, prompt, response_chosen)
    rejected_log_pi = _response_log_prob(lm, prompt, response_rejected)
    rejected_log_ref = _response_log_prob(lm_ref, prompt, response_rejected)

    # log-ratio 差异 × beta
    chosen_diff = chosen_log_pi - chosen_log_ref
    rejected_diff = rejected_log_pi - rejected_log_ref
    logits_diff = beta * (chosen_diff - rejected_diff)

    # DPO loss: -log σ(logits_diff)
    return -F.logsigmoid(logits_diff)
