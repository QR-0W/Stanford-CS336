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
    """
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
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
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
