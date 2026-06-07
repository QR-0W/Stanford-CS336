"""SFT 实验工具。

这里放 SFT experiment 中可复用的数据构造、batch tokenization 和 microbatch
训练逻辑。官方 MATH SFT 数据已经是 {"prompt": str, "response": str}；
自学版 GSM8K 则从 question/answer 字段构造 R1-Zero 风格 prompt/response。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from cs336_alignment.evaluation import extract_gsm8k_final_answer, read_prompt_template


@dataclass(frozen=True)
class SFTExample:
    """一条 SFT 样本。"""

    prompt: str
    response: str
    source: dict[str, Any]


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 样本。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}")
            records.append(record)
    return records


def gsm8k_answer_to_r1_response(answer: str) -> str:
    """把 GSM8K 标准解答转成 R1-Zero prompt 后续 response。"""
    if "####" not in answer:
        raise ValueError("GSM8K answer must contain '####'")
    reasoning = answer.rsplit("####", maxsplit=1)[0].strip()
    final_answer = extract_gsm8k_final_answer(answer)
    return f"{reasoning}\n</think>\n<answer>{final_answer}</answer>"


def record_to_sft_example(
    record: dict[str, Any],
    dataset_format: str,
    prompt_template: str,
) -> SFTExample:
    """把不同数据格式统一成 SFTExample。"""
    if dataset_format in {"auto", "math_sft"} and "prompt" in record and "response" in record:
        return SFTExample(
            prompt=str(record["prompt"]),
            response=str(record["response"]),
            source=record,
        )

    if dataset_format in {"auto", "gsm8k"} and "question" in record and "answer" in record:
        prompt = prompt_template.format(question=record["question"])
        response = gsm8k_answer_to_r1_response(str(record["answer"]))
        return SFTExample(prompt=prompt, response=response, source=record)

    available = ", ".join(sorted(record.keys()))
    raise ValueError(f"Could not convert record to SFT example; keys: {available}")


def load_sft_examples(
    path: str | Path,
    dataset_format: str,
    prompt_template_path: str | Path,
    limit: int | None = None,
    seed: int = 0,
) -> list[SFTExample]:
    """读取并可选打乱/截断 SFT 样本。"""
    if dataset_format not in {"auto", "gsm8k", "math_sft"}:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")
    prompt_template = read_prompt_template(prompt_template_path)
    examples = [
        record_to_sft_example(record, dataset_format=dataset_format, prompt_template=prompt_template)
        for record in load_jsonl_records(path)
    ]
    rng = random.Random(seed)
    rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def get_filter_ground_truth(example: SFTExample) -> Any:
    """从 SFT 样本的原始字段中取过滤用 ground truth。"""
    source = example.source
    for key in ("ground_truth", "final_answer", "target"):
        if key in source:
            return source[key]
    if "answer" in source:
        answer = source["answer"]
        if isinstance(answer, str) and "####" in answer:
            return extract_gsm8k_final_answer(answer)
        return answer
    raise KeyError("Cannot filter SFT example without ground truth/answer/final_answer/target")


def filter_correct_examples(
    examples: list[SFTExample],
    reward_fn: Any,
) -> tuple[list[SFTExample], dict[str, Any]]:
    """只保留 reward 判定为正确的 SFT 样本。"""
    kept: list[SFTExample] = []
    rejected = 0
    for example in examples:
        ground_truth = get_filter_ground_truth(example)
        scores = reward_fn(example.response, ground_truth)
        if float(scores.get("answer_reward", scores.get("reward", 0.0))) == 1.0:
            kept.append(example)
        else:
            rejected += 1
    return kept, {"original_size": len(examples), "filtered_size": len(kept), "rejected_size": rejected}


def select_train_size(examples: list[SFTExample], train_size: str, seed: int) -> list[SFTExample]:
    """按 handout 的数据规模实验选择训练子集。"""
    if train_size == "full":
        return list(examples)
    size = int(train_size)
    if size > len(examples):
        raise ValueError(f"Requested train_size={size}, but only {len(examples)} examples are available")
    rng = random.Random(seed)
    selected = list(examples)
    rng.shuffle(selected)
    return selected[:size]


def tokenize_prompt_and_output_batch(
    examples: list[SFTExample],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    """把一个 batch 的 prompt/response 转成 input_ids、labels 和 response_mask。"""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define either pad_token_id or eos_token_id")

    encoded_rows: list[tuple[list[int], list[int], list[int]]] = []
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(example.response, add_special_tokens=False)
        token_ids = (prompt_ids + response_ids)[:max_sequence_length]
        response_len = max(0, len(token_ids) - len(prompt_ids))
        encoded_rows.append((prompt_ids, response_ids[:response_len], token_ids))

    max_len = max(len(token_ids) for _, _, token_ids in encoded_rows)
    if max_len < 2:
        raise ValueError("At least two tokens are required to create input_ids/labels")

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []
    for prompt_ids, response_ids, token_ids in encoded_rows:
        padded = token_ids + [pad_token_id] * (max_len - len(token_ids))
        input_ids.append(padded[:-1])
        labels.append(padded[1:])
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        response_mask.append(
            [
                prompt_len - 1 <= label_position < prompt_len + response_len - 1
                for label_position in range(max_len - 1)
            ]
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def make_sft_dataloader(
    examples: list[SFTExample],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_sequence_length: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[SFTExample]:
    """创建按 batch 动态 padding 的 SFT dataloader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)

    def collate_fn(batch: list[SFTExample]) -> dict[str, torch.Tensor]:
        return tokenize_prompt_and_output_batch(
            batch,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )

    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
    )


def gather_label_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """从每个位置的词表分布中取出 label token 的 log-prob。"""
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def sft_microbatch_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    gradient_accumulation_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """执行一个真实模型上的 SFT microbatch backward。"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    response_mask = batch["response_mask"].to(device)

    logits = model(input_ids=input_ids).logits
    policy_log_probs = gather_label_log_probs(logits, labels)
    masked_sum = (policy_log_probs * response_mask.float()).sum()
    response_tokens = response_mask.sum().clamp_min(1)
    loss = -masked_sum / response_tokens / gradient_accumulation_steps
    loss.backward()
    return loss.detach() * gradient_accumulation_steps, {
        "response_tokens": float(response_tokens.detach().cpu()),
        "mean_response_log_prob": float((masked_sum / response_tokens).detach().cpu()),
    }


def iterate_epochs(dataloader: Iterable[dict[str, torch.Tensor]], epochs: int) -> Iterable[dict[str, torch.Tensor]]:
    """把有限 dataloader 展开成固定 epoch 数的 batch 流。"""
    for _ in range(epochs):
        yield from dataloader
