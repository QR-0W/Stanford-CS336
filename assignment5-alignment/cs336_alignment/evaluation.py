"""数学推理模型评估工具。

这个模块把作业 5 中会反复用到的评估流程拆出来：读取 JSONL、把题目
填入 prompt、调用 vLLM 生成、用 reward function 打分，并把结果汇总成
后续写 notes / writeup 需要的统计量。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable


# 官方 MATH 数据和开源替代数据的字段名不完全一致，所以这里允许几个常见别名。
QUESTION_KEYS = ("question", "problem", "prompt")
GROUND_TRUTH_KEYS = ("ground_truth", "answer", "final_answer", "target")
DATASET_FORMATS = ("auto", "math", "gsm8k")


def normalize_numeric_answer(answer: Any) -> str:
    """把数字答案归一化成便于比较的字符串。"""
    text = str(answer).strip()
    text = text.replace(",", "").replace("$", "")
    text = text.strip().rstrip(".")
    return text


def extract_gsm8k_final_answer(answer: Any) -> str:
    """从 GSM8K 解答中的 `####` 后抽取最终短答案。"""
    if not isinstance(answer, str):
        return normalize_numeric_answer(answer)
    if "####" not in answer:
        raise ValueError("GSM8K answer is missing the final-answer marker '####'")
    final_answer = answer.rsplit("####", maxsplit=1)[-1]
    return normalize_numeric_answer(final_answer)


def maybe_extract_gsm8k_final_answer(answer: Any) -> Any:
    """auto 模式下只在看到 GSM8K 标记时抽取短答案。"""
    if isinstance(answer, str) and "####" in answer:
        return extract_gsm8k_final_answer(answer)
    return answer


def read_prompt_template(path: str | Path) -> str:
    """从磁盘读取 prompt 模板。"""
    return Path(path).read_text(encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，并要求每一行都是 JSON object。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_number}, got {type(record).__name__}")
            records.append(record)
    return records


def get_first_present(record: dict[str, Any], keys: Iterable[str], field_name: str) -> Any:
    """从多个候选字段名中取第一个存在的字段。"""
    for key in keys:
        if key in record:
            return record[key]
    available = ", ".join(sorted(record.keys()))
    expected = ", ".join(keys)
    raise KeyError(f"Could not find {field_name}; expected one of [{expected}], available [{available}]")


def get_question(record: dict[str, Any]) -> str:
    """从 MATH 或替代数据样本中抽取题目文本。"""
    question = get_first_present(record, QUESTION_KEYS, "question")
    if not isinstance(question, str):
        raise TypeError(f"Question must be a string, got {type(question).__name__}")
    return question


def get_ground_truth(record: dict[str, Any], dataset_format: str = "auto") -> str | int | float | list[Any]:
    """从 MATH 或替代数据样本中抽取标准答案。"""
    if dataset_format not in DATASET_FORMATS:
        raise ValueError(f"Unknown dataset format: {dataset_format}")

    ground_truth = get_first_present(record, GROUND_TRUTH_KEYS, "ground truth")
    if dataset_format == "gsm8k":
        return extract_gsm8k_final_answer(ground_truth)
    if dataset_format == "auto":
        return maybe_extract_gsm8k_final_answer(ground_truth)
    return ground_truth


def format_prompt(prompt_template: str, question: str) -> str:
    """把题目填入 R1-Zero 或 question-only prompt 模板。"""
    return prompt_template.format(question=question)


def build_math_prompts(
    examples: list[dict[str, Any]],
    prompt_template: str,
    dataset_format: str = "auto",
) -> tuple[list[str], list[Any], list[str]]:
    """批量构造模型输入 prompt、标准答案和原始题目。"""
    questions = [get_question(example) for example in examples]
    ground_truths = [get_ground_truth(example, dataset_format=dataset_format) for example in examples]
    prompts = [format_prompt(prompt_template, question) for question in questions]
    return prompts, ground_truths, questions


def extract_r1_zero_answer(response: str) -> str | None:
    """按 R1-Zero 格式从 response 中抽取 `<answer>` 内的内容。"""
    match = re.search(r"</think>\s*<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def extract_last_number(text: str) -> str | None:
    """从文本中抽取最后一个数字，兼容 `$18`、`70,000` 等常见 GSM8K 输出。"""
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])


def numeric_answers_equal(predicted: Any, ground_truth: Any) -> bool:
    """比较两个短数字答案，优先用有理数/Decimal 避免字符串格式差异。"""
    predicted_text = normalize_numeric_answer(predicted)
    ground_truth_text = normalize_numeric_answer(ground_truth)
    if predicted_text == ground_truth_text:
        return True

    try:
        return Fraction(predicted_text) == Fraction(ground_truth_text)
    except (ValueError, ZeroDivisionError):
        pass

    try:
        return Decimal(predicted_text) == Decimal(ground_truth_text)
    except InvalidOperation:
        return False


def r1_zero_numeric_reward_fn(response: str, ground_truth: Any) -> dict[str, float]:
    """用于 GSM8K 自学版的轻量 R1-Zero 数字 reward。

    需要模型输出包含 &LT;think&GT; / &LT;answer&GT; 标签格式。
    """
    model_answer = extract_r1_zero_answer(response)
    if model_answer is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

    if numeric_answers_equal(model_answer, ground_truth):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}

    predicted_number = extract_last_number(model_answer) or model_answer
    if numeric_answers_equal(predicted_number, ground_truth):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


def question_only_numeric_reward_fn(response: str, ground_truth: Any) -> dict[str, float]:
    """用于 question_only prompt 的 GSM8K 轻量数字 reward。

    与 r1_zero_numeric_reward_fn 不同：
    - 不要求 &LT;think&GT; / &LT;answer&GT; 格式（模型在 question_only prompt 下
      不会被要求输出特定标签）。
    - 直接从整个 response 中抽取最后一个数字与 ground truth 比较。
    - format_reward 始终为 1.0（无格式要求）。
    """
    predicted_number = extract_last_number(response)
    if predicted_number is None:
        return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}

    if numeric_answers_equal(predicted_number, ground_truth):
        return {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
    return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 reward 均值和 handout 要求的三类输出数量。"""
    totals = Counter()
    reward_sums = Counter()

    for result in results:
        scores = result["scores"]
        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))
        reward = float(scores.get("reward", 0.0))

        reward_sums["format_reward"] += format_reward
        reward_sums["answer_reward"] += answer_reward
        reward_sums["reward"] += reward

        # handout 要求分别统计：格式正确且答案正确、格式正确但答案错误、格式错误。
        if format_reward == 1.0 and answer_reward == 1.0:
            totals["format_1_answer_1"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            totals["format_1_answer_0"] += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            totals["format_0_answer_0"] += 1
        else:
            totals["other"] += 1

    total = len(results)
    means = {
        key: (value / total if total else 0.0)
        for key, value in reward_sums.items()
    }
    return {
        "num_examples": total,
        "mean_scores": means,
        "category_counts": dict(totals),
    }


def collect_category_examples(
    results: list[dict[str, Any]],
    max_examples_per_category: int,
) -> dict[str, list[dict[str, Any]]]:
    """为人工错误分析保存每类输出的代表样本。"""
    categories: dict[str, list[dict[str, Any]]] = {
        "format_1_answer_1": [],
        "format_1_answer_0": [],
        "format_0_answer_0": [],
        "other": [],
    }
    for result in results:
        scores = result["scores"]
        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))
        if format_reward == 1.0 and answer_reward == 1.0:
            category = "format_1_answer_1"
        elif format_reward == 1.0 and answer_reward == 0.0:
            category = "format_1_answer_0"
        elif format_reward == 0.0 and answer_reward == 0.0:
            category = "format_0_answer_0"
        else:
            category = "other"

        if len(categories[category]) < max_examples_per_category:
            categories[category].append(result)
    return categories


def write_json(path: str | Path, payload: Any) -> None:
    """把对象写成方便阅读的 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """把多条记录写成 JSONL 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, Any], dict[str, float]],
    prompts: list[str],
    ground_truths: list[Any],
    examples: list[dict[str, Any]],
    eval_sampling_params: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用 vLLM 生成回答，并用指定 reward function 逐条打分。"""
    if len(prompts) != len(ground_truths) or len(prompts) != len(examples):
        raise ValueError("prompts, ground_truths, and examples must have the same length")

    # vLLM 会返回与 prompts 等长的 RequestOutput 列表，每条取第一个采样结果。
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results: list[dict[str, Any]] = []
    for index, (example, prompt, ground_truth, output) in enumerate(
        zip(examples, prompts, ground_truths, outputs, strict=True)
    ):
        if not output.outputs:
            response = ""
            finish_reason = "no_output"
        else:
            first_output = output.outputs[0]
            response = first_output.text
            finish_reason = getattr(first_output, "finish_reason", None)

        scores = reward_fn(response, ground_truth)
        results.append(
            {
                "index": index,
                "example": example,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                "finish_reason": finish_reason,
                "scores": scores,
            }
        )

    return results, summarize_scores(results)


def mean_or_none(values: list[float]) -> float | None:
    """计算均值；空列表返回 None，方便 JSON 日志区分 0 和缺失。"""
    if not values:
        return None
    return sum(values) / len(values)


def get_generated_response(output: Any) -> tuple[str, str | None, Any | None]:
    """从 vLLM RequestOutput 中取第一条生成文本、结束原因和原始生成对象。"""
    if not getattr(output, "outputs", None):
        return "", "no_output", None
    first_output = output.outputs[0]
    response = getattr(first_output, "text", "")
    finish_reason = getattr(first_output, "finish_reason", None)
    return response, finish_reason, first_output


def response_length(response: str, generation: Any | None = None, tokenizer: Any | None = None) -> int:
    """优先使用 vLLM token_ids 统计长度；否则退回 tokenizer 或 whitespace 计数。"""
    if generation is not None and getattr(generation, "token_ids", None) is not None:
        return len(generation.token_ids)
    if tokenizer is not None:
        return len(tokenizer.encode(response, add_special_tokens=False))
    return len(response.split())


def log_generations(
    vllm_model: Any,
    reward_fn: Callable[[str, Any], dict[str, float]],
    prompts: list[str],
    ground_truths: list[Any],
    eval_sampling_params: Any,
    examples: list[dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    response_token_entropies: list[list[float]] | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """生成并记录训练期样本，用于 SFT/RL 的人工检查和指标追踪。

    每条日志包含 prompt、模型 response、ground truth、reward 信息、response 长度和
    可选平均 token entropy。返回的 summary 还会统计整体、正确样本和错误样本的平均长度。
    """
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")
    if examples is not None and len(examples) != len(prompts):
        raise ValueError("examples must have the same length as prompts")
    if response_token_entropies is not None and len(response_token_entropies) != len(prompts):
        raise ValueError("response_token_entropies must have the same length as prompts")

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    if len(outputs) != len(prompts):
        raise ValueError("vLLM returned a different number of outputs than prompts")

    records: list[dict[str, Any]] = []
    response_lengths: list[float] = []
    correct_lengths: list[float] = []
    incorrect_lengths: list[float] = []
    average_entropies: list[float] = []

    for index, (prompt, ground_truth, output) in enumerate(zip(prompts, ground_truths, outputs, strict=True)):
        response, finish_reason, generation = get_generated_response(output)
        scores = reward_fn(response, ground_truth)
        length = response_length(response, generation=generation, tokenizer=tokenizer)
        entropy_values = response_token_entropies[index] if response_token_entropies is not None else []
        average_entropy = mean_or_none([float(value) for value in entropy_values])
        is_correct = float(scores.get("answer_reward", scores.get("reward", 0.0))) == 1.0

        response_lengths.append(float(length))
        if is_correct:
            correct_lengths.append(float(length))
        else:
            incorrect_lengths.append(float(length))
        if average_entropy is not None:
            average_entropies.append(average_entropy)

        record = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "scores": scores,
            "finish_reason": finish_reason,
            "response_length": length,
            "average_token_entropy": average_entropy,
        }
        if examples is not None:
            record["example"] = examples[index]
        records.append(record)

    summary = {
        **summarize_scores(records),
        "generation_stats": {
            "average_response_length": mean_or_none(response_lengths),
            "average_correct_response_length": mean_or_none(correct_lengths),
            "average_incorrect_response_length": mean_or_none(incorrect_lengths),
            "average_token_entropy": mean_or_none(average_entropies),
        },
    }

    if output_path is not None:
        write_jsonl(output_path, records)
    if summary_path is not None:
        write_json(summary_path, summary)
    return records, summary
