"""评估 Qwen 2.5 Math 1.5B 在数学题上的 zero-shot 表现。"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接用 `python scripts/math_baseline.py` 运行，而不依赖包安装状态。
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.evaluation import (
    build_math_prompts,
    collect_category_examples,
    evaluate_vllm,
    load_jsonl,
    read_prompt_template,
    r1_zero_numeric_reward_fn,
    write_json,
    write_jsonl,
)


LOCAL_MODELSCOPE_MODEL_PATH = PROJECT_ROOT / "models/Qwen/Qwen2___5-Math-1___5B"
DEFAULT_MODEL_PATH = str(LOCAL_MODELSCOPE_MODEL_PATH) if LOCAL_MODELSCOPE_MODEL_PATH.exists() else "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data/gsm8k/test.jsonl"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/math_baseline_gsm8k"


def parse_args() -> argparse.Namespace:
    """解析 zero-shot baseline 的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="HF 模型名或本地模型路径。")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="MATH 或替代数据集的 JSONL 路径。")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "math", "gsm8k"],
        default="gsm8k",
        help="数据集格式；GSM8K 会从 `####` 后抽取短答案。",
    )
    parser.add_argument(
        "--reward-fn",
        choices=["auto", "r1_zero", "numeric"],
        default="auto",
        help="reward 函数；GSM8K 默认使用轻量 numeric reward，MATH 默认使用官方 r1_zero_reward_fn。",
    )
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="Prompt 模板路径。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="指标和 JSONL 输出目录。")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条，方便 smoke test。")
    parser.add_argument("--seed", type=int, default=0, help="vLLM 采样随机种子。")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样 temperature。")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling 的 top-p。")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成 token 数。")
    parser.add_argument("--stop", default="</answer>", help="vLLM 生成停止字符串。")
    parser.add_argument(
        "--exclude-stop-from-output",
        action="store_true",
        help="不把 stop string 写入 response；R1-Zero 评估通常不要打开这个选项。",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel 大小。")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM 可使用的 GPU 显存比例。",
    )
    parser.add_argument("--dtype", default="auto", help="vLLM 模型 dtype。")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="向 vLLM 加载模型时传入 trust_remote_code=True。",
    )
    parser.add_argument(
        "--slow-grader",
        action="store_true",
        help="使用 r1_zero_reward_fn 中更慢但召回更高的判分路径。",
    )
    parser.add_argument(
        "--category-examples",
        type=int,
        default=25,
        help="每个 reward 类别保存多少条样本用于人工分析。",
    )
    return parser.parse_args()


def make_sampling_params(args: argparse.Namespace) -> Any:
    """按 handout 的 zero-shot 设置创建 vLLM SamplingParams。"""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[args.stop] if args.stop else None,
        include_stop_str_in_output=not args.exclude_stop_from_output,
    )


def make_llm(args: argparse.Namespace) -> Any:
    """创建 vLLM 模型实例。"""
    from vllm import LLM

    return LLM(
        model=args.model,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def select_reward_fn(args: argparse.Namespace) -> Any:
    """根据数据集格式选择 reward function。"""
    reward_name = args.reward_fn
    if reward_name == "auto":
        reward_name = "numeric" if args.dataset_format == "gsm8k" else "r1_zero"

    if reward_name == "numeric":
        return r1_zero_numeric_reward_fn

    # 官方 MATH reward 依赖 math_verify / latex2sympy2_extended，只在确实需要时导入。
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    return partial(r1_zero_reward_fn, fast=not args.slow_grader)


def main() -> None:
    """运行 zero-shot 评估，并保存后续分析所需的完整结果。"""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.data)
    if args.limit is not None:
        # 小样本运行用于确认数据 schema、prompt 和 vLLM 环境是否正常。
        examples = examples[: args.limit]

    prompt_template = read_prompt_template(args.prompt)
    prompts, ground_truths, _ = build_math_prompts(
        examples,
        prompt_template,
        dataset_format=args.dataset_format,
    )

    llm = make_llm(args)
    sampling_params = make_sampling_params(args)
    reward_fn = select_reward_fn(args)

    results, metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        examples=examples,
        eval_sampling_params=sampling_params,
    )

    run_config = {
        "model": args.model,
        "data": args.data,
        "dataset_format": args.dataset_format,
        "reward_fn": args.reward_fn,
        "prompt": args.prompt,
        "limit": args.limit,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stop": args.stop,
        "include_stop_str_in_output": not args.exclude_stop_from_output,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "slow_grader": args.slow_grader,
    }
    metrics_with_config = {"config": run_config, **metrics}

    write_jsonl(output_dir / "generations.jsonl", results)
    write_json(output_dir / "metrics.json", metrics_with_config)
    write_json(
        output_dir / "category_examples.json",
        collect_category_examples(results, max_examples_per_category=args.category_examples),
    )

    print(f"Wrote generations to {output_dir / 'generations.jsonl'}")
    print(f"Wrote metrics to {output_dir / 'metrics.json'}")
    print(metrics_with_config)


if __name__ == "__main__":
    main()
