"""运行 Assignment 5 的 SFT experiment。

官方版本使用 `/data/a5-alignment/MATH/sft.jsonl`；本机自学版默认使用仓库自带
GSM8K train split 构造 R1-Zero 风格 reasoning SFT 样本。脚本支持按不同 train
size 从同一 base model 重新训练，并保存 train logs / checkpoint / 可选 vLLM 评估。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.evaluation import (  # noqa: E402
    build_math_prompts,
    evaluate_vllm,
    load_jsonl,
    read_prompt_template,
    r1_zero_numeric_reward_fn,
    write_json,
    write_jsonl,
)
from cs336_alignment.sft import (  # noqa: E402
    filter_correct_examples,
    iterate_epochs,
    load_sft_examples,
    make_sft_dataloader,
    select_train_size,
    sft_microbatch_step,
)


LOCAL_MODELSCOPE_MODEL_PATH = PROJECT_ROOT / "models/Qwen/Qwen2___5-Math-1___5B"
DEFAULT_MODEL_PATH = str(LOCAL_MODELSCOPE_MODEL_PATH) if LOCAL_MODELSCOPE_MODEL_PATH.exists() else "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "data/gsm8k/train.jsonl"
DEFAULT_EVAL_DATA = PROJECT_ROOT / "data/gsm8k/test.jsonl"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/sft_experiment_gsm8k"


def parse_train_sizes(text: str) -> list[str]:
    """解析 `128,256,full` 形式的数据规模列表。"""
    sizes = [item.strip() for item in text.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one train size is required")
    for item in sizes:
        if item != "full" and int(item) <= 0:
            raise argparse.ArgumentTypeError("train sizes must be positive integers or 'full'")
    return sizes


def parse_args() -> argparse.Namespace:
    """解析 SFT experiment 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="HF model id 或本地模型路径。")
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA), help="SFT JSONL 路径。")
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL_DATA), help="评估 JSONL 路径。")
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "gsm8k", "math_sft"],
        default="gsm8k",
        help="训练数据格式；GSM8K 会构造 R1-Zero response。",
    )
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="R1-Zero prompt 模板路径。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="实验输出目录。")
    parser.add_argument("--train-sizes", type=parse_train_sizes, default=parse_train_sizes("128"), help="如 128,256,512,1024,full。")
    parser.add_argument("--epochs", type=int, default=1, help="每个 train size 的 epoch 数。")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 train size 最多训练多少 optimizer steps。")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="microbatch size。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="梯度累积步数。")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="AdamW 学习率。")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay。")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="cosine schedule warmup 比例。")
    parser.add_argument("--max-sequence-length", type=int, default=1024, help="prompt+response 最大 token 数。")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping 阈值。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备。")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16", help="模型训练 dtype。")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="HuggingFace attention backend；本机无 flash-attn 时用 sdpa。",
    )
    parser.add_argument("--save-model", action="store_true", help="保存每个 train size 的 checkpoint。")
    parser.add_argument("--filter-correct", action="store_true", help="训练前只保留 reward 判定答案正确的 SFT 样本。")
    parser.add_argument(
        "--filter-reward",
        choices=["numeric", "r1_zero"],
        default="numeric",
        help="--filter-correct 使用的 reward；GSM8K 默认 numeric，MATH 可用 r1_zero。",
    )
    parser.add_argument("--eval-with-vllm", action="store_true", help="训练后用 vLLM 评估 checkpoint。")
    parser.add_argument("--eval-limit", type=int, default=128, help="vLLM 评估样本数；None 不限制。")
    parser.add_argument("--eval-gpu-memory-utilization", type=float, default=0.60, help="vLLM 显存比例。")
    parser.add_argument("--log-every", type=int, default=1, help="每多少 optimizer steps 记录一次 train log。")
    return parser.parse_args()


def model_dtype(name: str) -> torch.dtype:
    """把 CLI dtype 转为 torch dtype。"""
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def count_optimizer_steps(num_batches: int, epochs: int, gradient_accumulation_steps: int, max_steps: int | None) -> int:
    """估算 optimizer step 数，用于 scheduler。"""
    steps = (num_batches * epochs + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    return min(steps, max_steps) if max_steps is not None else steps


def save_train_log(path: Path, records: list[dict[str, Any]]) -> None:
    """保存训练日志。"""
    write_jsonl(path, records)


def maybe_evaluate_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any] | None:
    """可选：释放训练模型后，用 vLLM 对 checkpoint 做小规模评估。"""
    if not args.eval_with_vllm:
        return None

    from vllm import LLM, SamplingParams

    prompt_template = read_prompt_template(args.prompt)
    eval_examples = load_jsonl(args.eval_data)
    if args.eval_limit is not None:
        eval_examples = eval_examples[: args.eval_limit]
    prompts, ground_truths, _ = build_math_prompts(eval_examples, prompt_template, dataset_format="gsm8k")

    llm = LLM(
        model=str(checkpoint_dir),
        dtype="auto",
        gpu_memory_utilization=args.eval_gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results, metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_numeric_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        examples=eval_examples,
        eval_sampling_params=sampling_params,
    )
    write_jsonl(checkpoint_dir / "eval_generations.jsonl", results)
    write_json(checkpoint_dir / "eval_metrics.json", metrics)
    return metrics


def select_filter_reward_fn(args: argparse.Namespace) -> Any:
    """选择过滤 SFT traces 用的 reward function。"""
    if args.filter_reward == "numeric":
        return r1_zero_numeric_reward_fn

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    return r1_zero_reward_fn


def train_one_size(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    all_examples: list[Any],
    train_size: str,
) -> dict[str, Any]:
    """从 base model 重新训练一个指定 train size 的 SFT run。"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected_examples = select_train_size(all_examples, train_size=train_size, seed=args.seed)
    dataloader = make_sft_dataloader(
        selected_examples,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        max_sequence_length=args.max_sequence_length,
        shuffle=True,
        seed=args.seed,
    )
    total_steps = count_optimizer_steps(
        num_batches=len(dataloader),
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    if total_steps <= 0:
        raise ValueError("No optimizer steps to run; adjust data size or batch settings")

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_dir = Path(args.output_dir) / f"train_size_{train_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log: list[dict[str, Any]] = []
    micro_step = 0
    optimizer_step = 0
    pending_losses: list[float] = []
    pending_response_tokens = 0.0

    progress = tqdm(total=total_steps, desc=f"SFT size={train_size}")
    optimizer.zero_grad(set_to_none=True)
    for batch in iterate_epochs(dataloader, epochs=args.epochs):
        loss, metadata = sft_microbatch_step(
            model,
            batch=batch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
        )
        pending_losses.append(float(loss.cpu()))
        pending_response_tokens += metadata["response_tokens"]
        micro_step += 1

        should_step = micro_step % args.gradient_accumulation_steps == 0
        is_last_requested_step = args.max_steps is not None and optimizer_step + 1 >= args.max_steps
        if should_step or is_last_requested_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            if optimizer_step % args.log_every == 0:
                record = {
                    "train_size": train_size,
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "loss": sum(pending_losses) / len(pending_losses),
                    "response_tokens": pending_response_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
                train_log.append(record)
                progress.set_postfix(loss=f"{record['loss']:.4f}")
            pending_losses = []
            pending_response_tokens = 0.0

        if args.max_steps is not None and optimizer_step >= args.max_steps:
            break

    progress.close()

    save_train_log(run_dir / "train_log.jsonl", train_log)
    checkpoint_dir = run_dir / "checkpoint"
    if args.save_model or args.eval_with_vllm:
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    eval_metrics = None
    if args.eval_with_vllm:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_metrics = maybe_evaluate_checkpoint(args, checkpoint_dir)

    summary = {
        "train_size": train_size,
        "num_train_examples": len(selected_examples),
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "final_train_loss": train_log[-1]["loss"] if train_log else None,
        "eval_metrics": eval_metrics,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    if not args.eval_with_vllm:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def main() -> None:
    """运行一个或多个 train-size 的 SFT 实验。"""
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "config.json", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_examples = load_sft_examples(
        args.train_data,
        dataset_format=args.dataset_format,
        prompt_template_path=args.prompt,
        seed=args.seed,
    )
    if args.filter_correct:
        all_examples, filter_summary = filter_correct_examples(
            all_examples,
            reward_fn=select_filter_reward_fn(args),
        )
        write_json(Path(args.output_dir) / "filter_summary.json", filter_summary)
    summaries = [train_one_size(args, tokenizer, all_examples, train_size) for train_size in args.train_sizes]
    write_json(Path(args.output_dir) / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
