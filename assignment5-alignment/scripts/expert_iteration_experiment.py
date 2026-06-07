"""运行 Expert Iteration (Algorithm 2) on MATH/GSM8K。

官方版本使用 /data/a5-alignment/MATH/train.jsonl；自学版默认使用 GSM8K train split。
每一轮 EI 步骤：采样问题 → vLLM rollout → filter 正确响应 → SFT → 保存 checkpoint + 评估。
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

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
    SFTExample,
    iterate_epochs,
    load_sft_examples,
    make_sft_dataloader,
    record_to_sft_example,
    sft_microbatch_step,
)

DEFAULT_BASE_MODEL = "models/Qwen/Qwen2___5-Math-1___5B"
DEFAULT_TRAIN_DATA = "data/gsm8k/train.jsonl"
DEFAULT_EVAL_DATA = "data/gsm8k/test.jsonl"
DEFAULT_PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
DEFAULT_OUTPUT_DIR = "outputs/expert_iteration_gsm8k"


def resolve_model_arg(raw: str) -> str:
    """解析 --base-model，优先用 Modelscope 本地路径。"""
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return str(candidate.resolve())
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Qwen 2.5 Math 1.5B path / HF id")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA, help="问题 JSONL")
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA, help="评估 JSONL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, help="R1-Zero prompt 模板")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="EI 实验输出目录")
    parser.add_argument("--dataset-format", choices=["auto", "gsm8k"], default="gsm8k", help="训练数据格式")
    parser.add_argument("--n-ei-steps", type=int, default=5, help="EI 轮数")
    parser.add_argument("--question-batch-sizes", type=str, default="512", help="每轮采样的问题数，逗号分隔")
    parser.add_argument("--G", type=int, default=2, help="每道题的 rollout 数 (vLLM n=G)")
    parser.add_argument("--sft-epochs", type=int, default=1, help="每轮 SFT epoch 数")
    parser.add_argument("--sft-max-steps", type=int, default=None, help="每轮 SFT 最多 steps")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="microbatch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--eval-limit", type=int, default=128, help="vLLM 评估样本数")
    parser.add_argument("--log-every", type=int, default=1, help="SFT 日志频率")
    return parser.parse_args()


def model_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float32


def create_sft_dataloader_from_examples(
    examples: list[SFTExample],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Any:
    """从 SFT examples 创建 dataloader。"""
    if len(examples) < args.per_device_batch_size:
        raise RuntimeError(
            f"Not enough SFT examples ({len(examples)}) for batch size {args.per_device_batch_size}"
        )
    return make_sft_dataloader(
        examples,
        tokenizer=tokenizer,
        batch_size=args.per_device_batch_size,
        max_sequence_length=args.max_sequence_length,
        shuffle=True,
        seed=args.seed,
    )


def log_ei_step(ei_dir: Path, ei_step: int, records: list[dict]) -> None:
    """保存每轮 EI 日志。"""
    log_path = ei_dir / f"ei_step_{ei_step:03d}" / "sft_train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(log_path, records)


def rollout_with_vllm(
    ei_dir: Path,
    ei_step: int,
    questions: list[str],
    ground_truths: list[Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """用 vLLM 对一组问题做 G 次 rollout 并打分。"""
    from vllm import LLM, SamplingParams

    rollout_dir = ei_dir / f"ei_step_{ei_step:03d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = read_prompt_template(args.prompt)
    prompts = [prompt_template.format(question=q) for q in questions]

    checkpoint_dir = ei_dir / f"ei_step_{ei_step:03d}" / "current_policy"
    llm = LLM(
        model=str(checkpoint_dir.resolve()),
        dtype="auto",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        n=args.G,
        seed=args.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    records: list[dict[str, Any]] = []
    correct_examples: list[SFTExample] = []
    for idx, (question, ground_truth, output) in enumerate(zip(questions, ground_truths, outputs, strict=True)):
        for completion in output.outputs:
            response = completion.text
            scores = r1_zero_numeric_reward_fn(response, ground_truth)
            is_correct = float(scores.get("answer_reward", 0.0)) == 1.0
            record = {
                "ei_step": ei_step,
                "question_idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "response": response,
                "scores": scores,
                "is_correct": is_correct,
            }
            records.append(record)
            if is_correct:
                correct_examples.append(
                    SFTExample(
                        prompt=prompts[idx],
                        response=response,
                        source={"question": question, "ground_truth": ground_truth},
                    )
                )
    write_jsonl(rollout_dir / "rollouts.jsonl", records)
    write_json(
        rollout_dir / "rollout_summary.json",
        {"num_rollout": len(records), "num_correct": len(correct_examples)},
    )
    return correct_examples


def run_sft_on_examples(
    device: torch.device,
    model: torch.nn.Module,
    sft_examples: list[SFTExample],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    ei_step: int,
    ei_dir: Path,
) -> None:
    """在给定的 SFT examples 上做训练（就地修改 model）。"""
    dataloader = create_sft_dataloader_from_examples(sft_examples, tokenizer, args)
    total_batches = len(dataloader) * args.sft_epochs
    total_steps = (total_batches + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    if args.sft_max_steps is not None:
        total_steps = min(total_steps, args.sft_max_steps)
    if total_steps <= 0:
        print(f"[EI Step {ei_step}] No SFT steps; skipping training")
        return

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_log: list[dict] = []
    micro_step = 0
    optimizer_step = 0
    pending_losses: list[float] = []
    pending_tokens = 0.0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc=f"[EI Step {ei_step}] SFT")
    for batch in iterate_epochs(dataloader, epochs=args.sft_epochs):
        loss, meta = sft_microbatch_step(model, batch, args.gradient_accumulation_steps, device)
        pending_losses.append(float(loss.cpu()))
        pending_tokens += meta["response_tokens"]
        micro_step += 1

        should_step = (micro_step % args.gradient_accumulation_steps == 0) or (
            args.sft_max_steps is not None and optimizer_step + 1 >= total_steps
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            progress.update(1)

            if optimizer_step % args.log_every == 0:
                train_log.append({
                    "ei_step": ei_step,
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "loss": sum(pending_losses) / len(pending_losses),
                    "response_tokens": pending_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                })
                progress.set_postfix(loss=f"{train_log[-1]['loss']:.4f}")
            pending_losses = []
            pending_tokens = 0.0
        if args.sft_max_steps is not None and optimizer_step >= args.sft_max_steps:
            break
    progress.close()
    log_ei_step(ei_dir, ei_step, train_log)


def eval_checkpoint(
    device: torch.device,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    args: argparse.Namespace,
    ei_step: int,
) -> dict[str, Any]:
    """两阶段评估：保存 checkpoint → 释放训练模型 → vLLM 加载 → 打分。"""
    from vllm import LLM, SamplingParams

    # 因为 vLLM 加载模型后可能还会占另一份显存，所以先保存 checkpoint，再释放训练模型引用。
    # （这里释放由外部 manage，此函数只负责读 checkpoint 文件夹）
    prompt_template = read_prompt_template(args.prompt)
    eval_examples = load_jsonl(args.eval_data)
    if args.eval_limit is not None:
        eval_examples = eval_examples[: args.eval_limit]
    prompts, ground_truths, _ = build_math_prompts(eval_examples, prompt_template, dataset_format="gsm8k")

    llm = LLM(
        model=str(checkpoint_dir.resolve()),
        dtype="auto",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results, metrics = evaluate_vllm(llm, r1_zero_numeric_reward_fn, prompts, ground_truths, eval_examples, sampling_params)
    eval_dir = checkpoint_dir.parent
    write_jsonl(eval_dir / "eval_generations.jsonl", results)
    write_json(eval_dir / "eval_metrics.json", metrics)
    return metrics


def train_ei(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "config.json", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_arg(args.base_model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取问题集（只看 question/ground_truth，不用 SFT 标注）
    train_examples = load_sft_examples(
        args.train_data,
        dataset_format=args.dataset_format,
        prompt_template_path=args.prompt,
        seed=args.seed,
    )
    questions = [ex.prompt for ex in train_examples]
    ground_truths = [
        (
            ex.source.get("ground_truth", ex.source.get("final_answer"))
            if isinstance(ex.source, dict)
            and any(k in ex.source for k in ("ground_truth", "final_answer"))
            else ex.response.rsplit("<answer>", maxsplit=1)[-1].replace("</answer>", "").strip()
        )
        for ex in train_examples
    ]

    device = torch.device(args.device)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_arg(args.base_model),
        dtype=model_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    model.train()

    ei_dir = Path(args.output_dir)
    question_batch_sizes = [int(x.strip()) for x in args.question_batch_sizes.split(",")]

    all_metrics: list[dict] = []
    rng = __import__("random").Random(args.seed)

    for ei_step in range(1, args.n_ei_steps + 1):
        batch_size = question_batch_sizes[min(ei_step - 1, len(question_batch_sizes) - 1)]
        indices = rng.sample(range(len(questions)), min(batch_size, len(questions)))
        batch_questions = [questions[i] for i in indices]
        batch_ground_truths = [ground_truths[i] for i in indices]

        print(f"\n[EI Step {ei_step}] Sampled {len(batch_questions)} questions")

        # 保存当前策略供 vLLM 加载（需先释放训练模型引用）
        checkpoint_dir = ei_dir / f"ei_step_{ei_step:03d}" / "current_policy"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Rollout
        print(f"[EI Step {ei_step}] Generating {args.G} rollouts per question...")
        correct_examples = rollout_with_vllm(
            ei_dir, ei_step, batch_questions, batch_ground_truths, args=args
        )
        print(f"[EI Step {ei_step}] Kept {len(correct_examples)} correct response(s) for SFT")

        # SFT
        if correct_examples:
            run_sft_on_examples(device, model, correct_examples, tokenizer, args, ei_step, ei_dir)
        else:
            print(f"[EI Step {ei_step}] No correct SFT examples; skipping SFT")

        # Eval
        print(f"[EI Step {ei_step}] Saving policy + evaluating...")
        eval_ckpt_dir = ei_dir / f"ei_step_{ei_step:03d}" / "eval_checkpoint"
        eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(eval_ckpt_dir)
        tokenizer.save_pretrained(eval_ckpt_dir)

        # 把训练模型移到 CPU 并释放引用，让 vLLM 用另一份显存
        model.to("cpu")
        metrics = eval_checkpoint(device, model, eval_ckpt_dir, args, ei_step)
        model.to(device)
        model.train()

        if metrics:
            metrics["ei_step"] = ei_step
            all_metrics.append(metrics)
            print(f"[EI Step {ei_step}] Accuracy {metrics.get('mean_scores', {}).get('answer_reward', 0):.4f}")

    write_json(ei_dir / "all_ei_metrics.json", all_metrics)
    print(f"\nResults written to {ei_dir}")


if __name__ == "__main__":
    train_ei(parse_args())
