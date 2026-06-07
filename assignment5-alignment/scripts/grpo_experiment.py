"""GRPO 训练（精简版）—— 模仿 EI 脚本的 GPU 管理方式。

每个 step：save ckpt → model.cpu() → vLLM rollout → model.cuda() → train
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from typing import Any, Callable
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cs336_alignment.evaluation import (
    load_jsonl, log_generations, question_only_numeric_reward_fn,
    r1_zero_numeric_reward_fn, read_prompt_template, build_math_prompts,
    write_json, write_jsonl,
)
from cs336_alignment.sft import SFTExample, load_sft_examples, make_sft_dataloader

# ---- 配置 ----
LOCAL_MODEL = PROJECT_ROOT / "models/Qwen/Qwen2___5-Math-1___5B"
MODEL = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "Qwen/Qwen2.5-Math-1.5B"
TRAIN = PROJECT_ROOT / "data/gsm8k/train.jsonl"
EVAL = PROJECT_ROOT / "data/gsm8k/test.jsonl"
PROMPT_R1 = PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt"
PROMPT_QO = PROJECT_ROOT / "cs336_alignment/prompts/question_only.prompt"
OUT = PROJECT_ROOT / "outputs/grpo_section8"
REWARDS = {"r1_zero_numeric": r1_zero_numeric_reward_fn, "question_only": question_only_numeric_reward_fn}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--train-data", default=str(TRAIN)); p.add_argument("--eval-data", default=str(EVAL))
    p.add_argument("--prompt", default="r1_zero"); p.add_argument("--output-dir", default=str(OUT))
    p.add_argument("--experiment-name", default="grpo"); p.add_argument("--reward-fn", default="r1_zero_numeric")
    p.add_argument("--n-steps", type=int, default=50); p.add_argument("--G", type=int, default=4)
    p.add_argument("--rollout-batch-size", type=int, default=16)
    p.add_argument("--normalize-by-std", action="store_true", default=True)
    p.add_argument("--no-normalize-by-std", dest="normalize_by_std", action="store_false")
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--loss-type", default="grpo_clip", choices=["no_baseline","reinforce_with_baseline","grpo_clip","grpo_no_clip"])
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--length-norm", default="masked_mean", choices=["masked_mean","masked_normalize"])
    p.add_argument("--normalize-constant", type=float, default=1024.0)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-sequence-length", type=int, default=1024)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens", type=int, default=1024)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--off-policy-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn-implementation", default="sdpa")
    return p.parse_args()


def resolve_prompt(arg: str) -> Path:
    return {"r1_zero": PROMPT_R1, "question_only": PROMPT_QO}.get(arg, Path(arg))


def get_stop(arg: str):
    return ["</answer>"] if arg == "r1_zero" else None


# ---- 核心函数 ----

def group_advantages(reward_fn, responses, gts, G, eps, use_std):
    raw = torch.tensor([float(reward_fn(r, g)["reward"]) for r, g in zip(responses, gts)], dtype=torch.float32)
    adv = torch.empty_like(raw)
    for g in range(len(raw) // G):
        grp = raw[g*G:(g+1)*G]; m = grp.mean()
        adv[g*G:(g+1)*G] = (grp - m) / (grp.std(unbiased=True) + eps) if use_std else grp - m
    meta = {"raw_reward_mean": float(raw.mean()), "raw_reward_std": float(raw.std(unbiased=True)),
            "raw_reward_min": float(raw.min()), "raw_reward_max": float(raw.max())}
    return adv, raw, meta


def log_probs_from_logits(logits, labels):
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def per_token_loss(adv, policy_lp, old_lp, loss_type, cliprange):
    if loss_type in ("no_baseline", "reinforce_with_baseline"):
        return -adv * policy_lp, {}
    ratio = torch.exp(policy_lp - old_lp)
    if loss_type == "grpo_no_clip":
        return -adv * ratio, {}
    cr = torch.clamp(ratio, 1-cliprange, 1+cliprange)
    loss_u, loss_c = adv * ratio, adv * cr
    return -torch.min(loss_u, loss_c), {"clip_fraction": (loss_u != loss_c).float().mean().detach()}


def vllm_rollout(model_path, questions, gts, prompt_tmpl, reward_fn, args):
    from vllm import LLM, SamplingParams
    prompts = [prompt_tmpl.format(question=q) for q in questions]
    llm = LLM(model=model_path, dtype="auto", gpu_memory_utilization=args.vllm_gpu_memory_utilization)
    sp = SamplingParams(temperature=args.sampling_temperature, max_tokens=args.sampling_max_tokens,
                        n=args.G, seed=args.seed, stop=get_stop(args.prompt), include_stop_str_in_output=True)
    outputs = llm.generate(prompts, sp)
    del llm
    P, R, G, S, recs = [], [], [], [], []
    for idx, (q, gt, prompt, out) in enumerate(zip(questions, gts, prompts, outputs, strict=True)):
        for c in out.outputs:
            scores = reward_fn(c.text, gt)
            P.append(prompt); R.append(c.text); G.append(gt); S.append(scores)
            recs.append({"question_idx": idx, "question": q, "ground_truth": gt, "response": c.text, "scores": scores})
    return P, R, G, S, recs


# ---- 训练 ----

def train(args):
    out_dir = Path(args.output_dir) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", vars(args))

    reward_fn = REWARDS[args.reward_fn]
    prompt_path = resolve_prompt(args.prompt)
    prompt_tmpl = read_prompt_template(prompt_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA_VISIBLE_DEVICES 已做映射
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    model.to(device); model.train()

    train_exs = load_sft_examples(args.train_data, dataset_format="gsm8k", prompt_template_path=prompt_path, seed=args.seed)
    questions = [ex.prompt for ex in train_exs]
    ground_truths = [ex.source.get("ground_truth", "") if isinstance(ex.source, dict) else "" for ex in train_exs]
    rng = __import__("random").Random(args.seed)

    total_steps = args.n_steps
    warmup = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    train_log = []
    progress = tqdm(range(1, total_steps + 1), desc=f"GRPO [{args.experiment_name}]")

    for step in progress:
        # 采样问题
        bs = min(args.rollout_batch_size, len(questions))
        idxs = rng.sample(range(len(questions)), bs)
        bq = [questions[i] for i in idxs]; bg = [ground_truths[i] for i in idxs]

        # 保存 checkpoint + 释放 GPU → vLLM rollout（模仿 EI 脚本的方式）
        ckpt = out_dir / "current_checkpoint"; ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)

        was_training = model.training
        model.to("cpu"); gc.collect(); torch.cuda.empty_cache()

        # vLLM rollout（若失败,保存已有结果后退出）
        try:
            prompts, responses, rep_gts, scores, _ = vllm_rollout(str(ckpt), bq, bg, prompt_tmpl, reward_fn, args)
        except Exception as e:
            print(f"[{args.experiment_name}] vLLM failed at step {step}: {e}", file=sys.stderr)
            write_jsonl(out_dir / "train_log.jsonl", train_log)
            (out_dir / "FAILED").touch()
            raise

        gc.collect(); torch.cuda.empty_cache()
        model.to(device)
        if was_training: model.train()

        # Advantage
        advantages, raw_rewards, reward_meta = group_advantages(reward_fn, responses, rep_gts, args.G, args.advantage_eps, args.normalize_by_std)

        # Dataloader（无 shuffle，保证 old/policy log_probs 对齐）
        exs = [SFTExample(prompt=p, response=r, source={"ground_truth": gt}) for p, r, gt in zip(prompts, responses, rep_gts)]
        dl = make_sft_dataloader(exs, tokenizer, args.per_device_batch_size, args.max_sequence_length, shuffle=False, seed=args.seed + step)

        # Old log_probs
        old_lp_dict = {}
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                ids, labs = batch["input_ids"].to(device), batch["labels"].to(device)
                olp = log_probs_from_logits(model(input_ids=ids).logits, labs).detach().cpu()
                for i in range(ids.shape[0]): old_lp_dict[bi * args.per_device_batch_size + i] = olp[i]

        # Policy gradient updates
        micro_step = 0; num_updates = 0
        optimizer.zero_grad(set_to_none=True)
        for _epoch in range(args.off_policy_epochs):
            for bi, batch in enumerate(dl):
                ids, labs, mask = batch["input_ids"].to(device), batch["labels"].to(device), batch["response_mask"].to(device)
                plp = log_probs_from_logits(model(input_ids=ids).logits, labs)
                b_start = bi * args.per_device_batch_size; n_s = ids.shape[0]
                olp_b = torch.stack([old_lp_dict[b_start + i].to(device) for i in range(n_s)])
                adv_b = advantages[b_start:b_start + n_s].to(device)
                ptl, lmeta = per_token_loss(adv_b.unsqueeze(-1), plp, olp_b, args.loss_type, args.cliprange)
                masked = ptl * mask.float()
                if args.length_norm == "masked_normalize":
                    per_s = masked.sum(dim=1) / args.normalize_constant
                else:
                    per_s = masked.sum(dim=1) / mask.float().sum(dim=1).clamp_min(1)
                loss = per_s.mean() / args.gradient_accumulation_steps
                loss.backward(); micro_step += 1
                if micro_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); num_updates += 1
        if micro_step % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); num_updates += 1

        scheduler.step()  # step the LR scheduler once per GRPO step (not per optimizer update)

        step_log = {"step": step, "lr": scheduler.get_last_lr()[0], "reward_metadata": reward_meta,
                     "num_rollouts": len(responses), "num_correct": int(raw_rewards.sum()),
                     "mean_reward": float(raw_rewards.mean()), "num_updates": num_updates}
        train_log.append(step_log)
        progress.set_postfix(reward=f"{step_log['mean_reward']:.3f}", correct=step_log["num_correct"])

        # 增量保存：每步完成后立即写入，避免最终崩溃导致数据丢失
        write_jsonl(out_dir / "train_log.jsonl", train_log)
        # 每 10 步保存一次 checkpoint
        if step % 10 == 0:
            model.save_pretrained(out_dir / f"checkpoint_step_{step:04d}")
            tokenizer.save_pretrained(out_dir / f"checkpoint_step_{step:04d}")

    (out_dir / "DONE").touch()
    print(f"[{args.experiment_name}] Done → {out_dir}")


if __name__ == "__main__":
    train(parse_args())
