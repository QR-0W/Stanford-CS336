#!/bin/bash
# GRPO Section 8 — 完整实验顺序执行脚本
# 每轮 2 个实验并行 (GPU 1, 2)，分 6 轮完成

PY=/mdata/wjx/miniconda3/envs/Conf_Test/bin/python
SCRIPT=/mdata/wjx/CS336/assignment5-alignment/scripts/grpo_experiment.py
OUT=/mdata/wjx/CS336/assignment5-alignment/outputs/grpo_section8
mkdir -p "$OUT"

B="--n-steps 50 --G 4 --rollout-batch-size 16 --per-device-batch-size 1 --gradient-accumulation-steps 4 --eval-every 10 --eval-limit 128 --max-sequence-length 1024 --vllm-gpu-memory-utilization 0.65 --seed 0 --output-dir $OUT"

run() {
  local gpu=$1; shift
  local name=$1; shift
  echo "[$(date)] [$name] Starting on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT $B "$@" --experiment-name "$name" > "$OUT/${name}.log" 2>&1
  echo "[$(date)] [$name] Done (exit=$?)"
}

# W1: baseline lr=1e-6 + without std (Dr. GRPO)  [ALREADY RUNNING]
# W2: lr=5e-6 + reinforcement_with_baseline
# W3: lr=1e-5 + grpo_no_clip
# W4: lr=5e-7 + masked_normalize
# W5: off_policy_4 + bs32
# W6: question_only + (extra)

echo "=== WAVE 2 ==="
run 1 exp8_1_lr_5e-6 --learning-rate 5e-6 --loss-type grpo_clip --normalize-by-std --off-policy-epochs 1 &
run 2 exp8_2_reinforce_baseline --loss-type reinforce_with_baseline --normalize-by-std --off-policy-epochs 1 &
wait

echo "=== WAVE 3 ==="
run 1 exp8_1_lr_1e-5 --learning-rate 1e-5 --loss-type grpo_clip --normalize-by-std --off-policy-epochs 1 &
run 2 exp8_7_grpo_no_clip --loss-type grpo_no_clip --normalize-by-std --off-policy-epochs 1 &
wait

echo "=== WAVE 4 ==="
run 1 exp8_1_lr_5e-7 --learning-rate 5e-7 --loss-type grpo_clip --normalize-by-std --off-policy-epochs 1 &
run 2 exp8_3_masked_normalize --loss-type reinforce_with_baseline --length-norm masked_normalize --normalize-by-std --off-policy-epochs 1 &
wait

echo "=== WAVE 5 ==="
run 1 exp8_5_off_policy_4 --loss-type grpo_clip --off-policy-epochs 4 --normalize-by-std &
run 2 exp8_2_no_baseline --loss-type no_baseline --normalize-by-std --off-policy-epochs 1 &
wait

echo "=== WAVE 6 ==="
run 1 exp8_6_off_bs32 --loss-type grpo_clip --off-policy-epochs 1 --rollout-batch-size 32 --normalize-by-std &
run 2 exp8_8_question_only --loss-type reinforce_with_baseline --prompt question_only --reward-fn question_only --normalize-by-std --off-policy-epochs 1 &
wait

echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
for d in "$OUT"/exp8_*/; do
  name=$(basename "$d")
  if [ -f "$d/DONE" ]; then
    echo "OK $name"
  else
    echo "MISSING $name"
  fi
done
