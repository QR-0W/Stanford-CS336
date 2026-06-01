#!/bin/bash
# Section 8 GRPO Experiments — 并行启动脚本
# 使用 Conf_Test conda 环境 (vLLM 0.18.1)
# 每个实验独占一个 GPU，按顺序调度

set -euo pipefail

PYTHON=/mdata/wjx/miniconda3/envs/Conf_Test/bin/python
SCRIPT=/mdata/wjx/CS336/assignment5-alignment/scripts/grpo_experiment.py
OUTDIR=/mdata/wjx/CS336/assignment5-alignment/outputs/grpo_section8

COMMON="
  --n-steps 50
  --G 4
  --rollout-batch-size 16
  --per-device-batch-size 1
  --gradient-accumulation-steps 4
  --learning-rate 1e-6
  --eval-every 10
  --eval-limit 128
  --max-sequence-length 1024
  --vllm-gpu-memory-utilization 0.65
  --loss-type grpo_clip
  --normalize-by-std
  --off-policy-epochs 1
  --seed 0
  --output-dir $OUTDIR
"

mkdir -p "$OUTDIR"

echo "============================================"
echo "Starting Section 8 GRPO experiments"
echo "Output: $OUTDIR"
echo "============================================"

# Wave 1: 3 experiments in parallel (GPU 0,1,2)
# Exp 8.1: LR sweep base (lr=1e-6)
echo "[$(date)] Launching Wave 1..."

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_1_lr_1e-6 \
  --learning-rate 1e-6 \
  > "$OUTDIR/exp8_1_lr_1e-6.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_2_no_baseline \
  --loss-type no_baseline \
  > "$OUTDIR/exp8_2_no_baseline.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_4_without_std \
  --no-normalize-by-std \
  > "$OUTDIR/exp8_4_without_std.log" 2>&1 &

echo "Wave 1 launched (3 jobs). Waiting..."
wait
echo "[$(date)] Wave 1 complete"

# Wave 2: 3 experiments in parallel
echo "[$(date)] Launching Wave 2..."

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_1_lr_5e-6 \
  --learning-rate 5e-6 \
  > "$OUTDIR/exp8_1_lr_5e-6.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_2_reinforce_baseline \
  --loss-type reinforce_with_baseline \
  > "$OUTDIR/exp8_2_reinforce_baseline.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_7_grpo_no_clip \
  --loss-type grpo_no_clip \
  > "$OUTDIR/exp8_7_grpo_no_clip.log" 2>&1 &

echo "Wave 2 launched (3 jobs). Waiting..."
wait
echo "[$(date)] Wave 2 complete"

# Wave 3: 3 experiments in parallel
echo "[$(date)] Launching Wave 3..."

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_1_lr_1e-5 \
  --learning-rate 1e-5 \
  > "$OUTDIR/exp8_1_lr_1e-5.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_3_masked_normalize \
  --length-norm masked_normalize \
  --loss-type reinforce_with_baseline \
  > "$OUTDIR/exp8_3_masked_normalize.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_5_off_policy_4 \
  --off-policy-epochs 4 \
  > "$OUTDIR/exp8_5_off_policy_4.log" 2>&1 &

echo "Wave 3 launched (3 jobs). Waiting..."
wait
echo "[$(date)] Wave 3 complete"

# Wave 4: remaining experiments
echo "[$(date)] Launching Wave 4..."

CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_1_lr_5e-7 \
  --learning-rate 5e-7 \
  > "$OUTDIR/exp8_1_lr_5e-7.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_6_off_bs32 \
  --rollout-batch-size 32 \
  --off-policy-epochs 1 \
  > "$OUTDIR/exp8_6_off_bs32.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT \
  $COMMON \
  --experiment-name exp8_8_question_only \
  --prompt question_only \
  --reward-fn question_only \
  --loss-type reinforce_with_baseline \
  > "$OUTDIR/exp8_8_question_only.log" 2>&1 &

echo "Wave 4 launched (3 jobs). Waiting..."
wait
echo "[$(date)] Wave 4 complete"

echo ""
echo "============================================"
echo "ALL Section 8 experiments COMPLETE!"
echo "Results: $OUTDIR"
echo "============================================"

# List completed experiments
echo ""
echo "Experiment directories:"
ls -d "$OUTDIR"/exp8_*/

# Check for DONE markers
echo ""
echo "Completion status:"
for d in "$OUTDIR"/exp8_*/; do
  name=$(basename "$d")
  if [ -f "$d/DONE" ]; then
    echo "  ✅ $name"
  else
    echo "  ❌ $name (no DONE marker)"
  fi
done
