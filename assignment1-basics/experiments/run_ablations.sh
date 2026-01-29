#!/bin/bash
# Ablation Experiments Runner
# Runs 4 ablation experiments on 3 GPUs concurrently

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Qwen3
export PYTHONPATH=/mdata/wjx/CS336/assignment1-basics:$PYTHONPATH

cd /mdata/wjx/CS336/assignment1-basics

# Common parameters
TRAIN_DATA="output/tiny_stories_train.npy"
VAL_DATA="output/tiny_stories_valid.npy"
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
NUM_LAYERS=4
NUM_HEADS=16
BATCH_SIZE=32
NUM_STEPS=10000
LR="1e-3"
MIN_LR="1e-4"
WARMUP_STEPS=500
OUTPUT_BASE="experiments/ablations"

mkdir -p $OUTPUT_BASE

# Function to run ablation
run_ablation() {
    local NAME=$1
    local GPU=$2
    local EXTRA_ARGS=$3
    local THIS_LR=$4
    
    THIS_LR=${THIS_LR:-$LR}  # Use default LR if not specified
    
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    mkdir -p $OUTPUT_DIR/checkpoints $OUTPUT_DIR/runs
    
    echo "[$GPU] Starting: $NAME (LR=$THIS_LR)"
    
    CUDA_VISIBLE_DEVICES=$GPU nohup python cs336_basics/train.py \
        --train_data=$TRAIN_DATA \
        --val_data=$VAL_DATA \
        --vocab_size=$VOCAB_SIZE \
        --context_length=$CONTEXT_LENGTH \
        --d_model=$D_MODEL \
        --d_ff=$D_FF \
        --num_layers=$NUM_LAYERS \
        --num_heads=$NUM_HEADS \
        --batch_size=$BATCH_SIZE \
        --num_steps=$NUM_STEPS \
        --lr=$THIS_LR \
        --min_lr=$MIN_LR \
        --warmup_steps=$WARMUP_STEPS \
        --log_interval=100 \
        --eval_interval=500 \
        --eval_steps=50 \
        --save_interval=5000 \
        --checkpoint_dir=$OUTPUT_DIR/checkpoints \
        --log_dir=$OUTPUT_DIR/runs \
        --run_name=$NAME \
        --device=cuda:0 \
        $EXTRA_ARGS \
        > $OUTPUT_DIR/training.log 2>&1 &
    
    echo $! > $OUTPUT_DIR/pid.txt
    echo "[$GPU] PID: $(cat $OUTPUT_DIR/pid.txt)"
}

echo "================================================"
echo "Starting Ablation Experiments"
echo "================================================"

# 1. No RMSNorm (high LR)
run_ablation "no_rmsnorm_lr1e-3" "0" "--no_rmsnorm" "1e-3"

# 2. No RMSNorm (low LR for stability)
run_ablation "no_rmsnorm_lr1e-4" "1" "--no_rmsnorm" "1e-4"

# 3. Post-Norm
run_ablation "post_norm" "2" "--norm_type=post" "1e-3"

echo "First batch started. Waiting for completion..."
wait

echo "First batch done. Starting second batch..."

# 4. NoPE (No Position Embedding)
run_ablation "nope" "0" "--no_rope" "1e-3"

# 5. SiLU (parameter-matched: d_ff = 1344 * 1.5 = 2016)
run_ablation "silu_ffn" "1" "--ffn_type=silu --d_ff=2016" "1e-3"

# 6. Baseline (for comparison, already done in final_run_v2)
# run_ablation "baseline" "2" "" "1e-3"

echo "Second batch started. Waiting for completion..."
wait

echo "================================================"
echo "All ablation experiments completed!"
echo "Results in: $OUTPUT_BASE/"
echo "================================================"
