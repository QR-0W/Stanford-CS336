#!/bin/bash
# OpenWebText Training Run
# Same architecture as TinyStories but trained on OWT (Vocab=32k)

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Qwen3
export PYTHONPATH=/mdata/wjx/CS336/assignment1-basics:$PYTHONPATH

cd /mdata/wjx/CS336/assignment1-basics

# OWT Config
TRAIN_DATA="output/owt_train.npy"
VAL_DATA="output/owt_valid.npy"
VOCAB_SIZE=32000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
NUM_LAYERS=4
NUM_HEADS=16
BATCH_SIZE=32
NUM_STEPS=40000
WARMUP_STEPS=1000
OUTPUT_BASE="experiments/owt"

mkdir -p $OUTPUT_BASE

# Function to run training
run_owt() {
    local NAME=$1
    local GPU=$2
    local LR=$3
    local MIN_LR=$4
    
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    mkdir -p $OUTPUT_DIR/checkpoints $OUTPUT_DIR/runs
    
    echo "[$GPU] Starting OWT Run: $NAME (LR=$LR)"
    
    CUDA_VISIBLE_DEVICES=$GPU nohup python -u cs336_basics/train.py \
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
        --lr=$LR \
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
        > $OUTPUT_DIR/training.log 2>&1 &
    
    echo $! > $OUTPUT_DIR/pid.txt
    echo "[$GPU] PID: $(cat $OUTPUT_DIR/pid.txt)"
}

echo "================================================"
echo "Starting OpenWebText Experiments"
echo "================================================"

# Run 1: Same LR as TinyStories (1e-3)
run_owt "owt_lr1e-3" "0" "1e-3" "1e-4"

# Run 2: Lower LR (5e-4) - safer for OWT
run_owt "owt_lr5e-4" "1" "5e-4" "5e-5"

echo "OWT experiments launched. Waiting for completion..."
wait

echo "================================================"
echo "OWT experiments completed!"
echo "================================================"
