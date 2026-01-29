#!/bin/bash
# Final Training Run
# Target: Validation Loss < 1.45
# Config: 40k steps, LR=1e-3, Batch 32, Context 256 (Total ~327M tokens)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Qwen3

export PYTHONPATH=/mdata/wjx/CS336/assignment1-basics:$PYTHONPATH

# Create output directory
mkdir -p experiments/final_run

RUN_NAME="final_run_optimal_lr_1e-3"
OUTPUT_DIR="experiments/final_run"

echo "Starting final training run: $RUN_NAME"

nohup python cs336_basics/train.py \
    --train_data=output/tiny_stories_train.npy \
    --val_data=output/tiny_stories_valid.npy \
    --vocab_size=10000 \
    --context_length=256 \
    --d_model=512 \
    --d_ff=1344 \
    --num_layers=4 \
    --num_heads=16 \
    --rope_theta=10000.0 \
    --batch_size=32 \
    --num_steps=40000 \
    --lr=1e-3 \
    --weight_decay=0.1 \
    --beta1=0.9 \
    --beta2=0.95 \
    --eps=1e-8 \
    --grad_clip=1.0 \
    --log_interval=100 \
    --eval_interval=500 \
    --eval_steps=50 \
    --save_interval=5000 \
    --checkpoint_dir=$OUTPUT_DIR/checkpoints \
    --log_dir=$OUTPUT_DIR/runs \
    --run_name=$RUN_NAME \
    --device=cuda:0 \
    > $OUTPUT_DIR/training.log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Logs: $OUTPUT_DIR/training.log"
