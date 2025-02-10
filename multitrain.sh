#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

###############################################################################
# Common environment variables (unchanged between runs)
###############################################################################
export VOCAB_SIZE=32000   # 50304
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="c4"       # "slimpajama"

# Quantization configuration
export W_QUANT="FourEightMaskedQuantizer"
export A_QUANT="NoQuantizer"
export W_BITS=16
export A_BITS=16
export W_QUANT_KWARGS="{}"
export A_QUANT_KWARGS="{}"

###############################################################################
# 1) 30M configuration
###############################################################################
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export LR=0.0012
export TOKENS=3000000000 # 3B
export MODEL_SIZE_PREFIX="30M"

# Calculate iterations and warmup steps
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}-${DATASET}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "===== Running 30M configuration ====="
# torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
#     --distributed-backend nccl \
#     --dataset ${DATASET} \
#     --model llama \
#     --compile \
#     --latest-ckpt-interval 20000 \
#     --acc-steps ${ACC_STEPS} \
#     --batch-size ${BATCH_SIZE} \
#     --wandb \
#     --wandb-project "llm-baselines" \
#     --wandb-run-prefix "${WANDB_PREFIX}" \
#     --n-layer ${N_LAYER} \
#     --n-embd ${N_EMBD} \
#     --n-head ${N_HEAD} \
#     --warmup-steps ${WARMUP_STEPS} \
#     --iterations ${ITERATIONS} \
#     --lr ${LR} \
#     --w-quant ${W_QUANT} \
#     --w-quant-kwargs "${W_QUANT_KWARGS}" \
#     --a-quant ${A_QUANT} \
#     --a-quant-kwargs "${A_QUANT_KWARGS}"

###############################################################################
# 2) 50M configuration
###############################################################################
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6
export LR=0.0012
export TOKENS=5000000000       # 5B
export MODEL_SIZE_PREFIX="50M"

# Calculate iterations and warmup steps
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}-${DATASET}"

echo "===== Running 50M configuration ====="
torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --model llama \
    --compile \
    --latest-ckpt-interval 20000 \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "llm-baselines" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"

###############################################################################
# 3) 100M configuration
###############################################################################
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8
export LR=0.0006
export TOKENS=10000000000      # 10B
export MODEL_SIZE_PREFIX="100M"

# Calculate iterations and warmup steps
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}-${DATASET}"

echo "===== Running 100M configuration ====="
torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --model llama \
    --compile \
    --latest-ckpt-interval 20000 \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "llm-baselines" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"

###############################################################################
# 4) 200M configuration
###############################################################################
export N_LAYER=10
export N_EMBD=1280
export N_HEAD=10
export LR=0.0003
export TOKENS=20000000000      # 20B
export MODEL_SIZE_PREFIX="200M"

# Calculate iterations and warmup steps
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}-${DATASET}"

echo "===== Running 200M configuration ====="
torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --model llama \
    --compile \
    --latest-ckpt-interval 20000 \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --wandb \
    --wandb-project "llm-baselines" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"
