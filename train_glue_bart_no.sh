#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pixel-test


export TORCH_EXTENSIONS_DIR='../deepspeed'


source activate_pixel.sh

# set the job_env to slurm (for submitit)
export SUBMITIT_EXECUTOR=slurm

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# set the offline mode for training with huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#export WORLD_SIZE=$((8))
echo "MASTER_PORT"=$MASTER_PORT
#echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_DISABLED=true

# Settings
export TASK="mnli"
export DATA_CACHE="../cache/datasets/glue/mnli"
#export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
#export MODEL='../experiment/unshuffle/pretrain_bart_BSZ128_binary_focal'
export MODEL='../experiment/unshuffle/bart_BSZ128_binary_focal_noNoise'
export RENDERING_BACKEND="pygame"  # Consider trying out both "pygame" and "pangocairo" to see which one works best
export POOLING_MODE="max" # Can be "mean", "max", "cls", or "pma1" to "pma8"
export SEQ_LEN=256
export BSZ=16
export GRAD_ACCUM=2  # We found that higher batch sizes can sometimes make training more stable
export LR=3e-5
export SEED=42
export NUM_STEPS=15000
export RUN_NAME="${TASK}-$(basename ${MODEL})-${POOLING_MODE}-${RENDERING_BACKEND}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
export output_dir="../experiment/train_glue/${TASK}/NoNoise-${POOLING_MODE}" 

srun --ntasks=8 --tasks-per-node=4 python scripts/our_model/run_glue_bart.py \
  --model_name_or_path=${MODEL} \
  --task_name=${TASK} \
  --data_cache_dir=${DATA_CACHE} \
  --rendering_backend=${RENDERING_BACKEND} \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --eval_accumulation_steps=1 \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name=${RUN_NAME} \
  --output_dir=${output_dir} \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=5 \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --fp16 \
  --half_precision_backend=auto \
  --seed=${SEED}
