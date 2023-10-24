#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pos


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

# Settings
export TREEBANK="UD_English-EWT"
export DATA_DIR="../cache/datasets/pos/ud-treebanks-v2.10/UD_English-EWT"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
#export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
#export MODEL='../experiment/unshuffle/pretrain_bart_BSZ128_binary_focal'
#export  MODEL='../cache/models/pixel-base-100000ckpt/'
#export MODEL='../experiment/unshuffle/bart_BSZ128_binary_focal_noNoise'
export MODEL='../experiment/unshuffle/bart_noise_grey'
#export MODEL='../experiment/unshuffle/bart_binary_focal_span'
export SEQ_LEN=256
export ENCODER=true #1 or 0
export BSZ=16
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=15000
export output_dir="../experiment/train_pos/Encoder_${ENCODER}/grey_noise" 
export RUN_NAME="${TREEBANK}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

srun --ntasks=16 --tasks-per-node=4 python scripts/our_model/run_pos_bart.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --data_dir=${DATA_DIR} \
  --rendering_backend=pygame\
  --do_train \
  --encoder_only=${ENCODER} \
  --do_eval \
  --do_predict \
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
  --report_to=none\
  --load_best_model_at_end=True \
  --metric_for_best_model="eval_accuracy" \
  --fp16 \
  --half_precision_backend=auto \
  --seed=${SEED}
