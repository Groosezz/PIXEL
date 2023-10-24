#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=qa


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
export HF_MODULES_CACHE='../cache/metrics'

# Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#export WORLD_SIZE=$((8))
echo "MASTER_PORT"=$MASTER_PORT
#echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export WANDB_DISABLED=true

  #--do_eval \
  #--evaluation_strategy=steps \
  #--eval_steps=500 \
  #--load_best_model_at_end=True \
  #--metric_for_best_model="eval_f1" \
  #--early_stopping \
  #--early_stopping_patience=5 \

# Settings
export DATASET_NAME="../cache/datasets/qa/squad"
export PROCESSOR="new_configs/panga/"
#export DATASET_CONFIG_NAME="secondary_task"
#export MODEL='../experiment/unshuffle/pretrain_bart_BSZ128_binary_focal'
#export MODEL='../experiment/unshuffle/bart_BSZ128_binary_focal_noNoise'
#export  MODEL='../cache/models/pixel-base-100000ckpt/'
#export  MODEL='../cache/models/pixel-base/'
#export MODEL='../experiment/unshuffle/bart_noise_grey'
export MODEL='../experiment/unshuffle/bart_binary_focal_span'
#export MODEL='../cache/models/pixel-base-100000ckpt/'
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export SEQ_LEN=400
export ENCODER=false
export STRIDE=160
export QUESTION_MAX_LEN=128
export BSZ=2
export GRAD_ACCUM=2
export LR=7e-5
export SEED=42
export NUM_STEPS=20000
  
export RUN_NAME="${DATASET_NAME}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
export output_dir="../experiment/train_qa/squad/Encoder_${ENCODER}/binary-span" 

srun --ntasks=16 --tasks-per-node=4 python scripts/our_model/run_qa_bart.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=${DATASET_NAME} \
  --dataset_config_name=${DATASET_CONFIG_NAME} \
  --remove_unused_columns=False \
  --do_train\
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --question_max_length=${QUESTION_MAX_LEN} \
  --doc_stride=${STRIDE} \
  --encoder_only=${ENCODER} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
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
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=5 \
  --report_to=none \
  --log_predictions \
  --fp16 \
  --half_precision_backend=auto \
  --seed=${SEED}
  #--fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
  
