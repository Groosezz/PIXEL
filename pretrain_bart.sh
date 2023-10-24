#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pixel



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




#srun torchrun --nnodes=2\
 #--nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
 #--rdzv_endpoint=127.0.0.1 --master_port=$MASTER_PORT ./scripts/our_model/pretrain_bart.py \
 # ./new_configs/pretrain_bart.json
 srun --ntasks=16 --tasks-per-node=4 python ./scripts/our_model/pretrain_bart.py ./new_configs/pretrain_bart.json






