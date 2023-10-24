#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pixel-test

export PARENT=`/bin/hostname -s`
export MPORT=13000
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
echo $HOSTLIST


source activate_pixel.sh

# set the job_env to slurm (for submitit)
export SUBMITIT_EXECUTOR=slurm

# set the offline mode for training with huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
#export OMP_NUM_THREADS=12

srun python -m torch.distributed.launch\
 --nproc_per_node 2 \
 --node_rank=$SLURM_PROCID \
 --master_addr="$PARENT" --master_port="$MPORT" \
  ./scripts/training/run_pretraining_test.py \
  ./configs/all_configs_test.json




