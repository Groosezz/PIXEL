#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

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
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=13000
#export OMP_NUM_THREADS=12

srun python ./scripts/test_render/train_projection_mode.py \
  ./new_configs/train_projection.json
#srun python ./scripts/test_render/train_projection.py \





