#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pixel-test

# Load the required modules
module load nvidia/nvhpc

#activate the pixel-env 
source .bashrc
conda activate pixel-env

srun ./scripts/training/run_pretraining.py
