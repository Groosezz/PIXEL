#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=sc118

#SBATCH --job-name=pixel-test



# Load the required modules
module load nvidia/nvhpc

#activate the base  = source .bashrc
__conda_setup="$('/work/sc118/sc118/xliao11/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/sc118/sc118/xliao11/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/work/sc118/sc118/xliao11/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/work/sc118/sc118/xliao11/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup


#activate the pixel-env
conda activate pixel-env

srun ./scripts/training/run_pretraining.py \
./configs/all_configs_test.json
