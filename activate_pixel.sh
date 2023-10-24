#module load nvidia/nvhpc/22.2


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

# set the job_env to slurm (for submitit)
export SUBMITIT_EXECUTOR=slurm

# set the offline mode for training with huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


