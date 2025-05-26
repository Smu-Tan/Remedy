#!/bin/bash

#SBATCH --job-name=download
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-20:00:00

pwd
source /home/stan1/anaconda3/bin/activate remedy
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "\.local/bin" | tr '\n' ':')
source /home/stan1/anaconda3/bin/activate remedy
export PATH="$PATH:$HOME/.local/bin"  # 添加到末尾
conda info --envs

export PATH=/home/stan1/anaconda3/envs/remedy/bin:$PATH
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-23 --local-dir Models/remedy-9B-23
