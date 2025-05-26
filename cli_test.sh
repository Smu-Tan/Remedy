#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G
#SBATCH --time=12:00:00

pwd
source /home/stan1/anaconda3/bin/activate remedy
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "\.local/bin" | tr '\n' ':')
source /home/stan1/anaconda3/bin/activate remedy
export PATH="$PATH:$HOME/.local/bin"  # 添加到末尾
conda info --envs


cd /ivi/ilps/datasets/shaomu/remedy-repo


port=$(( RANDOM % (50001 - 30000 + 1 ) + 31000 ))

####################################################
#################  5. evaluation ###################
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=20 

CHECKPOINT=/ivi/ilps/datasets/shaomu/remedy-repo/Models
SAVE_DIR=testcase


remedy-score \
    --model Models/remedy-9B-22 \
    --src_file testcase/en.src \
    --mt_file testcase/en-de.hyp \
    --ref_file testcase/de.ref \
    --src_lang en --tgt_lang de \
    --cache_dir $CHECKPOINT \
    --save_dir $SAVE_DIR \
    --num_gpus 4 \
    --calibrate
