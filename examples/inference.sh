#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=400G

conda activate remedy

####################################################
################### evaluation #####################
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=80 

# default
remedy-score \
    --model ShaomuTan/ReMedy-9B-22 \
    --src_file ./testcase/en.src \
    --mt_file ./testcase/en-de.hyp \
    --ref_file ./testcase/de.ref \
    --src_lang en --tgt_lang de \
    --cache_dir Models \
    --save_dir ./testcase \
    --num_gpus 4 \
    --calibrate

# QE Mode
remedy-score \
    --model ShaomuTan/ReMedy-9B-22 \
    --src_file ./testcase/en.src \
    --mt_file ./testcase/en-de.hyp \
    --no_ref \
    --src_lang en --tgt_lang de \
    --cache_dir Models \
    --save_dir ./testcase/QE \
    --num_gpus 4 \
    --calibrate