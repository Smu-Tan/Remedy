#!/bin/bash

#SBATCH --job-name=download
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-20:00:00

conda activate remedy

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-22 --local-dir Models/remedy-9B-22
#HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-23 --local-dir Models/remedy-9B-23
#HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-24 --local-dir Models/remedy-9B-24
