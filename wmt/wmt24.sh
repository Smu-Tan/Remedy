#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

conda activate remedy

####################################################
################### evaluation #####################
#export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=80 

NAME=REMEDY-9b
SIGMOID_NAME=REMEDY-9b-sigmoid
YEAR=wmt24
CHECKPOINT=Models/remedy-9B-24
SAVE_DIR=.
LOG_DIR=$YEAR/results


mkdir -p $LOG_DIR/MQM
mkdir -p $LOG_DIR/ESA


HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-24 --local-dir Models/remedy-9B-24

#### ------------ 1. wmt24 MQM ------------

# remedy inference
python -m remedy.toolbox.eval_wmt.WMT \
    --year $YEAR \
    --save_metric_name $NAME \
    --checkpoint_path $CHECKPOINT \
    --save_dir $SAVE_DIR \
    --num_gpus 4 \
    --MQM_only

# calibration
CALI_MQM_DIR=$YEAR/MQM/$NAME-calibration
python remedy/toolbox/calibration.py \
    --raw_score_dir $YEAR/MQM/$NAME \
    --output_score_dir $CALI_MQM_DIR\
    --per_lp_temp \


##### ------------ 1. wmt24 ESA ------------
# remedy inference
python -m remedy.toolbox.eval_wmt.WMT \
    --year $YEAR \
    --save_metric_name $NAME \
    --checkpoint_path $CHECKPOINT \
    --save_dir $SAVE_DIR \
    --num_gpus 4 \
    --DA_only

# calibration
CALI_DA_DIR=$YEAR/DA/$NAME-calibration
python remedy/toolbox/calibration.py \
    --raw_score_dir $YEAR/DA/$NAME \
    --output_score_dir $CALI_DA_DIR \
    --per_lp_temp \

# MQM
python remedy/toolbox/eval_wmt/extract_wmt24_mqm.py \
    REMEDY-9b wmt24/MQM/REMEDY-9b \
    2>&1 >> $LOG_DIR/MQM/remedy_raw_results.log
python remedy/toolbox/eval_wmt/extract_wmt24_mqm.py \
    REMEDY-9b-sigmoid wmt24/MQM/REMEDY-9b-sigmoid \
    2>&1 >> $LOG_DIR/MQM/remedy_sigmoid_results.log

python remedy/toolbox/eval_wmt/extract_wmt24_mqm.py \
    REMEDY-9b wmt24/MQM/REMEDY-9b-calibration \
    2>&1 >> $LOG_DIR/MQM/remedy_calibration_results.log

# ESA
python remedy/toolbox/eval_wmt/extract_wmt24_esa.py \
    REMEDY-9b wmt24/DA/REMEDY-9b \
    2>&1 >> $LOG_DIR/ESA/remedy_raw_results.log
python remedy/toolbox/eval_wmt/extract_wmt24_esa.py \
    REMEDY-9b-sigmoid wmt24/DA/REMEDY-9b-sigmoid \
    2>&1 >> $LOG_DIR/ESA/remedy_sigmoid_results.log
python remedy/toolbox/eval_wmt/extract_wmt24_esa.py \
    REMEDY-9b wmt24/DA/REMEDY-9b-calibration \
    2>&1 >> $LOG_DIR/ESA/remedy_calibration_results.log