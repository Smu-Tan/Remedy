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
YEAR=wmt22
CHECKPOINT=Models/remedy-9B-22
SAVE_DIR=.
LOG_DIR=$YEAR/results


mkdir -p $LOG_DIR/MQM/raw/seg
mkdir -p $LOG_DIR/MQM/sigmoid/seg
mkdir -p $LOG_DIR/MQM/calibration/seg
mkdir -p $LOG_DIR/DA/raw/seg
mkdir -p $LOG_DIR/DA/sigmoid/seg
mkdir -p $LOG_DIR/DA/calibration/seg


HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ShaomuTan/ReMedy-9B-22 --local-dir Models/remedy-9B-22

#### ------------ 1. wmt22 MQM ------------

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

## SYSTEM Level
# raw
python3 -m mt_metrics_eval.mtme -t $YEAR -l en-de,zh-en,en-ru \
    --matrix --matrix_corr accuracy --k_block 5 \
    --add_metrics_from_dir $YEAR/MQM/$NAME \
    2>&1 | tee $LOG_DIR/MQM/raw/sys_results.log

# sigmoid
python3 -m mt_metrics_eval.mtme -t $YEAR -l en-de,zh-en,en-ru \
    --matrix --matrix_corr accuracy --k_block 5 \
    --add_metrics_from_dir $YEAR/MQM/$SIGMOID_NAME \
    2>&1 | tee $LOG_DIR/MQM/sigmoid/sys_results.log
# calibration
python3 -m mt_metrics_eval.mtme -t $YEAR -l en-de,zh-en,en-ru \
    --matrix --matrix_corr accuracy --k_block 5 \
    --add_metrics_from_dir $CALI_MQM_DIR \
    2>&1 | tee $LOG_DIR/MQM/calibration/sys_results.log


## SEGMENT Level
language_pairs=(
    "en-de"
    "en-ru"
    "zh-en"
)
for lp in "${language_pairs[@]}"; do
    # raw
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $YEAR/MQM/$NAME \
        2>&1 >> $LOG_DIR/MQM/raw/seg/${lp}_results.log

    # sigmoid
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $YEAR/MQM/$SIGMOID_NAME \
        2>&1 >> $LOG_DIR/MQM/sigmoid/seg/${lp}_results.log

    # calibration
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $CALI_MQM_DIR \
        2>&1 >> $LOG_DIR/MQM/calibration/seg/${lp}_results.log
done


####### ------------ 1. wmt22 DA ------------
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


python3 -m mt_metrics_eval.mtme -t $YEAR \
    -l cs-uk,en-cs,en-de,en-hr,en-ja,en-liv,en-ru,en-uk,en-zh,liv-en,sah-ru,uk-cs,zh-en \
    --matrix --k_block 5 --matrix_corr accuracy \
    --g wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise \
    --add_metrics_from_dir $YEAR/DA/$NAME \
    2>&1 >> $LOG_DIR/DA/raw/sys_results.log

python3 -m mt_metrics_eval.mtme -t $YEAR \
    -l cs-uk,en-cs,en-de,en-hr,en-ja,en-liv,en-ru,en-uk,en-zh,liv-en,sah-ru,uk-cs,zh-en \
    --matrix --k_block 5 --matrix_corr accuracy \
    --g wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise \
    --add_metrics_from_dir $YEAR/DA/$SIGMOID_NAME \
    2>&1 >> $LOG_DIR/DA/sigmoid/sys_results.log

python3 -m mt_metrics_eval.mtme -t $YEAR \
    -l cs-uk,en-cs,en-de,en-hr,en-ja,en-liv,en-ru,en-uk,en-zh,liv-en,sah-ru,uk-cs,zh-en \
    --matrix --k_block 5 --matrix_corr accuracy \
    --g wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise,wmt-appraise \
    --add_metrics_from_dir $CALI_DA_DIR \
    2>&1 >> $LOG_DIR/DA/calibration/sys_results.log

language_pairs=(
    "cs-uk"
    "en-cs"
    "en-de"
    "en-hr"
    "en-ja"
    "en-liv"
    "en-ru"
    "en-uk"
    "en-zh"
    "liv-en"
    "sah-ru"
    "uk-cs"
    "zh-en"
)

for lp in "${language_pairs[@]}"; do
    # raw
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item --g wmt-appraise \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $YEAR/DA/$NAME \
        2>&1 >> $LOG_DIR/DA/raw/seg/${lp}_results.log
    # sigmoid
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item --g wmt-appraise \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $YEAR/DA/$SIGMOID_NAME \
        2>&1 >> $LOG_DIR/DA/sigmoid/seg/${lp}_results.log
    # calibration
    python3 -m mt_metrics_eval.mtme -t $YEAR -l "$lp" \
        --matrix --matrix_level seg --avg item --g wmt-appraise \
        --matrix_corr KendallWithTiesOpt --matrix_perm_test pairs \
        --matrix_corr_args "{'variant':'acc23', 'sample_rate':1.0}" --k 0 \
        --add_metrics_from_dir $CALI_DA_DIR \
        2>&1 >> $LOG_DIR/DA/calibration/seg/${lp}_results.log
    
done

python remedy/toolbox/eval_wmt/extract_result_wmt22.py \
    --result_dir wmt22/results \
    --mqm_folder MQM/raw \
    --da_folder DA/raw \
    --system_name $NAME \
    2>&1 >> $LOG_DIR/remedy_raw_results.log

python remedy/toolbox/eval_wmt/extract_result_wmt22.py \
    --result_dir wmt22/results \
    --mqm_folder MQM/sigmoid \
    --da_folder DA/sigmoid \
    --system_name $SIGMOID_NAME \
    2>&1 >> $LOG_DIR/remedy_sigmoid_results.log


python remedy/toolbox/eval_wmt/extract_result_wmt22.py \
    --result_dir wmt22/results \
    --mqm_folder MQM/calibration \
    --da_folder DA/calibration \
    --system_name $NAME \
    2>&1 >> $LOG_DIR/remedy_calibration_results.log