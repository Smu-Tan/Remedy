import os
import torch
import argparse
import numpy as np
import pandas as pd
from vllm import ModelRegistry, LLM
from mt_metrics_eval import meta_info
from mt_metrics_eval import data as mtme
from datasets import load_from_disk, Dataset
from typing import Any, Dict, List, Set, Tuple
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from remedy.toolbox.languages import LANG_MAP, get_full_lang_name
from remedy.toolbox.gemma2_remedy import Gemma2ForSequenceClassification
from remedy.toolbox.score import _tokenize, prepare_dataset, initialize_model

# Register the model
ModelRegistry.register_model("Gemma2ForSequenceClassification", Gemma2ForSequenceClassification)


def collect_language_pairs(year: str) -> Tuple[Set[str], Set[str]]:
    """Collect MQM and DA language pairs from metadata."""
    mqm_lps = {lp for lp, data in meta_info.DATA[year].items() 
               if data and data.std_gold and data.std_gold.get('seg') == 'mqm'}
    
    da_lps = {lp for lp, data in meta_info.DATA[year].items() 
              if data and data.std_gold}
    
    return mqm_lps, da_lps

def get_wmt_mqm_data(year, lps):
    """Get MQM data for specified language pairs."""
    which_refs = {}
    df_all = pd.DataFrame([])
    for lp in lps:
        evs = mtme.EvalSet(year, lp, read_stored_metric_scores=False)
        level = 'seg'
        which_ref = meta_info.DATA[year][lp].std_ref
        gold_name = meta_info.DATA[year][lp].std_gold['seg']
        which_refs[lp] = which_ref
        print(f'For {year}, {lp}: {level, gold_name, which_ref}')
        gold_scores = evs.Scores(level, gold_name)
        lp_src, lp_ref, lp_sys, lp_system, seg_id, human_ratings = [], [], [], [], [], []

        if year=='wmt24' and lp == 'en-de':
            system_names = evs.sys_outputs.keys() #- evs.human_sys_names - evs.outlier_sys_names
            system_names = sorted(system_names)
            print('wmt24 system_names without filter: ', system_names)
        else:
            system_names = evs.sys_outputs.keys() - evs.human_sys_names - evs.outlier_sys_names
            system_names = sorted(system_names)
            print('system_names: ', system_names)

        for system_name in system_names:
            lp_src += evs.src
            lp_ref += evs.all_refs[which_ref]
            lp_sys += evs.sys_outputs[system_name]
            lp_system += [system_name] * len(evs.src)
            seg_id += list(range(len(evs.src)))
            human_ratings += gold_scores[system_name]

        data_dict = {
            'src': lp_src, 'ref': lp_ref, 'mt': lp_sys,
            'lp': lp, 'seg_id': seg_id, 'system-name': lp_system,
            'human_ratings': human_ratings}
        df = pd.DataFrame.from_dict(data_dict)
        df_all = pd.concat([df_all, df])
    return df_all, which_refs

def get_wmt_da_data(year, lps):
    """Get DA data for specified language pairs."""
    level = 'seg'
    df_all = pd.DataFrame([])
    which_refs = {}
    for lp in lps:
        which_ref = 'refA'
        if year == 'wmt22':
            gold_name = 'wmt-appraise'
        elif year == 'wmt23':
            gold_name = 'da-sqm'
            if lp == 'he-en':
                continue
        elif year == 'wmt24':
            gold_name = 'esa'
            if lp == 'en-de' or lp == 'ja-zh':
                continue
        evs = mtme.EvalSet(year, lp, read_stored_metric_scores=False)
        if evs.human_score_names == set():
            continue
        gold_scores = evs.Scores(level, gold_name)
        if gold_scores is None:
            gold_scores = evs.Scores(level, 'wmt')
        print(f'\nFor {year}, {lp, level, gold_name}')
        lp_src, lp_ref, lp_sys, lp_system, seg_id, human_ratings = [], [], [], [], [], []
        system_names = evs.sys_outputs.keys() - evs.human_sys_names - evs.outlier_sys_names
        system_names = sorted(system_names)
        if 'refA' in evs.all_refs.keys() and evs.all_refs['refA'] is not None:
            which_ref = 'refA'
            print('use refA')
        elif 'refB' in evs.all_refs.keys() and evs.all_refs['refB'] is not None:
            which_ref = 'refB'
            print('no refA found, use refB')
        elif 'refC' in evs.all_refs.keys() and evs.all_refs['refC'] is not None:
            which_ref = 'refC'
            print('no refA,refB found, use refC')
        else:
            raise Exception("no refA, B, C found...")
        for system_name in system_names:
            if system_name != 'PROMT':
                lp_src += evs.src
                lp_ref += evs.all_refs[which_ref]
                lp_sys += evs.sys_outputs[system_name]
                lp_system += [system_name] * len(evs.src)
                seg_id += list(range(len(evs.src)))
                human_ratings += gold_scores[system_name]
        which_refs[lp] = which_ref
        data_dict = {
            'src': lp_src, 'ref': lp_ref, 'mt': lp_sys,
            'lp': lp, 'seg_id': seg_id, 'system-name': lp_system,
            'human_ratings': human_ratings}
        df = pd.DataFrame.from_dict(data_dict)
        df_all = pd.concat([df_all, df])
    return df_all, which_refs

def process_test_data(df_all, QE=False):
    """Process test data into the format required by the model."""
    data_dict = {
        'lp': [], 'chosen': [], 'rejected': [], 'system_name': [],
    }
    
    if not QE:
        prompt_template = "Translate the following {src_lang} text into natural, fluent {tgt_lang} sentence while preserving the original meaning. You are also given a translation template.\n{src_lang}:{src_sent}\nTemplate:{ref_sent}\n{tgt_lang}:"
    else:
        prompt_template = "Translate the following {src_lang} text into natural, fluent {tgt_lang} sentence while preserving the original meaning:\n{src_lang}:{src_sent}\n{tgt_lang}:"
        print('QE enabled, now no reference...')
    
    print(f'prompt_template: {prompt_template}')

    for _, row in df_all.iterrows():
        src_lang, tgt_lang = row['lp'].split('-')
        src_name = get_full_lang_name(src_lang)
        tgt_name = get_full_lang_name(tgt_lang)
        
        if not QE:
            mt = [
                {'content': prompt_template.format(src_lang=src_name, tgt_lang=tgt_name, src_sent=str(row['src']), ref_sent=str(row['ref'])), 'role': 'user'},
                {'content': str(row['mt']), 'role': 'assistant'}
            ]
        else:
            mt = [
                {'content': prompt_template.format(src_lang=src_name, tgt_lang=tgt_name, src_sent=str(row['src'])), 'role': 'user'},
                {'content': str(row['mt']), 'role': 'assistant'}
            ]
            
        data_dict['chosen'].append(mt)
        data_dict['rejected'].append(mt)
        data_dict['system_name'].append(str(row['system-name']))
        data_dict['lp'].append(row['lp'])

    return Dataset.from_dict(data_dict), df_all

def run_inference(llm, dataset):
    """Run inference on prepared dataset."""
    combined_texts = dataset['input_ids_chosen']
    return llm.encode(
        prompt_token_ids=combined_texts,
        use_tqdm=True,
        lora_request=None,
        prompt_adapter_request=None
    )

def calculate_scores(embeddings, df):
    """Calculate scores from model embeddings."""
    r = np.array([x.outputs.embedding[0] for x in embeddings])
    df['raw:seg'] = r
    df['sigmoid:seg'] = torch.sigmoid(torch.tensor(df['raw:seg'].to_numpy())).numpy()
    return df

def save_results(df, which_refs, save_dir, year, metric_name, subset_name, lps):
    """Save segment and system level scores for all language pairs."""
    for lp in lps:
        if lp not in df['lp'].unique():
            print(f"Skipping {lp}: no data available")
            continue
            
        if lp not in which_refs:
            print(f"Skipping {lp}: no reference information available")
            continue
            
        lp_df = df[df['lp'] == lp]
        which_ref = which_refs[lp]

        # Base save directory including subset name
        base_save_dir = os.path.join(save_dir, year, subset_name, metric_name, lp)
        os.makedirs(base_save_dir, exist_ok=True)
        
        # Save raw scores
        save_score_files(lp_df, base_save_dir, metric_name, which_ref, 'raw:seg')
        
        # Save sigmoid scores
        sigmoid_metric_name = f"{metric_name}-sigmoid"
        sigmoid_save_dir = os.path.join(save_dir, year, subset_name, sigmoid_metric_name, lp)
        os.makedirs(sigmoid_save_dir, exist_ok=True)
        save_score_files(lp_df, sigmoid_save_dir, sigmoid_metric_name, which_ref, 'sigmoid:seg')
        
        print(f"Saved results for {lp} in {save_dir}")

def save_score_files(lp_df, save_dir, metric_name, which_ref, score_column):
    """Save segment and system level scores for a specific language pair."""
    # Save segment-level scores
    seg_save_path = os.path.join(save_dir, f'{metric_name}-{which_ref}.seg.score')
    lp_df[['system-name', score_column]].to_csv(
        seg_save_path,
        header=None, 
        sep='\t', 
        index=None
    )

    # Calculate and save system-level scores
    sys_save_path = os.path.join(save_dir, f'{metric_name}-{which_ref}.sys.score')
    system_scores = lp_df.groupby('system-name')[score_column].mean().reset_index()
    system_scores.to_csv(
        sys_save_path,
        header=None,
        sep='\t',
        index=None
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VLLM Evaluation Script')
    parser.add_argument('--year', type=str, required=True,
                      help='which WMT year')
    parser.add_argument('--save_metric_name', type=str, required=True,
                      help='The Metric Name to save.')
    parser.add_argument('--no_ref', action='store_true',
                      help='No Reference (QE) Mode.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the results')
    parser.add_argument('--num_gpus', type=int, default=1,
                      help='Number of GPUs to use')
    parser.add_argument('--num_seqs', type=int, default=256,
                      help='num seqs')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum sequence length')
    parser.add_argument('--enable_truncate', action='store_true',
                      help='Enable sequence truncation')
    parser.add_argument('--MQM_only', action='store_true',
                      help='Inference for MQM test set only.')
    parser.add_argument('--DA_only', action='store_true',
                      help='Inference for DA/DA+SQM/ESA test set only.')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate arguments
    if args.MQM_only and args.DA_only:
        raise ValueError("Cannot set both MQM_only and DA_only to True")
    
    QE = args.no_ref
    year = args.year
    
    # 1. Prepare Model
    llm = initialize_model(args.checkpoint_path, args.max_length, args.enable_truncate, args.num_gpus, args.num_seqs)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)
    
    # 2. Get language pairs
    mqm_lps, da_lps = collect_language_pairs(year)
    
    if args.MQM_only:
        lps = mqm_lps
        subset_name = "MQM"
    elif args.DA_only:
        lps = da_lps
        subset_name = "DA"
    else:
        lps = mqm_lps.union(da_lps)
        subset_name = "ALL"
    
    print(f'mqm_lps: {list(mqm_lps)}')
    print(f'da_lps: {list(da_lps)}')
    print(f'all lps: {list(lps)}')
    
    # 3. Process data based on selected mode
    if args.MQM_only:
        # MQM only mode
        df_mqm, which_refs_mqm = get_wmt_mqm_data(year, lps)
        ds_mqm, df_mqm = process_test_data(df_mqm, QE=QE)
        ds_mqm = prepare_dataset(ds_mqm, tokenizer, args.max_length, args.enable_truncate)
        
        # Run inference
        embeddings_mqm = run_inference(llm, ds_mqm)
        df_with_scores = calculate_scores(embeddings_mqm, df_mqm)
        
        # Save results
        save_results(
            df_with_scores, 
            which_refs_mqm, 
            args.save_dir, 
            year, 
            args.save_metric_name, 
            subset_name,
            which_refs_mqm.keys()  # Save only for language pairs with references
        )
    
    elif args.DA_only:
        # DA only mode
        df_da, which_refs_da = get_wmt_da_data(year, lps)
        ds_da, df_da = process_test_data(df_da, QE=QE)
        ds_da = prepare_dataset(ds_da, tokenizer, args.max_length, args.enable_truncate)
        
        # Run inference
        embeddings_da = run_inference(llm, ds_da)
        df_with_scores = calculate_scores(embeddings_da, df_da)
        
        # Save results
        save_results(
            df_with_scores, 
            which_refs_da, 
            args.save_dir, 
            year, 
            args.save_metric_name, 
            subset_name,
            which_refs_da.keys()  # Save only for language pairs with references
        )
    
    else:
        # Process both MQM and DA
        df_mqm, which_refs_mqm = get_wmt_mqm_data(year, mqm_lps)
        ds_mqm, df_mqm = process_test_data(df_mqm, QE=QE)
        ds_mqm = prepare_dataset(ds_mqm, tokenizer, args.max_length, args.enable_truncate)
        
        df_da, which_refs_da = get_wmt_da_data(year, da_lps - mqm_lps)  # Process only non-MQM language pairs for DA
        ds_da, df_da = process_test_data(df_da, QE=QE)
        ds_da = prepare_dataset(ds_da, tokenizer, args.max_length, args.enable_truncate)
        
        # Run inference for MQM
        embeddings_mqm = run_inference(llm, ds_mqm)
        df_mqm_scores = calculate_scores(embeddings_mqm, df_mqm)
        
        # Save MQM results
        save_results(
            df_mqm_scores, 
            which_refs_mqm, 
            args.save_dir, 
            year, 
            args.save_metric_name, 
            "MQM",
            which_refs_mqm.keys()
        )
        
        # Run inference for DA
        embeddings_da = run_inference(llm, ds_da)
        df_da_scores = calculate_scores(embeddings_da, df_da)
        
        # Save DA results
        save_results(
            df_da_scores, 
            which_refs_da, 
            args.save_dir, 
            year, 
            args.save_metric_name, 
            "DA",
            which_refs_da.keys()
        )

if __name__ == "__main__":
    main()