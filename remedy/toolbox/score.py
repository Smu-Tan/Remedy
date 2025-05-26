import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import Any, Dict, List
from vllm import ModelRegistry, LLM
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from remedy import __version__
from remedy.toolbox.languages import LANG_MAP, get_full_lang_name
from remedy.toolbox.gemma2_remedy import Gemma2ForSequenceClassification
from remedy.toolbox.calibration import sigmoid, final_optimized_entropy

# Register the model
ModelRegistry.register_model("Gemma2ForSequenceClassification", Gemma2ForSequenceClassification)

def _tokenize(batch: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase, max_length=None) -> Dict[str, List[Any]]:
    """Tokenize a batch of data."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
    }
    for chosen in batch["chosen"]:
        if max_length:
            tokenized_chosen = tokenizer(chosen, max_length=max_length, truncation=True)
        else:
            tokenized_chosen = tokenizer(chosen)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
    return new_examples

def prepare_dataset(test_dataset, tokenizer, max_length, enable_truncate=True):
    """Prepare dataset for model inference."""
    if "input_ids_chosen" not in test_dataset.column_names:
        test_dataset = test_dataset.map(
            maybe_apply_chat_template, 
            fn_kwargs={"tokenizer": tokenizer}
        )
        
        fn_kwargs = {
            "tokenizer": tokenizer,
            'max_length': max_length
        } if enable_truncate else {"tokenizer": tokenizer}
            
        test_dataset = test_dataset.map(
            _tokenize,
            fn_kwargs=fn_kwargs,
            batched=True,
            num_proc=20,
        )
    return test_dataset

def initialize_model(model_path, max_length, enable_truncate=True, num_gpus=1, num_seqs=256, cache_dir=None):
    """Initialize the LLM model for inference."""
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    model_kwargs = {
        "model": model_path,
        "tensor_parallel_size": num_gpus,
        "gpu_memory_utilization": 0.99,
        "dtype": "bfloat16",
        "task": 'embedding',
        "enforce_eager": False,
        "disable_log_stats": True,
        "device": 'cuda',
        "max_num_seqs": num_seqs,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False
    }
        
    if enable_truncate:
        model_kwargs["max_model_len"] = max_length
        
    return LLM(**model_kwargs)

def load_translation_data(src_file, mt_file, ref_file, src_lang, tgt_lang):
    """Load translation data from provided files."""
    # Read source file
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f.readlines()]
    
    # Read MT file
    with open(mt_file, 'r', encoding='utf-8') as f:
        mt_lines = [line.strip() for line in f.readlines()]
    
    # Check if source and MT files have the same number of lines
    if len(src_lines) != len(mt_lines):
        raise ValueError(f"Source file has {len(src_lines)} lines, but MT file has {len(mt_lines)} lines")
    
    # Read reference file if provided
    ref_lines = None
    if ref_file:
        with open(ref_file, 'r', encoding='utf-8') as f:
            ref_lines = [line.strip() for line in f.readlines()]
        if len(ref_lines) != len(src_lines):
            raise ValueError(f"Reference file has {len(ref_lines)} lines, but source file has {len(src_lines)} lines")
    
    # Create language pair
    lp = f"{src_lang}-{tgt_lang}"
    
    # Create DataFrame
    data_dict = {
        'src': src_lines,
        'mt': mt_lines,
        'lp': [lp] * len(src_lines),
        'seg_id': list(range(len(src_lines))),
        'system-name': ['custom_system'] * len(src_lines),
        'human_ratings': [0.0] * len(src_lines)  # Dummy ratings since we don't have them
    }
    
    if ref_lines:
        data_dict['ref'] = ref_lines
    else:
        data_dict['ref'] = [''] * len(src_lines)  # Empty refs for QE mode
    
    df = pd.DataFrame.from_dict(data_dict)
    
    return df

def process_data_for_scoring(df_all, QE=False):
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

def calculate_scores(embeddings, df, calibrate=False):
    """Calculate scores from model embeddings."""
    r = np.array([x.outputs.embedding[0] for x in embeddings])
    df['raw:seg'] = r
    df['sigmoid:seg'] = torch.sigmoid(torch.tensor(df['raw:seg'].to_numpy())).numpy()
    
    # Apply calibration if requested
    if calibrate:
        # Find optimal temperature using entropy-based calibration
        raw_scores = df['raw:seg'].to_numpy()
        best_temp = final_optimized_entropy(raw_scores)
        print(f"Calibration: Using optimal temperature {best_temp:.4f}")
        
        # Apply calibration with the optimal temperature
        df['calibration:seg'] = sigmoid(raw_scores, best_temp)
        df['calibration_temp'] = best_temp
    
    return df

def save_score_results(df, save_dir, metric_name, src_lang, tgt_lang, args=None):
    """Save scoring results in a structured format."""
    lp = f"{src_lang}-{tgt_lang}"
    
    # Create save directory
    base_save_dir = os.path.join(save_dir, metric_name)
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Calculate system-level scores (overall averages)
    system_raw_score = df['raw:seg'].mean()
    system_sigmoid_score = df['sigmoid:seg'].mean()
    
    # Calculate calibration score if available
    has_calibration = 'calibration:seg' in df.columns
    system_calibration_score = df['calibration:seg'].mean() if has_calibration else None
    calibration_temp = df['calibration_temp'].iloc[0] if has_calibration else None
    
    # Save segment-level raw scores
    raw_save_path = os.path.join(base_save_dir, f'{lp}_raw_scores.txt')
    with open(raw_save_path, 'w', encoding='utf-8') as f:
        for score in df['raw:seg']:
            f.write(f"{score}\n")
    
    # Save segment-level sigmoid scores
    sigmoid_save_path = os.path.join(base_save_dir, f'{lp}_sigmoid_scores.txt')
    with open(sigmoid_save_path, 'w', encoding='utf-8') as f:
        for score in df['sigmoid:seg']:
            f.write(f"{score}\n")
    
    # Save calibration scores if available
    if has_calibration:
        calibration_save_path = os.path.join(base_save_dir, f'{lp}_calibration_scores.txt')
        with open(calibration_save_path, 'w', encoding='utf-8') as f:
            for score in df['calibration:seg']:
                f.write(f"{score}\n")
    
    # Save detailed results with source, MT, and scores
    detailed_save_path = os.path.join(base_save_dir, f'{lp}_detailed_results.tsv')
    columns_to_save = ['src', 'mt', 'ref', 'raw:seg', 'sigmoid:seg']
    if has_calibration:
        columns_to_save.append('calibration:seg')
    
    df[columns_to_save].to_csv(
        detailed_save_path,
        sep='\t',
        index=False
    )
    
    # Create result JSON with metadata
    result = {
        "metric_name": metric_name,
        "raw_score": float(system_raw_score),  # Original raw score before sigmoid
        "sigmoid_score": float(system_sigmoid_score),  # Sigmoid transformed score (0-1 range)
    }
    
    # Add calibration info if available
    if has_calibration:
        result["calibration_score"] = float(system_calibration_score)
        result["calibration_temp"] = float(calibration_temp)
    
    # Add other metadata
    result.update({
        "signature": f"metric_name:{metric_name}|lp:{lp}|ref:{'yes' if 'ref' in df.columns and not df['ref'].isna().all() else 'no'}|version:{__version__}",
        "language_pair": lp,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "segments": len(df),
        "version": __version__
    })
    
    # Add command line arguments if provided
    if args:
        result["args"] = {k: v for k, v in vars(args).items() if v is not None and k not in ['func']}
    
    # Save result JSON
    result_path = os.path.join(base_save_dir, f'{lp}_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved results for {lp} in {base_save_dir}")
    print(f"Segment-level raw scores: {raw_save_path}")
    print(f"Segment-level sigmoid scores: {sigmoid_save_path}")
    
    if has_calibration:
        print(f"Segment-level calibration scores: {calibration_save_path}")
        print(f"System score: {system_sigmoid_score:.6f} (raw: {system_raw_score:.6f}, calibrated: {system_calibration_score:.6f})")
    else:
        print(f"System score: {system_sigmoid_score:.6f} (raw: {system_raw_score:.6f})")
    
    print(f"Detailed results: {detailed_save_path}")
    print(f"Result JSON: {result_path}") 