"""Command line interface for Remedy MT Score"""
import argparse
import sys
import os
from typing import Optional

# Import needed modules
from remedy.toolbox.score import (
    initialize_model, 
    load_translation_data,
    process_data_for_scoring,
    prepare_dataset,
    run_inference,
    calculate_scores,
    save_score_results,
    AutoTokenizer
)
from remedy.toolbox.languages import is_supported_language, get_supported_languages, LANG_MAP

def extract_model_name(model_path):
    """Extract model name from path for use as metric name."""
    # Get the base directory name
    base_name = os.path.basename(os.path.normpath(model_path))
    
    # For HF model IDs like "ShaomuTan/ReMedy-9B-22", extract just the model name
    if '/' in base_name:
        return base_name.split('/')[-1]
    
    return base_name

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Remedy Machine Translation Scoring Tool")
    
    # Required arguments
    parser.add_argument('--src_file', type=str, required=True,
                      help='Path to source file')
    parser.add_argument('--mt_file', type=str, required=True,
                      help='Path to MT file')
    parser.add_argument('--src_lang', type=str, required=True,
                      help='Source language code (ISO 639-1 or 639-3)')
    parser.add_argument('--tgt_lang', type=str, required=True,
                      help='Target language code (ISO 639-1 or 639-3)')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                      help='HuggingFace model ID or path to local model directory')
    parser.add_argument('--cache_dir', type=str,
                      help='Directory where models are cached (for HuggingFace model IDs)')
    
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save the results')
    
    # Optional arguments
    parser.add_argument('--ref_file', type=str,
                      help='Path to reference file (optional)')
    parser.add_argument('--no_ref', action='store_true',
                      help='No Reference (QE) Mode')
    parser.add_argument('--calibrate', action='store_true',
                      help='Apply entropy-based calibration to scores')
    parser.add_argument('--num_gpus', type=int, default=1,
                      help='Number of GPUs to use')
    parser.add_argument('--num_seqs', type=int, default=256,
                      help='Number of sequences')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum sequence length')
    parser.add_argument('--enable_truncate', action='store_true',
                      help='Enable sequence truncation')
    parser.add_argument('--version', action='store_true',
                      help='Show version information')
    parser.add_argument('--list_languages', action='store_true',
                      help='List all supported language codes')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_languages:
        print("Supported language codes:")
        languages = get_supported_languages()
        for i, lang in enumerate(languages):
            print(f"{lang}: {LANG_MAP[lang]}", end="\t")
            if (i + 1) % 5 == 0:
                print()
        print()
        sys.exit(0)
    
    return args

def main():
    """Main entry point for the remedy-score command."""
    args = parse_arguments()
    
    if args.version:
        from remedy import __version__
        print(f"Remedy MT Score v{__version__}")
        return 0
    
    # Validate language codes
    if not is_supported_language(args.src_lang):
        print(f"Error: Source language '{args.src_lang}' is not supported.")
        print("Use --list_languages to see all supported language codes.")
        return 1
        
    if not is_supported_language(args.tgt_lang):
        print(f"Error: Target language '{args.tgt_lang}' is not supported.")
        print("Use --list_languages to see all supported language codes.")
        return 1
    
    # Extract metric name from model path
    metric_name = extract_model_name(args.model)
    print(f"Using metric name: {metric_name}")
    
    # Validate arguments
    if args.no_ref and not args.ref_file:
        print("QE mode enabled: no reference file will be used")
    elif not args.no_ref and not args.ref_file:
        raise ValueError("Reference file is required when QE mode is not enabled")
    
    QE = args.no_ref
    
    print("Running translation quality scoring")
    print(f"Source file: {args.src_file}")
    print(f"MT file: {args.mt_file}")
    print(f"Reference file: {args.ref_file}")
    print(f"Source language: {args.src_lang}")
    print(f"Target language: {args.tgt_lang}")
    print(f"Model: {args.model}")
    if args.cache_dir:
        print(f"Cache directory: {args.cache_dir}")
    print(f"Metric name: {metric_name}")
    print(f"QE mode: {QE}")
    print(f"Calibration: {args.calibrate}")
    
    # 1. Prepare Model
    print("Initializing model...")
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        print(f"Set cache directory to: {args.cache_dir}")
    
    llm = initialize_model(args.model, args.max_length, args.enable_truncate, args.num_gpus, args.num_seqs, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    # 2. Load translation data
    print("Loading translation data...")
    df_data = load_translation_data(
        args.src_file, args.mt_file, args.ref_file, args.src_lang, args.tgt_lang
    )
    print(f"Loaded {len(df_data)} sentences")
    
    # 3. Process data
    print("Processing data...")
    ds_data, df_data = process_data_for_scoring(df_data, QE=QE)
    ds_data = prepare_dataset(ds_data, tokenizer, args.max_length, args.enable_truncate)
    
    # 4. Run inference
    print("Running inference...")
    embeddings = run_inference(llm, ds_data)
    
    # Apply calibration if requested
    if args.calibrate:
        print("Applying entropy-based calibration...")
        df_with_scores = calculate_scores(embeddings, df_data, calibrate=True)
    else:
        df_with_scores = calculate_scores(embeddings, df_data)
    
    # 5. Save results
    print("Saving scores...")
    save_score_results(
        df_with_scores, 
        args.save_dir, 
        metric_name, 
        args.src_lang, 
        args.tgt_lang,
        args
    )
    
    print("Scoring completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())