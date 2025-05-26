#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
from collections import OrderedDict

def extract_metrics(result_dir,system_name):
    # Systems to extract metrics for
    target_systems = [
        "GEMBA-MQM[noref]",
        "XCOMET-Ensemble",
        "MetricX-23",
        system_name
    ]
    
    # Get all log files in the directory with their original order
    log_files = []
    for file in sorted(os.listdir(result_dir)):
        if re.match(r'[a-z]+-[a-z]+_results\.log', file):
            log_files.append(file)
    
    # Keep original order exactly as found - use an OrderedDict for results
    # Store the raw file order first
    file_order = [f.split('_')[0] for f in log_files]
    
    # Initialize results dictionary with all target systems
    results = OrderedDict()
    for lang_pair in file_order:
        results[lang_pair] = {}
    
    # Process each log file
    for log_file in log_files:
        # Extract language pair from filename (e.g., 'en-de' from 'en-de_results.log')
        lang_pair = log_file.split('_')[0]
        
        # Read and process the file
        file_path = os.path.join(result_dir, log_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip the first line (header)
            for line in lines[1:]:
                # Check if line contains system information
                line_parts = line.strip().split()
                if not line_parts or len(line_parts) < 3:
                    continue
                
                # Identify the rank position (which will be all digits)
                rank_pos = None
                for i, part in enumerate(line_parts):
                    if part.isdigit():
                        rank_pos = i
                        break
                
                if rank_pos is None or rank_pos == 0:
                    continue
                
                # Extract system name (everything before the rank)
                system_name = ' '.join(line_parts[:rank_pos]).strip()
                
                # Extract accuracy (should be right after the rank)
                if rank_pos + 1 < len(line_parts):
                    try:
                        acc = float(line_parts[rank_pos + 1])
                        # Check if this is one of our target systems
                        if system_name in target_systems:
                            results[lang_pair][system_name] = acc
                    except ValueError:
                        # Skip if we can't convert to float
                        continue
    
    return results, file_order

def extract_sys_metrics(result_file_path,system_name):
    # Check if file exists
    if not os.path.exists(result_file_path):
        print(f"Warning: System result file {result_file_path} not found")
        return {}
    
    # Target systems to extract metrics for
    target_systems = [
        "GEMBA-MQM[noref]",
        "XCOMET-Ensemble",
        "MetricX-23",
        system_name
    ]
    
    # Initialize results dictionary
    sys_results = {}
    
    # Read and process the file
    with open(result_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # Skip the first line (header)
        for line in lines[1:]:
            # Split line into parts
            line_parts = line.strip().split()
            if not line_parts or len(line_parts) < 3:
                continue
            
            # Find the rank position
            rank_pos = None
            for i, part in enumerate(line_parts):
                if part.isdigit():
                    rank_pos = i
                    break
            
            if rank_pos is None or rank_pos == 0:
                continue
            
            # Extract system name
            system_name = ' '.join(line_parts[:rank_pos]).strip()
            
            # Extract accuracy
            if rank_pos + 1 < len(line_parts):
                try:
                    acc = float(line_parts[rank_pos + 1])
                    if system_name in target_systems:
                        sys_results[system_name] = acc
                except ValueError:
                    continue
    
    return sys_results

def create_dataframe(seg_results, file_order, system_name, sys_results=None):
    # Target systems
    target_systems = [
        "GEMBA-MQM[noref]",
        "XCOMET-Ensemble",
        "MetricX-23",
        system_name
    ]
    
    # Create a DataFrame
    df_data = []
    for system in target_systems:
        # Start with system name
        row = [system]
        
        # Get segment-level values for calculating average
        seg_system_values = []
        for lang_pair in file_order:
            if lang_pair in seg_results and system in seg_results[lang_pair]:
                value = seg_results[lang_pair][system]
                seg_system_values.append(value)
        
        # Calculate segment-level average
        if seg_system_values:
            seg_avg = sum(seg_system_values) / len(seg_system_values)
        else:
            seg_avg = float('nan')
        
        # Get system-level result
        sys_value = float('nan')
        if sys_results and system in sys_results:
            sys_value = sys_results[system]
        
        # Calculate overall average (avg)
        values_for_avg = [v for v in [sys_value, seg_avg] if not pd.isna(v)]
        if values_for_avg:
            avg_value = sum(values_for_avg) / len(values_for_avg)
        else:
            avg_value = float('nan')
        
        # Add avg, sys, seg-avg to the row
        row.append(avg_value)  # avg
        row.append(sys_value)  # sys
        row.append(seg_avg)    # seg-avg
        
        # Add individual language pair values
        for lang_pair in file_order:
            if lang_pair in seg_results and system in seg_results[lang_pair]:
                value = seg_results[lang_pair][system]
                row.append(value)
            else:
                row.append(float('nan'))
        
        df_data.append(row)
    
    # Create DataFrame with ordered columns
    columns = ['System', 'avg', 'sys', 'seg-avg'] + file_order
    df = pd.DataFrame(df_data, columns=columns)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Extract metric values from result logs.')
    parser.add_argument('--result_dir', required=True, help='Directory containing result log files')
    parser.add_argument('--mqm_folder', default='wmt23-mqm/sigmoid', help='MQM folder name (default: wmt23-mqm)')
    parser.add_argument('--da_folder', default='wmt23-da/sigmoid', help='DA folder name (default: wmt23-da)')
    parser.add_argument('--system_name', default='REMET-9b-sigmoid', help='')
    args = parser.parse_args()
    

    # Extract segment-level metrics from files
    mqm_seg_file = os.path.join(args.result_dir, args.mqm_folder, 'seg')
    da_seg_file = os.path.join(args.result_dir, args.da_folder, 'seg')

    mqm_results, mqm_file_order = extract_metrics(mqm_seg_file, args.system_name)
    da_results, da_file_order = extract_metrics(da_seg_file, args.system_name)
    
    # Extract system-level metrics from files
    mqm_sys_file = os.path.join(args.result_dir, args.mqm_folder, 'sys_results.log')
    da_sys_file = os.path.join(args.result_dir, args.da_folder, 'sys_results.log')
    
    mqm_sys_results = extract_sys_metrics(mqm_sys_file, args.system_name)
    da_sys_results = extract_sys_metrics(da_sys_file, args.system_name)
    
    # Create pandas DataFrame with columns in requested order
    df_mqm = create_dataframe(mqm_results, mqm_file_order, args.system_name, mqm_sys_results)
    df_da = create_dataframe(da_results, da_file_order, args.system_name, da_sys_results)
    
    df_mqm.set_index('System', inplace=True)
    df_da.set_index('System', inplace=True)
    
    df_mqm_renamed = df_mqm.rename(columns={
        col: f"mqm-{col}" for col in df_mqm.columns
    })
    df_da_renamed = df_da.rename(columns={
        col: f"da-{col}" for col in df_da.columns
    })
    
    combined_df = pd.concat([df_mqm_renamed, df_da_renamed], axis=1)
    combined_df['avg'] = combined_df[['mqm-avg', 'da-avg']].mean(axis=1)
    
    cols = combined_df.columns.tolist()
    cols.remove('avg')
    combined_df = combined_df[['avg'] + cols]
    
    # Format the output (4 decimal places)
    numeric_columns = df_mqm.select_dtypes(include=['float64', 'int64']).columns
    df_mqm[numeric_columns] = df_mqm[numeric_columns] * 100
    numeric_columns = df_da.select_dtypes(include=['float64', 'int64']).columns
    df_da[numeric_columns] = df_da[numeric_columns] * 100
    
    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    combined_df[numeric_columns] = combined_df[numeric_columns] * 100
    
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print('\nMQM Result:')
    print(df_mqm)
    print('\nDA Result:')
    print(df_da)
    

if __name__ == '__main__':
    main()