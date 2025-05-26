import collections
import dataclasses
import json
import os

from mt_metrics_eval import data
from mt_metrics_eval import tasks
import numpy as np
import transformers

@dataclasses.dataclass
class Arguments:
  en_de: str = dataclasses.field(metadata={"help": "The en-de input file."})
  en_es: str = dataclasses.field(metadata={"help": "The en-es input file."})
  ja_zh: str = dataclasses.field(metadata={"help": "The ja-zh input file."})

  output_file: str = dataclasses.field(
      metadata={"help": "The output file with evaluation metrics."},
  )


def load_score_from_text(
    input_file: str, num_segments: int = None
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Load system-level and segment-level scores from a text file.
    
    Args:
        input_file: Path to the text file containing scores.
        num_segments: Optional number of segments. If None, inferred from data.
        
    Returns:
        A tuple containing (seg_scores, sys_scores)
    """
    # Collect scores by system ID
    raw_scores = collections.defaultdict(list)
    
    # Read the file and collect scores
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue  # Skip improperly formatted lines
                
            system_id = parts[0]
            try:
                # Note: If scores need to be reversed, add a negative sign
                score = float(parts[1])  # or use -1 * float(parts[1])
            except ValueError:
                continue  # Skip invalid scores
                
            # Add the score to the system's list
            raw_scores[system_id].append(score)
    
    # Determine segment count (if not provided)
    if num_segments is None:
        num_segments = max(len(raw_scores[system_id]) for system_id in raw_scores)
    
    # Create segment-level scores dictionary
    seg_scores = {}
    for system_id in raw_scores:
        seg_scores[system_id] = []
        for segment_idx in range(num_segments):
            if segment_idx < len(raw_scores[system_id]):
                seg_scores[system_id].append(raw_scores[system_id][segment_idx])
            else:
                seg_scores[system_id].append(None)  # Missing segments
    
    # Calculate system-level scores (average of segment scores)
    sys_scores = {}
    for system_id in seg_scores:
        cur_scores = np.asarray([s for s in seg_scores[system_id] if s is not None])
        if len(cur_scores) > 0:
            sys_scores[system_id] = np.mean(cur_scores)
        else:
            sys_scores[system_id] = None
    
    return seg_scores, sys_scores


import sys

# Get command line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <metric_name> <result_dir>")
    sys.exit(1)

metric_name = sys.argv[1]  # First command line argument: metric_name
result_dir = sys.argv[2]   # Second command line argument: result_dir
wmt24_lps = ["en-de", 
             "en-es", 
             "ja-zh"
             ]
evs_dict = {
    ("wmt24", lp): data.EvalSet("wmt24", lp, True) for lp in wmt24_lps
}
segment_counts_per_lp = {}
for lp in wmt24_lps:
  evs = evs_dict[("wmt24", lp)]
  gold_scores = evs.Scores("seg", "mqm")
  for _, scores in gold_scores.items():
    segment_counts_per_lp[lp] = len(scores)
    break

scores = {
      "en-de": load_score_from_text(
        f'{result_dir}/en-de/{metric_name}-refB.seg.score', segment_counts_per_lp["en-de"]),
      "en-es": load_score_from_text(
        f'{result_dir}/en-es/{metric_name}-refA.seg.score', segment_counts_per_lp["en-es"]),
      "ja-zh": load_score_from_text(
        f'{result_dir}/ja-zh/{metric_name}-refA.seg.score', segment_counts_per_lp["ja-zh"]),
  }

for lp in wmt24_lps:
    evs = evs_dict[("wmt24", lp)]
    seg_scores, sys_scores = scores[lp]
    evs._scores["seg"][f"{metric_name}-{evs.std_ref}"] = seg_scores  # pylint: disable=protected-access
    evs._scores["sys"][f"{metric_name}-{evs.std_ref}"] = sys_scores  # pylint: disable=protected-access
    evs._metric_names.add(f"{metric_name}-{evs.std_ref}")  # pylint: disable=protected-access
    evs._metric_basenames.add(metric_name)  # pylint: disable=protected-access

for evs in evs_dict.values():
    evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})

wmt24_tasks, wts = tasks.WMT24(wmt24_lps, k=0)
results = wmt24_tasks.Run(eval_set_dict=evs_dict)

# Calculate average correlations
avg_corrs = results.AverageCorrs(wts)

# Convert to format suitable for initial_column parameter
# Need to convert to (correlation, rank) tuples
initial_col = {}
for rank, (metric, corr) in enumerate(avg_corrs.items()):
    initial_col[metric] = (corr, rank + 1)

# Generate table with average correlations
table = results.Table(
    initial_column=initial_col,
    initial_column_header="Avg Corr",
    attr_list=["lang", "level", "corr_fcn"],  # Add these attributes as header rows
    nicknames={"en-de": "en-de", "en-es": "en-es", "ja-zh": "ja-zh", 
               "sys": "Sys", "seg": "Seg", 
               "pearson": "r", "KendallWithTiesOpt": "acc_eq", "pce": "SPA"}  # Optional abbreviations
)
print(table)

# Example command to run:
# python script.py Remedy-9b /path/to/results