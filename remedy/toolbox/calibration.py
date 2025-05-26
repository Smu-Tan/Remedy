import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats, signal
from sklearn.mixture import GaussianMixture

def sigmoid(x, temperature=1.0):
    """
    Apply sigmoid transformation to raw scores with temperature scaling.
    
    Args:
        x: Raw score or array of scores
        temperature: Temperature parameter for scaling. Higher values produce smoother transitions.
    
    Returns:
        Transformed score(s) in range [0, 1]
    """
    x = np.array(x, dtype=float)
    return 1 / (1 + np.exp(-x / temperature))


def final_optimized_entropy(scores, force_normal_dist=False, force_high_reward_dist=False):
    scores = np.array(scores, dtype=float)
    
    # 1. basic analysis
    q10, q25, q50, q75, q90 = np.percentile(scores, [10, 25, 50, 75, 90])
    iqr = q75 - q25
    min_score = np.min(scores)
    max_score = np.max(scores)
    total_range = max_score - min_score
    mean_score = np.mean(scores)
    
    # 2. get features
    # most high rewards
    high_score_ratio = np.mean(scores > 4.0)
    very_high_score_ratio = np.mean(scores > 5.0)
    
    # even distribution
    negative_ratio = np.mean(scores < 0)
    has_significant_negative = negative_ratio > 0.05 and min_score < -1.0
    has_significant_positive = np.mean(scores > 1.0) > 0.3
    
    # 3. kde analysis
    kde = stats.gaussian_kde(scores, bw_method='silverman')
    x_grid = np.linspace(min_score-0.5, max_score+0.5, 1000)
    density = kde(x_grid)
    
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(density, height=0.1*max(density), distance=len(x_grid)//20)
    
    num_peaks = len(peaks)
    is_bimodal = num_peaks >= 2
    
    # 4. most rewards cluster in high scores
    is_high_reward_dist_type = high_score_ratio > 0.7 and mean_score > 4.0 and very_high_score_ratio > 0.3
    
    # 5. bimodal reward distribution
    is_normal_dist_type = is_bimodal and has_significant_negative and has_significant_positive
    

    print(f"Distribution Analysis:")
    print(f"  Reward Range: {min_score:.2f} to {max_score:.2f}, mean: {mean_score:.2f}")
    print(f"  high score (>4) ratio: {high_score_ratio:.2f}, e-high score (>5) ratio: {very_high_score_ratio:.2f}")
    print(f"  negative score ratio: {negative_ratio:.2f}")
    print(f"  how many peaks: {num_peaks}")
    print(f"  High rewards cluster type: {is_high_reward_dist_type}")
    print(f"  Even distribution type: {is_normal_dist_type}")
    

    # 7. strategy to select temperature
    if is_high_reward_dist_type:
        print("High rewards cluster type，use very high temperature range.")
        temp_range = np.linspace(1.4, 1.9, 6) 
        temp_default = 1.7
    elif is_normal_dist_type:
        print("Even distribution type，use very low temperature range.")
        temp_range = np.linspace(0.15, 0.35, 5)  
        temp_default = 0.25
    elif high_score_ratio > 0.5:
        print("Main distribution is high scores，use relatively high temperature range.")
        temp_range = np.linspace(0.8, 1.5, 8)
        temp_default = 1.2
    elif is_bimodal:
        print("bimodal distribution type，use relatively low temperature range.")
        temp_range = np.linspace(0.3, 0.7, 8)
        temp_default = 0.5
    else:
        print("This is normal Distribution, search with default range.")
        temp_range = np.linspace(0.4, 1.2, 8)
        temp_default = 0.7

    # 8. Entropy-guided Temperature calibration
    max_entropy = -float('inf')
    best_temp = temp_default

    for temp in temp_range:
        transformed = sigmoid(scores, temp)
        hist, _ = np.histogram(transformed, bins=20, range=(0, 1), density=True)
        p = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(p * np.log2(p + 1e-10))

        if entropy > max_entropy:
            max_entropy = entropy
            best_temp = temp

    print(f"Finally selected temperature: {best_temp:.2f}")
    return best_temp



def collect_all_scores(raw_score_dir, save_plot=False, plot_dir=None):
    all_scores = []
    lp_scores = defaultdict(list)
    
    for root, dirs, files in os.walk(raw_score_dir):
        rel_path = os.path.relpath(root, raw_score_dir)
        lp = rel_path.split(os.path.sep)[0] if rel_path != '.' else 'global'
        
        for file in files:
            if file.endswith('.seg.score'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split('\t')
                            if len(parts) == 1:
                                parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    score = float(parts[1])
                                    all_scores.append(score)
                                    lp_scores[lp].append(score)
                                except ValueError:
                                    continue
                except Exception as e:
                    print(f"Failed when reading: {file_path} {e}")
    
    if save_plot and all_scores:
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            
        plt.figure(figsize=(12, 6))
        plt.hist(all_scores, bins=50, alpha=0.7)
        plt.title('Distribution of rewards')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, 'all_scores_distribution.png'), dpi=300)
        plt.close()
        
        for lp, scores in lp_scores.items():
            if len(scores) > 10:
                plt.figure(figsize=(10, 5))
                plt.hist(scores, bins=30, alpha=0.7)
                plt.title(f'Distribution for Langauge Pair: {lp}')
                plt.xlabel('Reward')
                plt.ylabel('Frequency')
                plt.grid(True, linestyle='--', alpha=0.5)
                
                if plot_dir:
                    plt.savefig(os.path.join(plot_dir, f'{lp}_scores_distribution.png'), dpi=300)
                plt.close()
    
    return all_scores, lp_scores

def visualize_temperatures(scores, temps, save_path=None):
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.hist(scores, bins=30, alpha=0.7, color='gray')
    plt.title('Original Distribution', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(2, 1, 2)
    score_min, score_max = np.min(scores), np.max(scores)
    x = np.linspace(min(score_min - 1, -5), max(score_max + 1, 5), 1000)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    for i, (method, temp) in enumerate(temps.items()):
        y = sigmoid(x, temp)
        color_idx = i % len(colors)
        plt.plot(x, y, label=f'{method}: temp={temp:.4f}', color=colors[color_idx], linewidth=2)
    
    plt.axvspan(score_min, score_max, alpha=0.2, color='gray')
    
    plt.title('Compare Sigmoid temperature', fontsize=14)
    plt.xlabel('Original Reward', fontsize=12)
    plt.ylabel('Normalized Reward with Sigmoid', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
    
    for method, temp in temps.items():
        print(f"{method.ljust(15)}: {temp:.4f}")
        
    return temps

def process_score_file(input_file, output_file, temperature):
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        systems = {}
        
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) == 1:
                    parts = line.split()
                
                if len(parts) >= 2:
                    system_name = parts[0]
                    try:
                        score = float(parts[1])
                        transformed_score = sigmoid(score, temperature)
                        fout.write(f"{system_name}\t{transformed_score:.16f}\n")
                        
                        if system_name not in systems:
                            systems[system_name] = []
                        systems[system_name].append(transformed_score)
                    except ValueError:
                        fout.write(line + '\n')
                else:
                    fout.write(line + '\n')
        
        sys_output_file = output_file.replace('.seg.score', '.sys.score')
        with open(sys_output_file, 'w') as f:
            for system, scores in systems.items():
                system_score = np.mean(scores)        
                f.write(f"{system}\t{system_score:.16f}\n")
        
        return True
    except Exception as e:
        print(f"Failed when processing {input_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Entropy-guided calibration with sigmoid.')
    parser.add_argument('--raw_score_dir', required=True, help='Raw reward folder')
    parser.add_argument('--output_score_dir', required=True, help='Output reward dir')
    parser.add_argument('--per_lp_temp', action='store_true', help='Calculate temperature for each language pair.')
    parser.add_argument('--plot_dir', type=str, default=None, help='folder to save plots')
    
    args = parser.parse_args()
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    all_scores, lp_scores = collect_all_scores(args.raw_score_dir, save_plot=(args.plot_dir is not None), plot_dir=args.plot_dir)

    # for all scores.
    print('finding optimal temperature for all scores...')
    global_best_temp = final_optimized_entropy(all_scores)
    print(f'global_best_temp={global_best_temp}')

    # for each language pair
    lp_temps = {}
    if args.per_lp_temp:
        for lp, scores in lp_scores.items():
            if len(scores) >= 10: 
                lp_temps[lp] = final_optimized_entropy(scores)        
                print(f"Temperature for Language Pair {lp}: {lp_temps[lp]:.4f}")
                
                # Visualize
                if args.plot_dir:
                    temps_vis= {'temperature: ':lp_temps[lp]}
                    visualize_temperatures(
                        scores, 
                        temps_vis,
                        save_path=os.path.join(args.plot_dir, f'{lp}_temperature.png')
                    )
            else:
                # when no enough data points, then use global_best_temp
                lp_temps[lp] = global_best_temp
    
    for root, dirs, files in os.walk(args.raw_score_dir):
        rel_path = os.path.relpath(root, args.raw_score_dir)
        current_temp = global_best_temp
        if args.per_lp_temp:
            lp = rel_path.split(os.path.sep)[0] if rel_path != '.' else 'global'
            current_temp = lp_temps.get(lp, global_best_temp)
        
        for file in files:
            if file.endswith('.seg.score'):
                input_file = os.path.join(root, file)
                if rel_path == '.':
                    output_file = os.path.join(args.output_score_dir, file)
                else:
                    output_file = os.path.join(args.output_score_dir, rel_path, file)
                
                _ = process_score_file(input_file, output_file, current_temp)

if __name__ == "__main__":
    main()