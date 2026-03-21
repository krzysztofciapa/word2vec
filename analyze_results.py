import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
LOGS_DIR = "logs"

def parse_training_logs(log_path):
    batches = []
    losses = []
    
    if not os.path.exists(log_path):
        return batches, losses
        
    current_offset = 0
    last_batch = -1
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Batch\s+(\d+)\s+\|\s+Loss:\s+([\d\.]+)', line)
            if match:
                batch_num = int(match.group(1))
                loss_val = float(match.group(2))
                
                if batch_num < last_batch:
                    current_offset += last_batch
                
                batches.append(batch_num + current_offset)
                losses.append(loss_val)
                last_batch = batch_num
                
    return batches, losses

def plot_training_curves():
    plt.figure(figsize=(12, 6))
    
    log_files = [f for f in os.listdir(LOGS_DIR) if f.startswith("train_") and f.endswith(".log")]
    if not log_files:
        print("No training logs found.")
        return

    for log_file in log_files:
        opt = log_file.replace("train_", "").replace(".log", "")
        batches, losses = parse_training_logs(os.path.join(LOGS_DIR, log_file))
        
        if batches:
            window = max(1, len(losses) // 100)
            if window > 1:
                smoothed_losses = [sum(losses[i:i+window])/window for i in range(len(losses)-window+1)]
                smoothed_batches = batches[window-1:]
            else:
                smoothed_losses = losses
                smoothed_batches = batches
            
            plt.plot(smoothed_batches, smoothed_losses, label=opt.upper(), linewidth=2, alpha=0.8)
            
    plt.title('Training Convergence Comparison', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Cumulative Training Batches', fontsize=12)
    plt.ylabel('Smoothed Training Loss', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(RESULTS_DIR, 'training_loss_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def plot_benchmarks():
    eval_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("eval_") and f.endswith(".json")]
    if not eval_files:
        print("No evaluation results found.")
        return

    results = []
    for f in eval_files:
        model_name = f.replace("eval_", "").replace(".json", "").upper()
        with open(os.path.join(RESULTS_DIR, f), 'r') as jf:
            data = json.load(jf)
            results.append({
                'name': model_name,
                'analogy': data.get('analogy_accuracy', 0.0) * 100.0,
                'wordsim': data.get('wordsim_spearman', 0.0)
            })

    # Sort by WordSim score
    results = sorted(results, key=lambda x: x['wordsim'], reverse=True)
    
    names = [r['name'] for r in results]
    analogies = [r['analogy'] for r in results]
    wordsims = [r['wordsim'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Analogy Bars (Left Axis)
    bars1 = ax1.bar(x - width/2, analogies, width, color='#3498db', alpha=0.8, label='Analogy Accuracy (%)', edgecolor='black')
    ax1.set_ylabel('Analogy Accuracy (%)', color='#2980b9', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_ylim(0, max(max(analogies)*1.2, 5))
    
    # WordSim Bars (Right Axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, wordsims, width, color='#e74c3c', alpha=0.8, label='WordSim Correlation', edgecolor='black')
    ax2.set_ylabel('Spearman Correlation (ρ)', color='#c0392b', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 1.0)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=11, fontweight='bold')
    
    plt.title('Comprehensive Word2Vec Benchmark Comparison', fontsize=18, fontweight='bold', pad=25)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=True)
    
    def autolabel(bars, ax, fmt='{:.1f}%', color='black', offset=0.01):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    fmt.format(height),
                    ha='center', va='bottom', color=color, fontweight='bold', fontsize=10)

    autolabel(bars1, ax1, '{:.1f}%', '#2980b9', offset=0.1)
    autolabel(bars2, ax2, '{:.3f}', '#c0392b', offset=0.01)
            
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    
    out_path = os.path.join(RESULTS_DIR, 'benchmarks.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating comprehensive analysis charts...")
    plot_training_curves()
    plot_benchmarks()
    print("\nAnalysis complete! Results saved in the results/ directory.")
