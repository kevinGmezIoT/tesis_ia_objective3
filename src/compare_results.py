import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_comparison(experiment_dirs, output_path="outputs/comparison"):
    os.makedirs(output_path, exist_ok=True)
    
    data = {}
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        history_path = exp_path / "history.json"
        eval_path = exp_path / "eval_results.json"
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_res = json.load(f)
            else:
                eval_res = None
                
            data[exp_path.name] = {
                'history': history,
                'eval': eval_res
            }

    if not data:
        print("No valid experiment directories found with history.json.")
        return

    # 1. Plot Training History Comparison
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    for name, d in data.items():
        if 'train_loss' in d['history']:
            plt.plot(d['history']['train_loss'], label=f'{name} (Train)')
        if 'val_loss' in d['history']:
            plt.plot(d['history']['val_loss'], label=f'{name} (Val)', linestyle='--')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    for name, d in data.items():
        if 'train_acc' in d['history']:
            plt.plot(d['history']['train_acc'], label=f'{name} (Train)')
        if 'val_acc' in d['history']:
            plt.plot(d['history']['val_acc'], label=f'{name} (Val)', linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "history_comparison.png"))
    print(f"History comparison saved to {output_path}/history_comparison.png")
    
    # 2. Plot Final Evaluation Bar Chart
    eval_metrics = ['mAP', 'ratio_inter_intra']
    available_metrics = []
    
    # Check which metrics are available across all experiments
    for name, d in data.items():
        if d['eval']:
            for m in eval_metrics:
                if m in d['eval'] and m not in available_metrics:
                    available_metrics.append(m)
    
    if available_metrics:
        experiments = list(data.keys())
        x = np.arange(len(available_metrics))
        width = 0.8 / len(experiments)
        
        plt.figure(figsize=(10, 6))
        for i, name in enumerate(experiments):
            if data[name]['eval']:
                values = [data[name]['eval'].get(m, 0) for m in available_metrics]
                plt.bar(x + (i - len(experiments)/2 + 0.5) * width, values, width, label=name)
        
        plt.ylabel('Score')
        plt.title('Final Evaluation Metrics Comparison')
        plt.xticks(x, available_metrics)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_path, "metrics_comparison.png"))
        print(f"Metrics comparison saved to {output_path}/metrics_comparison.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, help="Comma-separated list of experiment directories in 'outputs/'")
    args = parser.parse_args()

    if args.dirs:
        dirs = [os.path.join("outputs", d.strip()) for d in args.dirs.split(",")]
    else:
        # Default: compare all subdirectories in outputs/
        outputs_path = Path("outputs")
        if outputs_path.exists():
            dirs = [str(d) for d in outputs_path.iterdir() if d.is_dir() and (d / "history.json").exists()]
        else:
            dirs = []

    if not dirs:
        print("No experiments found in 'outputs/'. Use --dirs to specify.")
        return

    print(f"Comparing experiments: {dirs}")
    plot_comparison(dirs)

if __name__ == "__main__":
    main()
