import os
import argparse
import torch
import torch.utils.data as tordata
from tqdm import tqdm
import yaml
import json
import pandas as pd
import numpy as np

from gaitgl.data_loader import load_data
from model.model import Model
from utils.metrics import calculate_map, calculate_inter_intra_ratio

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_src, test_src, val_src = load_data(
        cfg["data_root"],
        cfg["index_csv"],
        cfg["resolution"]
    )
    
    # Load labels and metadata directly from CSV for categorization
    df = pd.read_csv(cfg["index_csv"])
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    model_obj = Model(
        hidden_dim=cfg["hidden_dim"],
        lr=cfg["lr"],
        hard_or_full_trip=cfg["hard_or_full_trip"],
        margin=cfg["margin"],
        num_workers=cfg["num_workers"],
        batch_size=(cfg["batch_P"], cfg["batch_K"]),
        restore_iter=0,
        total_iter=0,
        save_name=cfg["save_name"],
        train_pid_num=len(train_src.label_set) if train_src else 0,
        frame_num=cfg["frame_num"],
        model_name=cfg["model_name"],
        train_source=train_src,
        test_source=test_src,
        img_size=cfg["resolution"]
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_obj.m_resnet.load_state_dict(checkpoint['model_state_dict'])
    model_obj.m_resnet.eval()
    
    print("Extracting features...")
    feats, views, seq_types, labels = model_obj.transform('test', batch_size=1)
    feats_torch = torch.from_numpy(feats)
    
    # Map views and labels to indices for metrics
    unique_views = sorted(list(set(views)))
    view_to_idx = {v: i for i, v in enumerate(unique_views)}
    views_idx = [view_to_idx[v] for v in views]
    
    unique_pids = sorted(list(set(labels)))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    labels_idx = torch.tensor([pid_to_idx[l] for l in labels])

    def evaluate_subset(mask, name):
        if not np.any(mask):
            return None
        
        q_feats = feats_torch[mask]
        q_labels = [labels[i] for i, m in enumerate(mask) if m]
        q_views = [views_idx[i] for i, m in enumerate(mask) if m]
        
        # Mapping for intra-set metrics
        sub_unique_pids = sorted(list(set(q_labels)))
        sub_pid_to_idx = {pid: i for i, pid in enumerate(sub_unique_pids)}
        sub_labels_idx = torch.tensor([sub_pid_to_idx[l] for l in q_labels])
        
        mAP = calculate_map(q_feats, q_labels, q_views, feats_torch, labels, views_idx)
        ratio, inter, intra = calculate_inter_intra_ratio(q_feats, sub_labels_idx)
        
        return {
            'mAP': mAP,
            'ratio_inter_intra': ratio,
            'avg_inter_dist': inter,
            'avg_intra_dist': intra
        }

    results = {'OVERALL': evaluate_subset(np.ones(len(labels), dtype=bool), 'Overall')}
    
    # Categories to check
    categories = ['scene', 'illum_label', 'occlusion_label']
    for cat in categories:
        if cat in test_df.columns:
            print(f"Evaluating by {cat}...")
            results[cat] = {}
            unique_vals = test_df[cat].unique()
            for val in unique_vals:
                mask = (test_df[cat] == val).values
                res = evaluate_subset(mask, f"{cat}_{val}")
                if res:
                    results[cat][str(val)] = res
        else:
            print(f"Category '{cat}' not found in CSV.")

    print("\nCategorized Results Summary:")
    print(json.dumps(results, indent=4))
    
    output_dir = os.path.join("outputs", cfg['save_name'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_categorized.json"), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
