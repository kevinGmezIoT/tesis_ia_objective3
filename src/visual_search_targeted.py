import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from PIL import Image
from pathlib import Path

from gaitgl.data_loader import load_data
from model.model import Model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_representative_frame(data_root, seq_relative_path):
    full_dir = Path(data_root) / seq_relative_path
    imgs = sorted(list(full_dir.glob("*.png")))
    if not imgs: return None
    middle_idx = len(imgs) // 2
    return Image.open(imgs[middle_idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-ids", type=str, required=True, help="Comma-separated sequence_ids (from 'sequence_id' column)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_src, test_src, val_src = load_data(
        cfg["data_root"],
        cfg["index_csv"],
        cfg["resolution"]
    )
    
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
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_obj.m_resnet.load_state_dict(checkpoint['model_state_dict'])
    model_obj.m_resnet.eval()
    
    print("Extracting features...")
    feats, _, _, labels = model_obj.transform('test', batch_size=1)
    feats_torch = torch.from_numpy(feats)
    feats_norm = torch.nn.functional.normalize(feats_torch, p=2, dim=1)
    
    target_ids = [s.strip() for s in args.query_ids.split(",")]
    output_dir = Path("outputs") / cfg['save_name'] / "visual_search_targeted"
    output_dir.mkdir(parents=True, exist_ok=True)

    for tid in target_ids:
        idx_list = test_df.index[test_df['sequence_id'] == tid].tolist()
        if not idx_list:
            print(f"Warning: {tid} not found in test split.")
            continue
            
        q_idx = idx_list[0]
        q_feat = feats_norm[q_idx].unsqueeze(0)
        q_label = labels[q_idx]
        q_info = test_df.iloc[q_idx]
        
        dist = torch.mm(q_feat, feats_norm.t()).squeeze(0)
        sorted_indices = torch.argsort(dist, descending=True)
        
        top_indices = []
        for idx in sorted_indices:
            if idx.item() != q_idx:
                top_indices.append(idx.item())
            if len(top_indices) >= args.top_k: break
                
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(1, args.top_k + 1, 1)
        ax.imshow(get_representative_frame(cfg['data_root'], q_info['silhouette_dir']))
        ax.set_title(f"QUERY\n{tid}\nID: {q_label}")
        ax.axis('off')
        
        for i, m_idx in enumerate(top_indices):
            ax = fig.add_subplot(1, args.top_k + 1, i + 2)
            m_label = labels[m_idx]
            m_info = test_df.iloc[m_idx]
            
            is_correct = (m_label == q_label)
            color = 'green' if is_correct else 'red'
            
            ax.imshow(get_representative_frame(cfg['data_root'], m_info['silhouette_dir']))
            ax.set_title(f"Rank {i+1}\nID: {m_label}\nSim: {dist[m_idx]:.3f}", color=color)
            ax.axis('off')

        plt.suptitle(f"Targeted Search: {tid}")
        plt.tight_layout()
        plt.savefig(output_dir / f"search_{tid}.png")
        plt.close()
        print(f"Result saved for {tid}")

if __name__ == "__main__":
    main()
