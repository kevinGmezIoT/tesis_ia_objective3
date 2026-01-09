import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import yaml
import pandas as pd

from gaitgl.data_loader import load_data
from model.model import Model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_src, test_src, val_src = load_data(
        cfg["data_root"],
        cfg["index_csv"],
        cfg["resolution"]
    )
    
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
    
    print("Extracting features for t-SNE...")
    feats, _, _, labels = model_obj.transform('test', batch_size=1)
    
    # Subset to max-samples
    if len(feats) > args.max_samples:
        indices = np.random.choice(len(feats), args.max_samples, replace=False)
        feats = feats[indices]
        labels = [labels[i] for i in indices]
        
    print(f"Running t-SNE on {len(feats)} samples...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    feats_2d = tsne.fit_transform(feats)
    
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(labels)))
    # Mostramos los primeros 20 IDs con colores, el resto en gris
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == label]
        if i < 20:
            plt.scatter(feats_2d[idx, 0], feats_2d[idx, 1], label=f"ID {label}", alpha=0.7)
        else:
            plt.scatter(feats_2d[idx, 0], feats_2d[idx, 1], c='gray', alpha=0.1, label='_nolegend_')

    plt.title(f"t-SNE Embeddings - {cfg['save_name']}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    
    output_dir = Path("outputs") / cfg['save_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "tsne_visualization.png")
    print(f"Visualization saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    main()
