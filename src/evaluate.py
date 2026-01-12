import os
import argparse
import torch
import torch.utils.data as tordata
from tqdm import tqdm
import yaml
import json
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
    # Load data using our new loader
    train_src, test_src, val_src = load_data(
        cfg["data_root"],
        cfg["index_csv"],
        cfg["resolution"]
    )
    
    # We use the Model class to initialize the architecture
    # but we'll use evaluate mode
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
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_obj.m_resnet.load_state_dict(checkpoint['model_state_dict'])
    model_obj.m_resnet.eval()
    
    # Extract features using model_obj.transform logic
    # transform already handles the mean pooling and shape adjustments
    print("Extracting features (Test split)...")
    feats, views, seq_types, labels = model_obj.transform('test', batch_size=1)
    
    # Convert labels to ints for metrics directly
    labels_idx = torch.tensor([int(l) for l in labels])
    
    # views can serve as cam_id for mAP calculation
    # We need views as integers
    unique_views = sorted(list(set(views)))
    view_to_idx = {v: i for i, v in enumerate(unique_views)}
    views_idx = [view_to_idx[v] for v in views]

    print("Calculating metrics...")
    # Gait3D protocol usually allows same-camera matching across cycles
    # calculate_map(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids, filter_same_cam)
    mAP, rank1 = calculate_map(
        torch.from_numpy(feats), labels, views_idx,
        torch.from_numpy(feats), labels, views_idx,
        filter_same_cam=False
    )
    
    ratio, inter, intra = calculate_inter_intra_ratio(torch.from_numpy(feats), labels_idx)
    
    results = {
        'mAP': float(mAP),
        'Rank-1': float(rank1),
        'ratio_inter_intra': float(ratio),
        'avg_inter_dist': float(inter),
        'avg_intra_dist': float(intra)
    }
    
    print(f"Results for {cfg['model_name']}:")
    print(json.dumps(results, indent=4))
    
    output_dir = os.path.join("outputs", cfg['save_name'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
