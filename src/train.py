import argparse
import yaml
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
from pathlib import Path

from gaitgl.data_loader import load_data
from model.model import Model
from model.utils import TripletSampler
import torch.utils.data as tordata

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model_obj, train_loader, epoch, total_epochs):
    model_obj.m_resnet.train()
    model_obj.sample_type = 'random'
    
    epoch_metrics = {
        'loss': [],
        't_loss': [],
        'id_loss': [],
        'acc': []
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    for seq, view, seq_type, label, batch_frame in pbar:
        model_obj.restore_iter += 1
        model_obj.optimizer.zero_grad()
        
        # Consistent mapping with model_obj
        targets = torch.tensor([model_obj.label_map[li] for li in label]).long().cuda()
        
        # Sequences
        seq = np.float32(np.array(seq)).squeeze(0)
        seq = torch.from_numpy(seq).cuda()
        
        targets = Variable(targets)
        seq = Variable(seq)
        
        # Forward - now returns features and logits
        triplet_feature, logits = model_obj.m_resnet(seq)
        
        # Triplet Loss part
        triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()
        triplet_label = targets.unsqueeze(0).repeat(triplet_feature.size(0), 1)
        
        (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num, accuracy
         ) = model_obj.triplet_loss(triplet_feature, triplet_label)
        
        if model_obj.hard_or_full_trip == 'hard':
            t_loss = hard_loss_metric.mean()
        else:
            t_loss = full_loss_metric.mean()
            
        # Global ID Loss (CrossEntropy)
        if logits.dim() == 3:
            n_b, n_bins, n_c = logits.size()
            logits_flat = logits.reshape(-1, n_c)
            targets_expanded = targets.unsqueeze(1).repeat(1, n_bins).view(-1)
            id_loss = model_obj.id_loss(logits_flat, targets_expanded)
        else:
            id_loss = model_obj.id_loss(logits, targets)
        
        # Combined Loss: We can weight ID loss a bit higher initially
        loss = t_loss + id_loss
        
        if torch.isfinite(loss):
            loss.backward()
            # Gradient clipping to prevent explosion (NaN)
            torch.nn.utils.clip_grad_norm_(model_obj.m_resnet.parameters(), max_norm=5.0)
            model_obj.optimizer.step()
        else:
            print(f"Warning: Non-finite loss (t_loss: {t_loss.item():.4f}, id_loss: {id_loss.item():.4f}). Skipping step.")
            model_obj.optimizer.zero_grad()
        
        epoch_metrics['loss'].append(loss.item())
        epoch_metrics['t_loss'].append(t_loss.item())
        epoch_metrics['id_loss'].append(id_loss.item())
        epoch_metrics['acc'].append(accuracy.mean().item())
        
        pbar.set_postfix({
            'L': f"{np.mean(epoch_metrics['loss']):.3f}",
            'T': f"{np.mean(epoch_metrics['t_loss']):.3f}",
            'ID': f"{np.mean(epoch_metrics['id_loss']):.3f}",
            'acc': f"{np.mean(epoch_metrics['acc']):.3f}"
        })
        
    return np.mean(epoch_metrics['loss']), np.mean(epoch_metrics['acc'])

def validate(model_obj, val_loader, epoch, total_epochs):
    if val_loader is None:
        return 0, 0
        
    model_obj.m_resnet.eval()
    model_obj.sample_type = 'random'
    
    val_metrics = {
        'loss': [],
        'acc': []
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]")
        for seq, view, seq_type, label, batch_frame in pbar:
            # Filter IDs that might not be in training set (if any) or map them
            targets_list = []
            for li in label:
                if li in model_obj.label_map:
                    targets_list.append(model_obj.label_map[li])
                else:
                    # Fallback for validation identities not in training
                    targets_list.append(0) # Not ideal but prevents crash
            targets = torch.tensor(targets_list).long().cuda()
            seq = np.float32(np.array(seq)).squeeze(0)
            seq = torch.from_numpy(seq).cuda()
            
            triplet_feature, logits = model_obj.m_resnet(seq)
            triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()
            triplet_label = targets.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            
            (full_loss_metric, hard_loss_metric, _, _, accuracy
             ) = model_obj.triplet_loss(triplet_feature, triplet_label)
            
            t_loss = hard_loss_metric.mean() if model_obj.hard_or_full_trip == 'hard' else full_loss_metric.mean()
            
            if logits.dim() == 3:
                n_b, n_bins, n_c = logits.size()
                logits_flat = logits.reshape(-1, n_c)
                targets_expanded = targets.unsqueeze(1).repeat(1, n_bins).view(-1)
                id_loss = model_obj.id_loss(logits_flat, targets_expanded)
            else:
                id_loss = model_obj.id_loss(logits, targets)
                
            loss = t_loss + id_loss
            
            if torch.isfinite(loss):
                val_metrics['loss'].append(loss.item())
                val_metrics['acc'].append(accuracy.mean().item())
            else:
                print(f"Warning: Non-finite validation loss at epoch {epoch+1}. Skipping metrics for this batch.")
            pbar.set_postfix({'loss': f"{np.mean(val_metrics['loss']):.4f}"})
            
    return np.mean(val_metrics['loss']), np.mean(val_metrics['acc'])

def main():
    parser = argparse.ArgumentParser(description='Train GaitGL with SetNet')
    parser.add_argument('--config', type=str, default='src/configs/config_train.yaml')
    parser.add_argument('--resume_iter', type=int, default=0)
    args = parser.parse_args()

    conf = load_config(args.config)
    
    # Setup directories
    output_dir = Path("outputs") / conf["save_name"]
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {conf['index_csv']}...")
    train_src, test_src, val_src = load_data(
        conf["data_root"],
        conf["index_csv"],
        conf["resolution"]
    )

    train_pid_num = len(train_src.label_set) if train_src else 0
    print(f"Initializing model for {train_pid_num} IDs...")
    
    model_obj = Model(
        hidden_dim=conf["hidden_dim"],
        lr=conf["lr"],
        hard_or_full_trip=conf["hard_or_full_trip"],
        margin=conf["margin"],
        num_workers=conf["num_workers"],
        batch_size=(conf["batch_P"], conf["batch_K"]),
        restore_iter=args.resume_iter if args.resume_iter > 0 else conf["restore_iter"],
        total_iter=conf["total_iter"],
        save_name=conf["save_name"],
        train_pid_num=train_pid_num,
        frame_num=conf["frame_num"],
        model_name=conf["model_name"],
        train_source=train_src,
        test_source=test_src,
        img_size=conf["resolution"],
        plots_dir=str(output_dir / "plots"),
        save_iter=conf["save_iter"],
        plot_iter=conf["plot_iter"]
    )
    model_obj.work_dir = str(checkpoint_dir)

    # DataLoaders
    triplet_sampler = TripletSampler(train_src, (conf["batch_P"], conf["batch_K"]))
    train_loader = tordata.DataLoader(
        dataset=train_src,
        batch_sampler=triplet_sampler,
        collate_fn=model_obj.collate_fn,
        pin_memory=True,
        num_workers=conf["num_workers"])
    
    val_loader = None
    if val_src:
        val_sampler = TripletSampler(val_src, (conf["batch_P"], conf["batch_K"]))
        val_loader = tordata.DataLoader(
            dataset=val_src,
            batch_sampler=val_sampler,
            collate_fn=model_obj.collate_fn,
            pin_memory=True,
            num_workers=conf["num_workers"])

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    epochs = conf.get('epochs', 50) # Default to 50 if not specified
    
    print(f"Starting training: {epochs} epochs")
    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model_obj, train_loader, epoch, epochs)
        v_loss, v_acc = validate(model_obj, val_loader, epoch, epochs)
        
        # Step the Learning Rate scheduler
        model_obj.scheduler.step()
        print(f"Epoch {epoch+1} complete. New LR: {model_obj.optimizer.param_groups[0]['lr']}")
        
        history['train_loss'].append(float(t_loss))
        history['train_acc'].append(float(t_acc))
        history['val_loss'].append(float(v_loss))
        history['val_acc'].append(float(v_acc))
        
        # Save last
        model_obj.save("last.pth", epoch=epoch, history=history)
        
        # Save best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            model_obj.save("best.pth", epoch=epoch, history=history)
            print(f"New best model saved! Val Loss: {v_loss:.4f}")
            
        # History to JSON
        with open(output_dir / "history.json", 'w') as f:
            json.dump(history, f)

    print("Training finished.")

if __name__ == '__main__':
    main()
