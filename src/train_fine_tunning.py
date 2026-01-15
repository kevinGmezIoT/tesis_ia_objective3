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

def set_frozen_status(model, frozen=True):
    """Freezes or unfreezes the backbone of SetNet."""
    # Backbone layers
    backbone_keywords = ['set_layer', 'gl_layer']
    for name, param in model.named_parameters():
        is_backbone = any(kw in name for kw in backbone_keywords)
        if is_backbone:
            param.requires_grad = not frozen
    
    status = "FROZEN" if frozen else "UNFROZEN"
    print(f">>> Backbone status: {status}")

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
        
        targets = torch.tensor([model_obj.label_map[li] for li in label]).long().cuda()
        seq = np.float32(np.array(seq)).squeeze(0)
        seq = torch.from_numpy(seq).cuda()
        
        targets = Variable(targets)
        seq = Variable(seq)
        
        triplet_feature, logits = model_obj.m_resnet(seq)
        
        triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()
        triplet_label = targets.unsqueeze(0).repeat(triplet_feature.size(0), 1)
        
        (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num, accuracy
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_obj.m_resnet.parameters(), max_norm=10.0)
            model_obj.optimizer.step()
        else:
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
    
    val_metrics = {'loss': [], 'acc': []}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]")
        for seq, view, seq_type, label, batch_frame in pbar:
            targets_list = [model_obj.label_map[li] if li in model_obj.label_map else 0 for li in label]
            targets = torch.tensor(targets_list).long().cuda()
            seq = np.float32(np.array(seq)).squeeze(0)
            seq = torch.from_numpy(seq).cuda()
            
            triplet_feature, logits = model_obj.m_resnet(seq)
            triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()
            triplet_label = targets.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            
            (full_loss_metric, hard_loss_metric, _, _, accuracy) = model_obj.triplet_loss(triplet_feature, triplet_label)
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
            pbar.set_postfix({'loss': f"{np.mean(val_metrics['loss']):.4f}"})
            
    return np.mean(val_metrics['loss']), np.mean(val_metrics['acc'])

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GaitGL with SetNet and Frozen Backbone')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to the .pth file')
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Number of epochs to freeze backbone')
    parser.add_argument('--max_ids', type=int, default=0, help='Limit number of IDs for debugging')
    args = parser.parse_args()

    conf = load_config(args.config)
    output_dir = Path("outputs") / conf["save_name"]
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {conf['index_csv']}...")
    train_src, test_src, val_src = load_data(conf["data_root"], conf["index_csv"], conf["resolution"])

    if args.max_ids > 0:
        print(f"DEBUG MODE: Limiting training to {args.max_ids} IDs")
        selected_ids = sorted(list(train_src.label_set))[:args.max_ids]
        def filter_src(src, ids):
            if src is None: return None
            indices = [i for i, label in enumerate(src.label) if label in ids]
            src.label = [src.label[i] for i in indices]
            src.seq_dir = [src.seq_dir[i] for i in indices]
            src.seq_type = [src.seq_type[i] for i in indices]
            src.view = [src.view[i] for i in indices]
            src.label_set = set(src.label)
            src.data_size = len(src.label)
            src.__init__(src.seq_dir, src.label, src.seq_type, src.view, src.cache, src.resolution)
            return src
        train_src = filter_src(train_src, selected_ids)
        val_src = filter_src(val_src, selected_ids)

    train_pid_num = len(train_src.label_set) if train_src else 0
    print(f"Initializing model for {train_pid_num} IDs...")
    
    model_obj = Model(
        hidden_dim=conf["hidden_dim"],
        lr=conf["lr"],
        hard_or_full_trip=conf["hard_or_full_trip"],
        margin=conf["margin"],
        num_workers=conf["num_workers"],
        batch_size=(conf["batch_P"], conf["batch_K"]),
        restore_iter=0,
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

    # LOAD PRETRAINED WEIGHTS
    print(">>> Performing flexible weight loading...")
    model_obj.load_checkpoint(args.pretrained_model)

    # Initial Freeze
    if args.freeze_epochs > 0:
        set_frozen_status(model_obj.m_resnet, frozen=True)
        # Update optimizer to only focus on head for clarity
        model_obj.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_obj.m_resnet.parameters()), 
                                       lr=conf["lr"] * 2, weight_decay=1e-4)

    # DataLoaders
    triplet_sampler = TripletSampler(train_src, (conf["batch_P"], conf["batch_K"]))
    train_loader = tordata.DataLoader(dataset=train_src, batch_sampler=triplet_sampler, collate_fn=model_obj.collate_fn, pin_memory=True, num_workers=conf["num_workers"])
    
    val_loader = None
    if val_src:
        val_sampler = TripletSampler(val_src, (conf["batch_P"], conf["batch_K"]))
        val_loader = tordata.DataLoader(dataset=val_src, batch_sampler=val_sampler, collate_fn=model_obj.collate_fn, pin_memory=True, num_workers=conf["num_workers"])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs = conf.get('epochs', 60)
    
    print(f"Starting Training: {epochs} epochs (First {args.freeze_epochs} frozen)")
    for epoch in range(epochs):
        # Check for unfreeze
        if epoch == args.freeze_epochs and args.freeze_epochs > 0:
            print(">>> Unfreezing backbone for full fine-tuning...")
            set_frozen_status(model_obj.m_resnet, frozen=False)
            # Re-initialize optimizer with differential learning rates
            # 1e-5 for backbone, conf["lr"] for head
            backbone_params = []
            head_params = []
            for name, param in model_obj.m_resnet.named_parameters():
                if any(kw in name for kw in ['set_layer', 'gl_layer']):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            model_obj.optimizer = optim.Adam([
                {'params': backbone_params, 'lr': conf["lr"]},
                {'params': head_params, 'lr': conf["lr"] * 3}
            ], weight_decay=1e-4)
            # Update scheduler too
            model_obj.scheduler = optim.lr_scheduler.StepLR(model_obj.optimizer, step_size=20, gamma=0.5)

        t_loss, t_acc = train_epoch(model_obj, train_loader, epoch, epochs)
        v_loss, v_acc = validate(model_obj, val_loader, epoch, epochs)
        
        model_obj.scheduler.step()
        
        history['train_loss'].append(float(t_loss))
        history['train_acc'].append(float(t_acc))
        history['val_loss'].append(float(v_loss))
        history['val_acc'].append(float(v_acc))
        
        model_obj.save("last_ft.pth", epoch=epoch, history=history)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            model_obj.save("best_ft.pth", epoch=epoch, history=history)
            print(f"New best model saved! Val Loss: {v_loss:.4f}")
            
    print("Fine-tuning finished.")

if __name__ == '__main__':
    main()
