import math
import os
import os.path as osp
import random
import sys
from datetime import datetime
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim import lr_scheduler
from .network import TripletLoss, SetNet
from .network import vgg_c3d
from .utils import TripletSampler
from glob import glob
from os.path import join

import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=112,
                 plots_dir='./runs/plots',
                 save_iter=10000,
                 plot_iter=10000):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size
        self.plots_dir = plots_dir
        self.save_iter = save_iter
        self.plot_iter = plot_iter
        self.hidden_dim = hidden_dim
        # Initialize history for plotting
        self.history = {
            'iter': [],
            'hard_loss': [],
            'full_loss': [],
            'full_loss_num': [],
            'mean_dist': [],
            'accuracy': []
        }
        
        # Create a robust mapping from person_id to [0, num_classes-1]
        # This prevents CUDA device-side asserts (out of bounds labels)
        self.label_list = sorted(list(self.train_source.label_set))
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
        self.train_pid_num = len(self.label_list)
        print(f"Mapped {self.train_pid_num} unique IDs to range [0, {self.train_pid_num-1}]")

        print(train_pid_num)
        if self.model_name == "SetNet":
            self.m_resnet = SetNet(hidden_dim=self.hidden_dim, num_classes=self.train_pid_num)
        elif self.model_name == "Vgg_c3d":
            self.m_resnet = vgg_c3d.c3d_vgg_Fusion(hidden_dim=self.hidden_dim, num_classes=self.train_pid_num)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
            
        print("CUDA available:", torch.cuda.is_available())
        print("Device count:", torch.cuda.device_count())
        
        if torch.cuda.is_available():
            self.m_resnet = self.m_resnet.cuda()
            if torch.cuda.device_count() > 1:
                self.m_resnet = nn.DataParallel(self.m_resnet)
        else:
            print("Running on CPU")


        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        if torch.cuda.is_available():
            self.triplet_loss = nn.DataParallel(self.triplet_loss)
            self.triplet_loss.cuda()

        # Differential Learning Rate & Weight Decay Optimization
        # Classification heads learn faster to break the initial collapse
        backbone_params = []
        head_params = []
        for name, param in self.m_resnet.named_parameters():
            if 'fc_id' in name or 'fc_bin' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': self.lr},
            {'params': head_params, 'lr': self.lr * 3}
        ], weight_decay=1e-4) # Reduced from 5e-4 to allow better feature separation
        
        # Loss function for classification head with Label Smoothing
        self.id_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning Rate scheduler: Relaxed to give time for 4000 IDs convergence
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        print(f"Optimizer initialized with differential LR. Head LR: {self.lr*3}, Backbone LR: {self.lr}")
        print(self.optimizer)


        # self.losscse= []

        self.sample_type = 'all'

    def collate_fn(self, batch):
        # print('collate_fn-1-',batch.shape)
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def Order_select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                if len(frame_set)>self.frame_num-1:
                    choiceframe_set = frame_set[:len(frame_set)-self.frame_num+1]
                    frame_id_list= random.choices(choiceframe_set, k=1)
                    for i in range(self.frame_num-1):  
                        frame_id_list.append(frame_id_list[0]+(i+1))
                    _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    frame_id_list = [0]
                    for i in range(len(frame_set)-1):  
                        frame_id_list.append(frame_id_list[0]+(i+1))    
                    len_frame_id_list = len(frame_id_list)
                    for ll in range(self.frame_num-len_frame_id_list):
                        frame_id_list.append(frame_id_list[ll])
                    _ = [feature.loc[frame_id_list].values for feature in sample] 
            else:
                _ = [feature.values for feature in sample]              
            return _
        def select_frame(index):
            sample = seqs[index]
            # print(sample[0].shape)
            frame_set = frame_sets[index]
            
            if self.sample_type == 'random':
                frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                frame_id_list = sorted(frame_id_list)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs1 = []
        count = 1
        for sq in range(len(seqs)):
            seqs1.append(select_frame(sq))
            count +=1
        seqs = seqs1

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            # print('--2--')
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            # print(gpu_num,batch_per_gpu,batch_frames)
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

            # seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.m_resnet.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        # print('---1--')
        # print('batch_size-',self.batch_size)
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers)
        print('-len(train_loader)-',len(train_loader))
        train_label_set = list(self.train_source.label_set)
        # Track previous epoch to step scheduler
        last_epoch = 0
        for seq, view, seq_type, label, batch_frame in train_loader:
            # Step the Learning Rate scheduler based on iterations
            current_epoch = self.restore_iter // len(train_loader)
            if current_epoch > last_epoch:
                self.scheduler.step()
                last_epoch = current_epoch
                print(f"\nStep LR Scheduler: New LR = {self.optimizer.param_groups[0]['lr']}")

            self.restore_iter += 1
            self.optimizer.zero_grad()
            
            # Map labels to continuous indices [0, N-1]
            label_indices = [self.label_map[li] for li in label]
            targets = np.array(label_indices)
            #---------------seq-----------------------
            seq=np.array(seq)
            seq = np.float32(seq)
            seq = seq.squeeze(0)

            
            targets = torch.from_numpy(targets)
            targets = targets.cuda()
            seq = torch.from_numpy(seq)
            # seq = seq.unsqueeze(1) # For C3D
            seq = seq.cuda()



            targets = Variable(targets)
            seq = Variable(seq)
            # SetNet returns (n, bins, dim) and logits for classification
            triplet_feature, logits = self.m_resnet(seq)
            
            #--------------tri-------------------------
            # triplet_feature: [batch, bins, dim]
            # TripletLoss expects: [bins, batch, dim]
            triplet_feature = triplet_feature.permute(1, 0, 2).contiguous()
            triplet_label = targets.unsqueeze(0)
            triplet_label = triplet_label.repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num, accuracy
             ) = self.triplet_loss(triplet_feature, triplet_label)

            #--------------Combined Loss----------------
            # Triplet part
            if self.hard_or_full_trip == 'hard':
                t_loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                t_loss = full_loss_metric.mean()
            
            # Classification part (ID loss)
            # logits: [batch, bins, num_classes], targets: [batch]
            # We calculate CE loss for each bin and then average
            if logits.dim() == 3:
                n_b, n_bins, n_c = logits.size()
                logits_flat = logits.reshape(-1, n_c)
                targets_expanded = targets.unsqueeze(1).repeat(1, n_bins).view(-1)
                c_loss = self.id_loss(logits_flat, targets_expanded)
            else:
                c_loss = self.id_loss(logits, targets)
            
            # Final combined loss
            loss = t_loss + c_loss


            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            self.accuracy_list.append(accuracy.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % self.save_iter == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()
                self.save()
            
            if self.restore_iter % self.plot_iter == 0:
                self.plot_history()
            if self.restore_iter % 150000 == 0:
                self.optimizer = optim.Adam([
                    {'params': self.m_resnet.parameters()}], lr=0.00001)  

            if self.restore_iter % 100 == 0:
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', acc={0:.4f}'.format(np.mean(self.accuracy_list)), end='')
                print(', id_loss={0:.4f}'.format(c_loss.item()), end='') # New: show ID loss
                # print(', cse_loss_num={0:.8f}'.format(np.mean(self.losscse)), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                # Update history
                self.history['iter'].append(self.restore_iter)
                self.history['hard_loss'].append(np.mean(self.hard_loss_metric))
                self.history['full_loss'].append(np.mean(self.full_loss_metric))
                self.history['full_loss_num'].append(np.mean(self.full_loss_num))
                self.history['mean_dist'].append(self.mean_dist)
                self.history['accuracy'].append(np.mean(self.accuracy_list))

                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []
                self.accuracy_list = []



            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        with torch.no_grad():
            self.m_resnet.eval()
            source = self.test_source if flag == 'test' else self.train_source
            self.sample_type = 'all'
            data_loader = tordata.DataLoader(
                dataset=source,
                batch_size=batch_size,
                sampler=tordata.sampler.SequentialSampler(source),
                collate_fn=self.collate_fn,
                pin_memory=True,
                num_workers=self.num_workers)

            feature_list = list()
            view_list = list()
            seq_type_list = list()
            label_list = list()

            counttt=0
            for i, x in enumerate(data_loader):
                seq, view, seq_type, label, batch_frame = x
                seq=np.array(seq)
                seq = np.float32(seq)
                seq = torch.from_numpy(seq)
                seq = seq.squeeze(0)
                # seq = seq.unsqueeze(1) # For C3D
                seq = Variable(seq.cuda())
                # batcht,channelt,framet,ht,wt = seq.size()
                outputs,_ = self.m_resnet(seq)
                if counttt % 1000 == 0:
                    print(label, seq_type, view, '--', outputs.shape)
                counttt += 1
                
                # outputs: [batch, bins, dim] -> [batch, bins * dim]
                n, bins, dim = outputs.size()
                outputs = outputs.view(n, -1).data.cpu().numpy()
                
                feature_list.append(outputs)
                view_list += view
                seq_type_list += seq_type
                label_list += label


        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self, filename="last.pth", epoch=None, history=None):
        os.makedirs(self.work_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.m_resnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'restore_iter': self.restore_iter,
            'epoch': epoch,
            'history': history
        }
        save_path = osp.join(self.work_dir, filename)
        torch.save(checkpoint, save_path)
        # print(f"Checkpoint saved: {save_path}")

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        print(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter)))
        self.m_resnet.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

    def load_checkpoint(self, path):
        """Loads a checkpoint flexible, ignoring size mismatches for fine-tuning."""
        print(f"Loading weights from: {path}")
        checkpoint = torch.load(path)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model_dict = self.m_resnet.state_dict()
        
        # Filter out keys with different shapes (useful for changing num_classes)
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping layer {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"Skipping layer {k} as it is not in the current model")
        
        model_dict.update(filtered_dict)
        self.m_resnet.load_state_dict(model_dict)
        print(f"Successfully loaded {len(filtered_dict)} layers from checkpoint.")
        return checkpoint.get('epoch', 0), checkpoint.get('history', {})



    def plot_history(self):
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['iter'], self.history['hard_loss'], label='Hard Loss')
        plt.plot(self.history['iter'], self.history['full_loss'], label='Full Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(osp.join(self.plots_dir, 'loss.png'))
        plt.close()

        # Plot Mean Distance
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['iter'], self.history['mean_dist'], label='Mean Distance', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Mean Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig(osp.join(self.plots_dir, 'distance.png'))
        plt.close()

        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['iter'], self.history['accuracy'], label='Accuracy', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Triplet Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(osp.join(self.plots_dir, 'accuracy.png'))
        plt.close()
