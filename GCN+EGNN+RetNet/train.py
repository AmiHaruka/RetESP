'''
Author:  Kanbe 
Date: 2025-03-26 20:36:50
LastEditors:  Kanbe 
LastEditTime: 2025-04-17 20:39:19
FilePath: /RetESP/train.py
Description: 
'''

import argparse
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix, roc_auc_score
from RetESP import RetESP
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

mp.set_start_method('fork', force=True)
mp.set_forkserver_preload([])  


class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.meta = pickle.load(f)
        self.data = self.meta['data']
        self.max_seq_len = self.meta['max_seq_len']
        self.max_atoms = self.meta['max_atoms']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # 实时生成序列特征
        seq_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        seq = item['sequence']
        seq_tensor[:len(seq)] = torch.tensor(seq)
        # 实时生成分子特征
        atoms = item['atom_ids']
        atom_tensor = torch.zeros(self.max_atoms, dtype=torch.long)
        atom_tensor[:len(atoms)] = torch.tensor(atoms) 

        N = item['adj'].shape[0]
        maxN = self.max_atoms
        adj = torch.zeros(maxN, maxN, dtype=torch.float32)
        adj[:N, :N] = torch.tensor(item['adj'], dtype=torch.float32)
        edge = torch.zeros(maxN, maxN, dtype=torch.float32)
        edge[:N, :N] = torch.tensor(item['edge_attr'], dtype=torch.float32)
           
        return (
            torch.tensor(item['label'], dtype=torch.float32),
            seq_tensor,
            atom_tensor,
            adj,
            edge
        )

def process_data(file_path: str, batch_size: int) -> tuple:
    dataset = LazyDataset(file_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: tuple(torch.stack(samples) for samples in zip(*batch))
    )

def load_datasets(args, phase):
    def create_loader(data_path, shuffle=False):
        dataset = LazyDataset(data_path)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,  # 禁用多进程
            pin_memory=False,
            multiprocessing_context=None,  # 显式禁用多进程上下文
            collate_fn=lambda batch: tuple(
                t.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                non_blocking=True) for t in torch.utils.data._utils.collate.default_collate(batch)
            )
        )
    
    train_loader = create_loader(args.train_data, phase=='train') if phase in ['train', 'both'] else None
    test_loader = create_loader(args.test_data, False) if phase in ['test', 'both'] else None
    
    return (train_loader, test_loader) if phase == 'both' else (train_loader or test_loader)

def initialize_model(args, device):
    config = {
        'gcn_layer': args.gcn_layer,
        'egnn_layer': args.egnn_layer,
        'ret_layer': args.ret_layer,
        'heads': args.heads,
        'lr' : args.lr,
        'WD' : args.WD,
        'hidden_dim': args.hidden_dim,
        'ffn_size': args.ffn_size,
        'projection_dim': args.projection_dim
    }
    
    model = RetESP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.WD)
    return model, optimizer

def train_epoch(model, optimizer, train_loader, device, loss_fn):
    model.train()
    total_loss = 0
    all_cos_sim = []
    pbar = tqdm(train_loader, desc='Train', leave=False)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    grad_clip_value = 1.0
    
    for labels, seq, atoms, adj, edge_attr in train_loader:
        labels = labels.to(device).float()
        seq = seq.to(device)
        atoms = atoms.to(device)
        adj = adj.to(device)
        edge_attr = edge_attr.to(device)
        
        inputs = (seq, atoms, adj, edge_attr)
        optimizer.zero_grad()
        
        protein_embed, mol_embed = model(inputs)
        cos_sim = torch.nn.functional.cosine_similarity(protein_embed, mol_embed)
        loss = loss_fn(cos_sim, labels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_cos_sim.extend(cos_sim.detach().cpu().numpy())
        pbar.set_postfix(loss=total_loss/(pbar.n+1), cos_sim=np.mean(all_cos_sim))
    
    return {
        'loss': total_loss / len(train_loader),
        'cos_sim': np.mean(all_cos_sim)
    }

def run_validation(model, val_loader, loss_fn, device):
    model.eval()
    loss_sum = 0
    total_y_true = []
    total_y_pred = []
    total_y_prob = []
    
    with torch.no_grad():
        for labels, seq, atoms, adj, edge_attr in val_loader:
            seq = seq.to(device)
            atoms = atoms.to(device)
            adj = adj.to(device)
            edge_attr = edge_attr.to(device)
            labels = labels.to(device).float()
            
            inputs = (seq, atoms, adj, edge_attr)
            protein_embed, mol_embed = model(inputs)
            cos_sim = torch.nn.functional.cosine_similarity(protein_embed, mol_embed)
            
            loss = loss_fn(cos_sim, labels)
            loss_sum += loss.item()
            
            y_pred = (cos_sim > 0.5).float().cpu().numpy()
            total_y_true.extend(labels.cpu().numpy())
            total_y_pred.extend(y_pred)
            total_y_prob.extend(cos_sim.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(total_y_true, total_y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    specificity = tn/(tn+fp) 
    sensitivity = tp/(tp+fn)
    bacc = (specificity + sensitivity)/2
    auc = roc_auc_score(total_y_true, total_y_prob)
    
    return {
        'loss': loss_sum/len(val_loader),
        'acc': acc,
        'bacc': bacc,
        'auc': auc,
        'cos_sim': np.mean(total_y_prob)
    }

def main():

    parser = argparse.ArgumentParser(description='RetESP Training Config')
    
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                      help='运行模式: train-仅训练, test-仅测试, both-训练+测试 (默认)')
    
    # datasets
    parser.add_argument('--train_data', type=str, required=lambda x: x in ('train', 'both'),
                      help="训练集路径")
    parser.add_argument('--test_data', type=str, required=lambda x: x in ('test', 'both'),
                      help="测试集路径")
    # date_prepare
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--model_dir', type=str, default='modelstate', help='模型保存目录')
    # config of RetESP
    parser.add_argument('--gcn_layer', type=int, default=2, help='GCN层数')
    parser.add_argument('--egnn_layer', type=int, default=2, help='EGNN层数')
    parser.add_argument('--ret_layer', type=int, default=2, help='RetNet层数')
    parser.add_argument('--heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--WD', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='隐藏层维度')
    parser.add_argument('--ffn_size', type=int, default=1024, help='FFN层维度')
    parser.add_argument('--projection_dim', type=int, default=128, help='投影维度')
    parser.add_argument('--resume', type=bool, default=True, help='路径到已有检查点文件，如果不为空则从该检查点恢复训练')

    
    args = parser.parse_args()
    # writer = SummaryWriter(log_dir=args.model_dir + '/tb_logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    model, optimizer = initialize_model(args, device)
    loss_fn = nn.MSELoss().to(device)

    best_auc = 0
    early_stop_counter = 0
    patience = 100  # 连续100个epoch无改进则停止

    if args.mode in ['train', 'both']:
        train_loader = load_datasets(args, 'train')
        val_loader = load_datasets(args, 'test')

        start_epoch = 1
        best_auc = 0.0
        if args.resume and (Path(args.model_dir)/'best_model.pth').exists() :
            ckpt = torch.load(Path(args.model_dir)/'best_model.pth', map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_auc     = ckpt.get('best_auc', best_auc)
            print(f"Resuming training from epoch {ckpt['epoch']} | best_auc={best_auc:.4f}")

        print(" Now starting training...")
        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            
            train_metrics = train_epoch(model, optimizer, train_loader, device, loss_fn)
            val_metrics = run_validation(model, val_loader, loss_fn, device)

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']

                checkpoint = {
                    'epoch':      epoch,                 
                    'state_dict': model.state_dict(),    
                    'optimizer':  optimizer.state_dict(),
                    'best_auc':   best_auc,              
                    'config':     model.config 
                } 
                torch.save(checkpoint, Path(args.model_dir)/'best_model.pth')
                print(f"Checkpoint saved @ epoch {epoch} | best_auc={best_auc:.4f}")
                
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\n早停触发！连续{patience}个epoch验证集AUC无提升")
                    break

            # 分析
            if epoch == start_epoch:
                df = pd.DataFrame(columns=[
                    'epoch','train_loss','train_cos_sim',
                    'val_loss','val_acc','val_bacc','val_auc','val_cos_sim'
                ])
            df.loc[len(df)] = [
                epoch,
                train_metrics['loss'], train_metrics['cos_sim'],
                val_metrics['loss'], val_metrics['acc'],
                val_metrics['bacc'], val_metrics['auc'],
                val_metrics['cos_sim']
            ]
            df.to_csv(Path(args.model_dir)/'metrics.csv', index=False)
       
            print(f"\nEpoch {epoch}/{args.epochs} | 耗时 {time.time()-start_time:.1f}s")
            print(f"训练集: Loss {train_metrics['loss']:.4f} | CosSim {train_metrics['cos_sim']:.4f}")
            print(f"验证集: Acc {val_metrics['acc']:.4f} | BAcc {val_metrics['bacc']:.4f} | AUC {val_metrics['auc']:.4f}")
        
        #test_metrics = run_validation(model, val_loader, loss_fn, device)
        #print(f"\n测试结果 | AUC {test_metrics['auc']:.4f} | F1 {test_metrics['f1']:.4f}")

if __name__ == '__main__':
    main()
