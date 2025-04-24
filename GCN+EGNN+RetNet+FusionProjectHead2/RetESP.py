'''
Author:  Kanbe 
Date: 2025-03-19 19:55:18
LastEditors:  Kanbe 
LastEditTime: 2025-04-23 14:15:48
FilePath: /RetESP/GCN+EGNN+RetNet+FusionProjectHead2/RetESP.py
Description: 
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from retnet import RetNet
from GCN import GCN
from EGNNC import EGNNC
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

class RetESP(nn.Module):
    
    def __init__(self, config=None):
        torch.manual_seed(114514)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(114514)
        super().__init__()
        
        if config is None:
            self.config = {
                'gcn_layer': 2, 'egnn_layer': 2, 
                'ret_layer': 2, 'heads': 4, 
                'lr': 0.001, 'WD': 0.0005,
                'hidden_dim': 1024, 
                'ffn_size': 1024,    
                'projection_dim': 128
            }
        else:
            self.config = config
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in self.config:
            self.__dict__[i] = self.config[i]

        self.seq_ebd = nn.Embedding(9000, self.hidden_dim)
        nn.init.xavier_uniform_(self.seq_ebd.weight)
        self.retnet = RetNet(
            layers=config['ret_layer'],
            hidden_dim=config['hidden_dim'],
            ffn_size=config['ffn_size'],
            heads=config['heads']
        )
        
        self.mol_ebd = nn.Embedding(4096, self.hidden_dim)
        nn.init.xavier_uniform_(self.mol_ebd.weight)
        self.GCN = nn.ModuleList([GCN(self.hidden_dim, self.hidden_dim) 
                                for _ in range(self.gcn_layer)])
        self.EGNN = nn.ModuleList([EGNNC(self.hidden_dim, self.hidden_dim) 
                                 for _ in range(self.egnn_layer)])
        
        self.protein_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )
        self.mol_proj = nn.Sequential(
            nn.Linear(self.hidden_dim*2, 768),  # 合并GCN和EGNN
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

        for name, param in self.named_parameters():
            param.requires_grad = False

        # ProjectHead
        D = self.projection_dim
        self.pro_refine_layer      = nn.Linear(D, D)
        self.mol_refine_layer      = nn.Linear(D, D)
        self.pro_refine_layer2      = nn.Linear(D, D)
        self.mol_refine_layer2      = nn.Linear(D, D)
        self.pro_batch_norm_layer  = nn.BatchNorm1d(D)
        self.mol_batch_norm_layer  = nn.BatchNorm1d(D)
        self.shared_batch_norm     = nn.BatchNorm1d(D)
        self.relu                  = nn.ReLU()

        # ensure ProjectHead params are trainable
        for name, param in self.named_parameters():
            if name.startswith("pro_refine_") or name.startswith("mol_refine_") \
            or name.startswith("pro_batch_norm_") or name.startswith("mol_batch_norm_") \
            or name.startswith("shared_batch_norm"):
                param.requires_grad = True

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay=self.WD)
    
    def encode_protein(self, seq):
        x = self.seq_ebd(seq)        
        x = self.retnet(x)           
        x = x.mean(dim=1)           
        return self.protein_proj(x)  


    def encode_mol(self, mol, A, E):
        xs = self.mol_ebd(mol)
        
        # GCN
        xn = xs
        for layer in self.GCN:
            xn = F.leaky_relu(layer(xn, A))   
            
        # EGNN-attention
        xe = xs
        for layer in self.EGNN:
            xe = layer(xe, E) * torch.sigmoid(xe)  

        attn = torch.sigmoid((xn * xe).sum(-1))  
        attn = attn.unsqueeze(-1)                
        x_gcn = (attn * xn).sum(dim=1)          
        x_egnn = ((1-attn) * xe).sum(dim=1)       
        combined = torch.cat([x_gcn, x_egnn], dim=-1) 
        mol_embed = self.mol_proj(combined)            
        return mol_embed

    def forward(self, inputs):
        seq, mol, A, E = inputs
        protein_embed = self.encode_protein(seq)
        mol_embed = self.encode_mol(mol, A, E)

        refined_pro_embed = self.pro_refine_layer(protein_embed)
        refined_mol_embed = self.mol_refine_layer(mol_embed)

        refined_pro_embed = self.pro_batch_norm_layer(refined_pro_embed)
        refined_mol_embed = self.mol_batch_norm_layer(refined_mol_embed)

        refined_pro_embed = self.relu(refined_pro_embed)
        refined_mol_embed = self.relu(refined_mol_embed)

        refined_pro_embed = self.pro_refine_layer(refined_pro_embed)
        refined_mol_embed = self.mol_refine_layer(refined_mol_embed)

        refined_pro_embed = self.shared_batch_norm(refined_pro_embed)
        refined_mol_embed = self.shared_batch_norm(refined_mol_embed)

        refined_pro_embed = torch.nn.functional.normalize(refined_pro_embed, dim=1)
        refined_mol_embed = torch.nn.functional.normalize(refined_mol_embed, dim=1)

        return refined_pro_embed, refined_mol_embed
