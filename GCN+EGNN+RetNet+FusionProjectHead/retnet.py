'''
Author:  Kanbe 
Date: 2025-03-12 21:11:21
LastEditors:  Kanbe 
LastEditTime: 2025-03-19 19:57:04
FilePath: /Models/RetNet.py
Description: 
'''
import torch
import torch.nn as nn
from retention import MultiScaleRetention


class RetNet(nn.Module):
    
    def __init__(self, layers, hidden_dim, ffn_size, heads, fixed_seed=1220):
        super(RetNet, self).__init__()
        torch.manual_seed(fixed_seed)
        
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X #MultiScaleRetention(LayerNorm(X))+X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
        return X

   