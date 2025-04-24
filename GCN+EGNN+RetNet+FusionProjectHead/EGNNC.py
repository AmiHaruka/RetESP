'''
Author:  Kanbe 
Date: 2025-03-19 20:41:19
LastEditors:  Kanbe 
LastEditTime: 2025-04-16 20:04:44
FilePath: /RetESP/EGNNC.py
Description: 
'''
import torch
import math
from torch import nn


class EGNNC(nn.Module):
    
    """
    Simple EGNN(C) layer, similar to https://arxiv.org/abs/1809.02709
    """
    
    def __init__(self, in_features, out_features, bias=True):
        
        super(EGNNC, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.L = nn.LeakyReLU()
        self.BN = nn.BatchNorm1d(out_features)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, E):
        EX = torch.bmm(E, X)                 
        Y = EX @ self.W                       
        if self.bias is not None:
            Y = Y + self.bias                
     
        Y = Y.permute(0, 2, 1)                
        Y = self.BN(Y)                        
        Y = Y.permute(0, 2, 1)               
        
        return self.L(Y)


if __name__ == '__main__':
    E=torch.randn((9,9))
    X=torch.randn((9,3))
    model=EGNNC(3,3,bias=True)
    print(model.forward(X,E))
