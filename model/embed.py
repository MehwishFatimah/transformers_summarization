#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, device, vocab_size, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, device, d_model, max_seq_len, dropout = 0.1):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        
        pe = torch.zeros(max_seq_len, d_model)
        print('pe: {}'.format(pe.shape))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        print('pe: {}'.format(pe.shape))
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        print('\nPosEncoder\n-----------------')
        # make embeddings relatively larger
        print('x: {}'.format(x.shape))
        print('x: {}'.format(x))
        print('math.sqrt(self.d_model): {}'.format(math.sqrt(self.d_model)))
        x = x * math.sqrt(self.d_model)
        print('x: {}'.format(x.shape))
        print('after sqt x: {}'.format(x))
        #add constant to embedding
        seq_len = x.size(1)
        print('seq_len: {}'.format(seq_len))

        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        print('pe: {}'.format(pe.shape))
        pe.to(self.device)
        x = x + pe
        print('x + pe: {}'.format(x.shape))
        print('-----------------') 
        return self.dropout(x)