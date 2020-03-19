#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""

import torch
import torch.nn as nn
from model.sublayers import FeedForward, MultiHeadAttention, Norm

'''-------------------------------------------------------
'''
class EncoderLayer(nn.Module):
    def __init__(self, device, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(device, d_model)
        self.norm_2 = Norm(device, d_model)
        self.attn = MultiHeadAttention(device, heads, d_model, dropout=dropout)
        self.ff = FeedForward(device, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        print('\nEncLayer\n-----------------')
        x2 = self.norm_1(x)
        print('x-norm1: {}'.format(x2.shape))
        
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        print('x-drop: {}'.format(x.shape))
        
        x2 = self.norm_2(x)
        print('x-norm2: {}'.format(x2.shape))

        x = x + self.dropout_2(self.ff(x2))
        print('x-drop: {}'.format(x.shape))
        print('-----------------')        
        
        return x

'''-------------------------------------------------------
'''    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, device, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(device, d_model)
        self.norm_2 = Norm(device, d_model)
        self.norm_3 = Norm(device, d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(device, heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(device, heads, d_model, dropout=dropout)
        self.ff = FeedForward(device, d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        print('\nDecLayer\n-----------------')
        x2 = self.norm_1(x)
        print('x-norm1: {}'.format(x2.shape))

        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        print('x-drop: {}'.format(x.shape))

        x2 = self.norm_2(x)
        print('x-norm2: {}'.format(x2.shape))

        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        print('x-drop: {}'.format(x.shape))
        
        x2 = self.norm_3(x)
        print('x-norm3: {}'.format(x2.shape))
        
        x = x + self.dropout_3(self.ff(x2))
        print('x-drop: {}'.format(x.shape))
        print('-----------------')
        return x

