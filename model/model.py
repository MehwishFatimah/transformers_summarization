#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:42:38 2020

@author: fatimamh
"""

import model_config as config

import torch
import torch.nn as nn 
from model.layers import EncoderLayer, DecoderLayer
from model.embed import Embedder, PositionalEncoder
from model.sublayers import Norm
import copy

'''---------------------------------------------------------------
'''
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

'''---------------------------------------------------------------
'''
class Encoder(nn.Module):
    def __init__(self, device, vocab_size, d_model, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(device, vocab_size, d_model)
        
        self.pe = PositionalEncoder(device, d_model, max_seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(device, d_model, heads, dropout), N)
        self.norm = Norm(device, d_model)

    def forward(self, src, mask):
        #print('\nEncoder\n-----------------')
        x = self.embed(src)
        #print('x-emb: {}'.format(x.shape))

        x = self.pe(x)
        #print('x-pe: {}'.format(x.shape))

        for i in range(self.N):
            x = self.layers[i](x, mask)
            #print('x-layer{}: {}'.format(i, x.shape))
        x = self.norm(x)
        #print('x-norm: {}'.format(x.shape))
        #print('-----------------')        
        return x

'''---------------------------------------------------------------
'''    
class Decoder(nn.Module):
    def __init__(self, device, vocab_size, d_model, max_seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(device, vocab_size, d_model)
        self.pe = PositionalEncoder(device, d_model, max_seq_len, dropout=dropout)
        self.layers = get_clones(DecoderLayer(device, d_model, heads, dropout), N)
        self.norm = Norm(device, d_model)
    
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        #print('\nDecoder\n-----------------')
        x = self.embed(trg)
        #print('x-emb: {}'.format(x.shape))

        x = self.pe(x)
        #print('x-pe: {}'.format(x.shape))
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
            #print('x-layer{}: {}'.format(i, x.shape))
        
        x = self.norm(x)
        #print('x-norm: {}'.format(x.shape))
        #print('-----------------')        
        return x
        
'''---------------------------------------------------------------
'''
class Transformer(nn.Module):

    def __init__(self, device, src_vocab, trg_vocab, d_model, max_text_len, max_sum_len, N, heads, dropout):
        super().__init__()
        
        self.encoder = Encoder(device, src_vocab, d_model, max_text_len, N, heads, dropout)
        self.decoder = Decoder(device, trg_vocab, d_model, max_sum_len, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask):
        #print("ENCODER")
        e_outputs = self.encoder(src, src_mask)
        #print('e_outputs: {}'.format(e_outputs.shape))
        
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        #print('d_output: {}'.format(d_output.shape))
        #print()
        #print("OUTPUT")
        output = self.out(d_output)
        #print('output: {}'.format(output.shape))

        return output

'''---------------------------------------------------------------
'''
def get_model(device):
    
    assert config.d_model % config.n_heads == 0
    assert config.dropout < 1
    # from config to here, propogate to over all model modules
    model = Transformer(device, 
                        config.text_vocab, 
                        config.sum_vocab, 
                        config.d_model, 
                        config.max_text, 
                        config.max_sum, 
                        config.n_layers, 
                        config.n_heads, 
                        config.dropout)
    model = model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
        
    return model

'''
if __name__ == '__main__':
    
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    test = get_model(device)
    print(test.__dict__)
'''