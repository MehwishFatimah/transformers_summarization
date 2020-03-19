#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:28:08 2019
Modified on Wed Nov 6
Modified on Wed Feb 12
@author: fatimamh
"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import os
import argparse
import time
import resource
import random
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
import pickle
import dill
import json
#import model_config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_index = 0
UNK_index = 1
SP_index  = 2
EP_index  = 3   


# For freq
class Lang:
    def __init__(self, name=None):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "UNK", 2: "SP", 3: "EP"} # add UNK
        self.n_words = 4  # Count PAD, UNK, SP and EP

    '''---------------------------------------------'''
    def load_lang(self, data):
        self.name       = data['name']
        self.word2index = data['word2index']
        self.word2index = {k:int(v) for k,v in self.word2index.items()}
        
        self.word2count = data['word2count']
        self.word2count = {k:int(v) for k,v in self.word2count.items()}
        
        self.index2word = data['index2word']
        self.index2word = {int(k):v for k,v in self.index2word.items()}

        self.n_words    = int(data['n_words']) 

    '''---------------------------------------------'''
    def add_text(self, text):
        for word in text.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    '''---------------------------------------------'''
    def reset(self):
        self.word2index.clear()
        self.index2word.clear()
        self.word2count.clear()
        self.index2word = {0: "PAD", 1: "UNK", 2: "SP", 3: "EP"} # add UNK
        self.n_words = 4

    def refill(self, text):
        for word in text:
            self.add_word(word)
    
    '''---------------------------------------------'''
    def filter_most_common(self, ratio):
        sorted_list = Counter(OrderedDict(sorted(self.word2count.items(), key=lambda t: t[1], reverse=True)))
        print(len(sorted_list))
        sorted_list = sorted_list.most_common(ratio)
        sorted_list =  [i[0] for i in sorted_list]
        return sorted_list   
            
    def condensed_vocab(self, ratio):
        new_list = self.filter_most_common(ratio)
        self.reset()
        self.refill(new_list)    
    
    '''---------------------------------------------'''
    # Remove words below a certain threshold
    def filter_least_common(self, min_count):
        keep_words = list()
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        return keep_words 

    def trimmed_vocab(self, min_count):
        new_list = self.filter_least_common(min_count)
        self.reset()
        self.refill(new_list)     
        
        
'''
====================================================================
'''

def save_vocab(obj, file):
    
    print(file)
    with open(file, "w") as f:
        json.dump(json.dumps(obj.__dict__), f)
        return file
    
'''---------------------------------------------------------------'''
def load_vocab(file):
    
    with open(file, 'rb') as f:
        data = json.loads(json.load(f))
        print(type(data))
        vocab = Lang()
        vocab.load_lang(data)
        return vocab

'''---------------------------------------------------------------'''    
# modified for handling UNK
def vectorize(vocab, text):
    vector = list()
    for word in text.split():
        if word in vocab.word2index:
            #print('word: {}, index: {}'.format(word, vocab.word2index[word]))
            vector.append(vocab.word2index[word])
        else:
            #print('word: {}, UNK_index: {}'.format(word, UNK_index))
            vector.append(UNK_index)
    #print(vector)
    return vector 


'''---------------------------------------------------------------'''
def tensor_to_text(vocab, vector):
    
    #print(vocab.n_words)
    words = list()
    for i in range(len(vector)):
        idx = vector[i]
        if idx == PAD_index: continue
        #print('idx: {} == word: {}'.format(idx, vocab.index2word[idx]))
        words.append(vocab.index2word[idx])
    #print(vector)
    words = ' '.join(map(str, words)) 
    #print(words)
    return words 

'''---------------------------------------------------------------'''
def text_to_tensor(vocab, text, max_len):
    #print(lang.name)
    indexes = vectorize(vocab, text)
    indexes.append(EP_index)
    
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)




'''
if __name__ == '__main__':
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test = object_load(config.sum_vocab_c)
    print(test.__dict__)
'''    
