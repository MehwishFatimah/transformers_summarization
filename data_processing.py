#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:15:48 2019
Updated: 02 Dec 19
@author: fatimamh
"""
import resource
import time
import os
import sys
import numpy as np
import pandas as pd

from utils.memory_utils import get_size

# data processing utils
from utils.data_utils import Data
from utils.lang_utils import *
import data_config as config

class Process(object):
    def __init__(self):
        #self.config = dconfig
        self.text_vocab = Lang('text')
        self.sum_vocab = Lang('sum')

    def prepare_vocabs(self, df):
        source = df['text']
        target = df['summary']
        
        for i in range(len(source)):
            self.text_vocab.add_text(source[i])
            self.sum_vocab.add_text(target[i])    
           
    '''---------------------------------------------------------------'''
    def process_vacabs(self):
        
        files = config.csv_files 
        in_folder = config.root_in 
        dict_folder = config.vocab_folder
          
        for file in files:    
            file  = os.path.join(in_folder, file)
            df = pd.read_csv(file, encoding = 'utf-8')
        
            print('Training data:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
            print('text_vocab: {}, sum_vocab: {}'.format(self.text_vocab.n_words, self.sum_vocab.n_words)) 
            print()
            self.prepare_vocabs(df)
            print('text_vocab: {}, sum_vocab: {}'.format(self.text_vocab.n_words, self.sum_vocab.n_words)) 
            print('--------------------')
        
        # Store original full dictionaries
        print('text_vocab: {}, sum_vocab: {}'.format(self.text_vocab.n_words, self.sum_vocab.n_words))
        f_input = save_vocab(self.text_vocab, config.text_vocab_f)
        f_output = save_vocab(self.sum_vocab, config.sum_vocab_f)
        
        # Make condense dictionaries
        self.text_vocab.condensed_vocab(config.text_vocab)
        self.sum_vocab.condensed_vocab(config.sum_vocab)
        
        f_input = save_vocab(self.text_vocab, config.text_vocab_c)
        f_output = save_vocab(self.sum_vocab, config.sum_vocab_c)
        print()
        print('text_vocab: {}, sum_vocab: {}'.format(self.text_vocab.n_words, self.sum_vocab.n_words))
        # send condensed vocab, because these will use for tensors
        return f_input, f_output

    '''---------------------------------------------------------------'''
    def prepare_tensors(self, df, folder):
        print('here')
        source = df['text']
        target = df['summary']
            
        for i in range(len(source)):
            input_tensor = text_to_tensor(self.text_vocab, source[i], config.max_text)
            #print('input_tensor: {}'.format(input_tensor.shape))
            f_name = 'input_' + str(i+1) + '.pt'
            file = os.path.join(folder, f_name)
            #print(file)
            torch.save(input_tensor, file)

            target_tensor = text_to_tensor(self.sum_vocab, target[i], config.max_sum)
            #print('target_tensor: {}'.format(target_tensor.shape))
            f_name = 'target_' + str(i+1) + '.pt'
            file = os.path.join(folder, f_name) 
            #print(file)
            torch.save(target_tensor, file)

    '''---------------------------------------------------------------'''

    def process_tensors(self):    
        files = config.csv_files 
        in_folder = config.root_in 
        print(in_folder)
        
        print(self.text_vocab.n_words)
        print(self.sum_vocab.n_words)
        print()
        for file in files:
            folder = file.split('_')[1]
            #print(folder)
            file  = os.path.join(in_folder, file)
            df = pd.read_csv(file, encoding = 'utf-8')
            #df = df.head(5)
            #print('\n=========================================')
            print('Training data:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
            folder = os.path.join(in_folder, folder)
            #print(folder)
            self.prepare_tensors(df, folder)
            print('--------------------')

    '''---------------------------------------------------------------'''
    def data_processing(self):
        # Step 1: Convert data from json to csv. CLEAN AND SHORT 
        start_time = time.time()
        data = Data()
        print("cleaning data")
        #data.process_data(self.config)
        print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
            format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
        
        # Step 2: Generate vocabs
        start_time = time.time()
        print("processing data")
        in_vocab, out_vocab = self.process_vacabs()
        print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
            format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
 
        # Step 3:Generate tensors
        start_time = time.time()
        print("processing tensors")
        self.process_tensors()
        print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
            format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
       
'''----------------------------------------------------------------

if __name__ == '__main__':
    
    test = Process()
    test.data_processing()
'''    
