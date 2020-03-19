#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:21:02 2019
@author: fatimamh

"""

import argparse
import resource

import time
from datetime import datetime

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True # for fast training

from utils.file_utils import read_content
from utils.memory_utils import get_size

from data_processing import Process
from training import Train
#from testing import Test
#from evaluation import Evaluate

import model_config as config

'''----------------------------------------------------------------
'''
parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--p', type = bool,   default = False,   help = 'To process data.')

parser.add_argument('--ts', type = bool,   default = False,   help = 'To train the model from scratch.')
parser.add_argument('--tr', type = bool,   default = False,   help = 'To train the model by resuming.')
parser.add_argument('--er', type = int,   default = -1,   help = 'Epoch to resume.')
parser.add_argument('--tf', type = str,   default = None,   help = 'file to resume.')

parser.add_argument('--t', type = bool,   default = False,   help = 'To test the model on test data.')
parser.add_argument('--f', type = str,   default = None,   help = 'Model file to do testing')

parser.add_argument('--e', type = bool,   default = False,   help = 'To evaluate the model.')
parser.add_argument('--sf', type = str,   default = None,   help = 'Op: provide score file name')
parser.add_argument('--rf', type = str,   default = None,   help = 'Op: provide result file name')

'''----------------------------------------------------------------
'''
if __name__ == "__main__":
    args       = parser.parse_args()
    print('\n---------Printing all arguments:--------\n\
        {}\n----------------------------------------\n'.format(args))  
    '''------------------------------------------------------------
    Step 1: Prepare data tensors and language objects
    ------------------------------------------------------------'''
    if args.p:
        data = Process()
        data.data_processing()
    '''------------------------------------------------------------
    Step 2: Training, testing and evaluation
    ------------------------------------------------------------'''        
    
    if args.ts or args.tr: # training
        is_resume = args.tr
        epoch = args.er
        file = args.tf
        torch.cuda.empty_cache() 
        device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(device))
        train = Train(device)
        train.train_model(is_resume, epoch, file)
    
    if args.t:  # testing
        if args.f:
            file = args.f
        else:
            file = config.best_model
        
        torch.cuda.empty_cache() 
        device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(device))
        test = Test(device)
        test.test_model(file)

    if args.e: # rouge evaluation
        s_file = args.sf
        r_file = args.rf

        torch.cuda.empty_cache()
        device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device: {}'.format(device))
        test = Evaluate(device) # files to be pass here
        final = test.evaluate_summaries(config.s_summaries)

        # this gives size in kbs -- have to convert in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    memory = get_size(usage)

    print ('\n-------------------Memory and time usage:  {}.--------------------\n'.format(memory))
