#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:40:29 2020

@author: fatimamh
"""

import os
import sys
import time
from datetime import datetime
from numpy import random

import torch
import torch.nn as nn
import torch.optim as optim
import model_config as config
'''---------------------------------------------------------
'''
'''----------------------------------------------------------------
'''
def get_time(st, et):
    
    diff = str('{}d:{}h:{}m:{}s'.\
           format(et.day-st.day,
           et.hour-st.hour,
           et.minute-st.minute,
           et.second-st.second))

    return diff

'''----------------------------------------------------------------
'''
# TO DO: VErify
def total_params(model):
    for parameter in model.parameters():
            print(parameter.size(), len(parameter)) 
            print()
'''----------------------------------------------------------------
'''
def trainable_params(model):     
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
        
    print('params: {}'.format(params))