#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:06:36 2019

@author: fatimamh
"""

import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

'''----------------------------------------------------------------
'''
def model_param(model, config):
    # Optim connect it with model
    enc_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                                 lr = config["learning_rate"], 
                                 momentum = config["momentum"])
    dec_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                                 lr = config["learning_rate"], 
                                 momentum = config["momentum"]) 
    #enc_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                              lr=learning_rate)
    #dec_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                              lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index = config["PAD_index"])
    
    # gradient clip
    #clip = config["grad_clip"]

    return enc_optimizer, dec_optimizer, criterion#, clip

