#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:05:55 2020

@author: fatimamh
"""
import time
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from random import seed
from random import randint
import pandas as pd
#from model.evaluation import evaluate


'''
====================================================================
11. Plot functions
====================================================================
'''
def showPlot(folder, print_file): 
    
    df = pd.read_csv(print_file) 
    print(df.head())

    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.AutoLocator()#MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    #ax.xaxis.set_major_locator(loc)
    
    x = df['epoch']
    y = df['train_loss']
    z = df['eval_loss']
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, y, color='blue', linewidth=2, label='Train_loss')
    plt.plot(x, z, color='red', linewidth=2, label='Eval_loss')
    
    plt.legend(loc='best')
    #plt.plot(points)
    #seed(time.time())
    #print('seed: {}'.format(seed))
    #v1 = randint(1,100)
    #v2 = randint(1,100)
    file = os.path.splitext(print_file)[0] + '.png'
    #file = 'plot_loss_' + str(v1) + str(v2) + '.png'
    #file = os.path.join(folder, file)
    plt.savefig(file)
    print(file)

