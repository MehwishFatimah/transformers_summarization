#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:15:48 2019
Updated: 02 Dec 19
@author: fatimamh
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import os
from os import path
from rouge import Rouge
import model_config as config

class Evaluate(object):
    def __init__(self, device, score_file = config.scores, result_file = config.results):
        self.device = device
        self.df = None
        self.scores = score_file# outfile
        self.results = result_file # only scores
    '''----------------------------------------------------------------
    '''
    def get_scores(self):
        references = self.df['reference'].tolist()
        systems = self.df['system'].tolist()
        scores = []
        rouge = Rouge()
        for i in range(len(systems)):
            score = rouge.get_scores(systems[i], references[i])
            scores.append(score)
        
        self.df['score'] = scores
        self.df.to_csv(self.scores)
    '''----------------------------------------------------------------
    '''
    def get_average(self, x):
        return sum(x)/len(x)

    '''----------------------------------------------------------------
    '''
    def get_results(self):

        r1_f, r1_p, r1_r = ([] for _ in range(3))
        r2_f, r2_p, r2_r = ([] for _ in range(3))
        rl_f, rl_p, rl_r = ([] for _ in range(3))

        scores = self.df['score']
        for i in range(len(scores)):
            score = str(scores[i])
            score = score.replace('[', '')
            score = score.replace(']', '')
            score = score.replace("\'", "\"")
            dic = json.loads(score)
            #print(dic.keys())
            #print(dic.values())
            r1_f.append(dic['rouge-1']['f'])
            r1_p.append(dic['rouge-1']['p'])
            r1_r.append(dic['rouge-1']['r'])

            r2_f.append(dic['rouge-2']['f'])
            r2_p.append(dic['rouge-2']['p'])
            r2_r.append(dic['rouge-2']['r'])

            rl_f.append(dic['rouge-l']['f'])
            rl_p.append(dic['rouge-l']['p'])
            rl_r.append(dic['rouge-l']['r'])

        r1_f.append(self.get_average(r1_f))
        r1_p.append(self.get_average(r1_p))
        r1_r.append(self.get_average(r1_r))

        r2_f.append(self.get_average(r2_f))
        r2_p.append(self.get_average(r2_p))
        r2_r.append(self.get_average(r2_r))

        rl_f.append(self.get_average(rl_f))
        rl_p.append(self.get_average(rl_p))
        rl_r.append(self.get_average(rl_r))

        
        rouge = pd.DataFrame(np.column_stack([r1_f, r1_p, r1_r, r2_f, r2_p, r2_r, rl_f, rl_p, rl_r]), 
                             columns=['r1_f', 'r1_p', 'r1_r', 'r2_f', 'r2_p', 'r2_r', 'rl_f', 'rl_p', 'rl_r'])
        print(len(rouge), rouge.columns, rouge.head(5))
        rouge.to_csv(self.results)

    '''----------------------------------------------------------------
    '''
    def evaluate_summaries(self, file):
        
        if path.exists(file):
            self.df = pd.read_csv(file)
            print(self.df.head())
            self.get_scores()      
            self.get_results()

        return True

'''----------------------------------------------------------------
if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    test = Evaluate(device)
    final = test.evaluate_summaries(config.s_summaries)
    print(final)
'''
