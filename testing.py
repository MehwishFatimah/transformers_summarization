#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:50:39 2019

@author: fatimamh
"""
import argparse
import sys
import os
from os import path
import time
import numpy as np
import pandas as pd
from csv import writer
import torch

from torch.autograd import Variable
from utils.loader_utils import get_data
from model.model import S2SModel
from utils.lang_utils import load_vocab
from utils.lang_utils import tensor_to_text

import model_config as config

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--f', type = str,   default = None,   help = 'To resume training, provide model')

'''==============================================================================
'''
class Test(object):
	def __init__(self, device):
		self.device = device
		self.model = S2SModel(device).to(self.device)
		self.vocab = load_vocab(config.sum_vocab_c)
		self.file  = config.s_summaries
	
	'''==============================================================================
	'''
	def load_model(self, file=None):
		print('-------------Start: load_model-------------')
		if file is not None and path.exists(file):
			state = torch.load(file, map_location= lambda storage, location: storage)
			self.model.encoder.load_state_dict(state["encoder_state_dict"])
			self.model.decoder.load_state_dict(state["decoder_state_dict"])
			self.model.reduce_state.load_state_dict(state["reduce_state_dict"])
			print('-------------End: load_model-------------')
	
	'''==============================================================================
	'''
	def build_row(self, *args):
		row =[]
		for i in range(len(args)):
			row.append(args[i])
		return row
	'''==============================================================================
	'''
	def write_csv(self, file, content):
		with open(file, 'a+', newline='') as obj:
			headers = ['reference', 'system']
			csv_writer = writer(obj)
			is_empty = os.stat(file).st_size == 0
			if is_empty:
				csv_writer.writerow(headers)
			csv_writer.writerow(content)

	'''==============================================================================
	'''
	def get_text(self, reference, system):
		
		reference = reference.tolist()
		reference = [int(i) for i in reference]
		reference = tensor_to_text(self.vocab, reference)
		print('reference: {}'.format(reference))
		print()
		#system = system.tolist()
		system  = [int(i) for i in system]
		system = tensor_to_text(self.vocab, system)	
		print('system: {}'.format(system))
		print()
		
		row = self.build_row(reference, system)
		print('row: {}\n'.format(row))
		self.write_csv(self.file, row)

	'''==============================================================================
	'''
	def create_mock_batch(self, tensor, lens):

		mock = Variable(torch.zeros((config.batch_size, tensor.size()[0], tensor.size()[1]), dtype =torch.long))
		mock = tensor.repeat(1, config.batch_size)
		mock = mock.transpose(0,1).unsqueeze(2)

		len_mock = [lens] * config.batch_size
		return mock, len_mock
	'''==============================================================================
	'''
	def test_a_batch(self, input_tensor, target_tensor, input_lens, output_lens):

		'''------------------------------------------------------------
		1: Setup tensors
		------------------------------------------------------------'''
		batch_size = input_tensor.size()[0]
		for idx in range(batch_size): # take 1 example from batch
			# create mock batch from single example
			input_mock, input_lens_mock = self.create_mock_batch(input_tensor[idx], input_lens[idx])
			input_lens_mock[0] = config.max_text # there was a problem due to less text length
			#print('input_mock: {}'.format(input_mock.shape))
			#print('input_len_mock: {}'.format(len(input_len_mock)))
			
			best_summary = self.model.beam_decode(input_mock, input_lens_mock)			
			
			self.get_text(target_tensor[idx].squeeze(), best_summary.tokens[1:])
	
	'''==============================================================================
	'''
	def test_model(self, file=None):
		'''-----------------------------------------------
		Step 1: Get model from file
		-----------------------------------------------'''
		# todo: best model
		if file is None:
			file = "train" + "_e_{}_".format(config.epochs) + "all" + ".pth.tar"
			folder = os.path.join(config.log_folder, 'train')
			if os.path.exists(folder):
				file = os.path.join(folder, file)
			else:
				print('Train dir or file {} doesn\'t exist'.format(folder))
				sys.exit()	
		else:
			file = file
		
		if os.path.exists(file):
			self.load_model(file)
		else:
			print('Train file doesn\'t exist... {}'.format(file))
			sys.exit()
		
		'''-----------------------------------------------
		Step 2: Get data_loaders
		-----------------------------------------------'''
		start = time.time()
		test_loader = get_data("test")
		self.model.eval()
		batch_idx = 1
		total_batch = len(test_loader)
		'''------------------------------------------------------------
		3: Get batches from loader and pass to evaluate             
		------------------------------------------------------------'''
		for input_tensor, target_tensor, input_lens, output_lens in test_loader:
			print('\t---Test Batch: {}/{}---'.format(batch_idx, total_batch))
			self.test_a_batch(input_tensor, target_tensor, input_lens, output_lens)
			batch_idx +=1
		
		return True
'''	
if __name__ == '__main__':
	args = parser.parse_args()
	file = args.f 
	#print (os.getcwd())
	torch.cuda.empty_cache()
	device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device: {}'.format(device))
	test = Test(device)
	final = test.test_model(file)
	print(final)
'''
