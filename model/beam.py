#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Mar 19 13:42:38 2020

@author: fatimamh
"""

import model_config as config
import torch
from model.mask_utils import nopeak_mask
import torch.nn.functional as F
import math

def init_vars(device, src, model):

	SP_index = config.SP_index
	src_mask = (src != config.PAD_index).unsqueeze(-2)
	print('src_mask: {}'.format(src_mask.shape))
	e_output = model.encoder(src, src_mask)
	print('e_output: {}'.format(e_output.shape))

	outputs = torch.LongTensor([[SP_index]])
	outputs = outputs.to(device)
	print('outputs: {}'.format(outputs.shape))

	trg_mask = nopeak_mask(device, 1)
	print('trg_mask: {}'.format(trg_mask.shape))

	out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
	print('out: {}'.format(out.shape))
	out = F.softmax(out, dim=-1)
	print('Softmax out: {}'.format(out.shape))

	probs, ix = out[:, -1].data.topk(config.k)
	print('probs: {}'.format(probs.shape))
	print('ix: {}'.format(ix.shape))

	log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
	print('log_scores: {}'.format(log_scores.shape))

	outputs = torch.zeros(config.k, config.max_sum).long()
	print('outputs: {}'.format(outputs.shape))

	outputs = outputs.to(device)
	outputs[:, 0] = SP_index
	outputs[:, 1] = ix[0]
	print('outputs: {}'.format(outputs.shape))
	#print('outputs: {}'.format(outputs))

	e_outputs = torch.zeros(config.k, e_output.size(-2),e_output.size(-1))
	print('e_outputs: {}'.format(e_outputs.shape))
	
	e_outputs = e_outputs.to(device)
	e_outputs[:, :] = e_output[0]
	print('e_outputs: {}'.format(e_outputs.shape))
	#print('e_outputs: {}'.format(e_outputs))

	return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
	probs, ix = out[:, -1].data.topk(k)
	print('probs: {}'.format(probs.shape))
	print('ix: {}'.format(ix.shape))

	log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
	print('log_probs: {}'.format(log_probs.shape))

	k_probs, k_ix = log_probs.view(-1).topk(k)
	print('k_probs: {}'.format(k_probs.shape))
	print('k_ix: {}'.format(k_ix.shape))

	row = k_ix // k
	col = k_ix % k
	print('row: {}'.format(row))
	print('col: {}'.format(col))

	outputs[:, :i] = outputs[row, :i]
	outputs[:, i] = ix[row, col]
	print('outputs: {}'.format(outputs.shape))
	#print('outputs: {}'.format(outputs))

	log_scores = k_probs.unsqueeze(0)
	print('log_scores: {}'.format(log_scores.shape))

	return outputs, log_scores


def beam_search(device, src, model):
	#  to do init
	outputs, e_outputs, log_scores = init_vars(device, src, model)

	EP_index = config.EP_index
	src_mask = (src != config.PAD_index).unsqueeze(-2)
	print('src_mask: {}'.format(src_mask.shape))
	ind = None
	for i in range(2, config.max_sum):
		trg_mask = nopeak_mask(device, i)
		print('trg_mask: {}'.format(trg_mask.shape))

		out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask))
		print('out: {}'.format(out.shape))
		out = F.softmax(out, dim=-1)
		print('Softmax out: {}'.format(out.shape))
		
		outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, config.k)
		print('outputs: {}'.format(outputs.shape))
		print('log_scores: {}'.format(log_scores.shape))

		ones = (outputs==EP_index).nonzero() # Occurrences of end symbols for all input summaries.
		print('ones: {}'.format(ones.shape))

		summary_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
		print('summary_lengths: {}'.format(summary_lengths.shape))

		for vec in ones:
			i = vec[0]
			if summary_lengths[i]==0: # First end symbol has not been found yet
				summary_lengths[i] = vec[1] # Position of first end symbol

		num_finished_summaries = len([s for s in summary_lengths if s > 0])
		print('summary_lengths: {}'.format(num_finished_summaries))

		if num_finished_summaries == config.k:
			alpha = 0.7
			div = 1/(summary_lengths.type_as(log_scores)**alpha)
			_, ind = torch.max(log_scores * div, 1)
			ind = ind.data[0]
			break

	if ind is None:
		return outputs[0][1:]
	else:
		return outputs[ind][1:]

