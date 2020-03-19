import torch
import numpy as np
from torch.autograd import Variable

import model_config as config

def nopeak_mask(device, size):
	np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	#print('np_mask: {}'.format(np_mask))

	np_mask =  Variable(torch.from_numpy(np_mask) == 0)
	#print('np_mask: {}'.format(np_mask.shape))

	np_mask = np_mask.to(device)

	return np_mask

def create_masks(device, src, trg):
    
	src_mask = (src != config.PAD_index).unsqueeze(-2)
	print('src_mask: {}'.format(src_mask.shape))

	if trg is not None:
		trg_mask = (trg != config.PAD_index).unsqueeze(-2)
		print('trg_mask: {}'.format(trg_mask.shape))        
		size = trg.size(1) # get seq_len for matrix
		#print('trg_size: {}'.format(size))

		np_mask = nopeak_mask(device, size)

		np_mask = np_mask.to(device)
		trg_mask = trg_mask.to(device)
		trg_mask = trg_mask & np_mask

	else:
		trg_mask = None

	return src_mask, trg_mask