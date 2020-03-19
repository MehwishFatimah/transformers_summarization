import os

'''----------------------------------------------------'''
#input setup
root_in 	 = "/hits/basement/nlp/fatimamh/test_data"
train_folder = os.path.join(root_in, "train")
val_folder 	 = os.path.join(root_in, "val")
test_folder  = os.path.join(root_in, "test")

#vocab setup
vocab_folder = os.path.join(root_in, "dict")
text_vocab_f = os.path.join(vocab_folder, "en_text_vocab_f.json") #full
sum_vocab_f	 = os.path.join(vocab_folder, "en_sum_vocab_f.json") #full
text_vocab_c = os.path.join(vocab_folder, "en_text_vocab_c.json") #condense
sum_vocab_c	 = os.path.join(vocab_folder, "en_sum_vocab_c.json") #condense

'''----------------------------------------------------'''
#out_folder 
root_out 	= "/hits/basement/nlp/fatimamh/test_data"
#"check_point_folder"
log_folder 	= os.path.join(root_out, "log")
best_model 	= os.path.join(log_folder, "best_model.pth.tar")
check_point = "/hits/basement/nlp/fatimamh/test_data/log/train/model_ep_2_b_5.pth.tar"

#output setup
out_folder  = os.path.join(root_out, "out")
s_summaries = os.path.join(out_folder, "test_summaries.csv") #system summaries
scores 	    = os.path.join(out_folder, "summaries_with_scores.csv")
results     = os.path.join(out_folder, "scores.csv")
epoch_loss  = os.path.join(out_folder, "epoch_loss.csv")

'''----------------------------------------------------'''
#data setting
train_docs	= 64
val_docs	= 16
test_docs	= 16
max_text	= 400
max_sum	  	= 100
text_vocab	= 30004 # include 4 special in count
sum_vocab	= 10004 
PAD_index	= 0
UNK_index 	= 1
SP_index  	= 2
EP_index  	= 3

'''----------------------------------------------------'''
# Hyperparameters
emb_dim	 	= 100
hid_dim	 	= 100
batch_size  	= 16 # 8 
beam_size	= 16 # for BATCH SIZE8 it is 4
k 		= 3 # beam size in new code
print_every	= 1
plot_every  	= 2

epochs	 	= 10
d_model 	= 512
n_layers	= 6  
n_heads		= 8
dropout		= 0.1
lr 		= 0.0001
SGDR		= True
load_weights	= None

