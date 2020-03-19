import os

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

json_files  = ["en_train_sub.json","en_val_sub.json","en_test_sub.json"]
csv_files   = ["en_train_sub.csv", "en_val_sub.csv", "en_test_sub.csv"]

max_text	= 400 
max_sum	  	= 100
text_vocab	= 30000 # exclude 4 special in count
sum_vocab	= 10000 

PAD_index	= 0
UNK_index 	= 1
SP_index  	= 2
EP_index  	= 3
