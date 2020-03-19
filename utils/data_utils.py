#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:48:19 2019
Modified on Wed Nov 06
Modified on Wed Feb 12

@author: fatimamh

"""
import argparse
import os
import re
import pandas as pd
import time
import resource

class Data(object):
	def __init__(self):
		
		self.clean_text = True
		self.short_text = True
		self.tex_len = None
		self.sum_len = None

	'''---------------------------------------------'''
	def replace_tags_a(self, text):

		text = text.replace('<ARTICLE>', ' ')
		text = text.replace('</ARTICLE>', ' ')
		text = text.replace('<TITLE>', ' ')
		text = text.replace('</TITLE>', ' ')
		text = text.replace('<HEADING>', ' ')
		text = text.replace('</HEADING>', ' ')
		text = text.replace('<SECTION>', ' ')
		text = text.replace('</SECTION>', ' ')
		text = text.replace('<S>', ' ')
		text = text.replace('</S>', ' ')
		text = text.replace('\n', ' ')
		text = re.sub('\s+', ' ', text)
		return text

	'''---------------------------------------------'''
	def replace_tags_s(self, text):

		text = text.replace('<SUMMARY>', ' ')
		text = text.replace('</SUMMARY>', ' ')
		text = text.replace('<S>', ' ')
		text = text.replace('</S>', ' ')
		text = text.replace('\n', ' ')
		text = re.sub('\s+', ' ', text)
		return text

	'''---------------------------------------------'''
	def make_text_short(self, text, length):
		
		text = text.split()
		#print(len(text))
		short_text = text[0]
		if len(text) >= length:
			end = length-1 # 1 token less for adding EP token
		else:
			end = len(text)
		#print(end)
		for i in range(1, end):
			short_text = short_text + ' ' + text[i]
		return short_text

	'''---------------------------------------------'''
	def clean_data(self, df):

		#print(clean_text, short_text, t_len, s_len)
		print('Data before cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))

		if self.clean_text:
			print('cleaning text')
			df['text']    = df['text'].apply(lambda x: self.replace_tags_a(x))
			df['summary'] = df['summary'].apply(lambda x: self.replace_tags_s(x))

		if 'index' in df.columns:
			del df['index']

		if self.short_text:
			print('shortening text')
			df['text']    = df['text'].apply(lambda x: self.make_text_short(x, self.tex_len))
			df['summary'] = df['summary'].apply(lambda x: self.make_text_short(x, self.sum_len))

		print('Data after cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
		return df

	'''---------------------------------------------'''
	def process_data(self, config, ext = '.csv'):
		files = config.json_files
		folder = config.root_in 
		out_folder = config.root_in

		self.tex_len = config.max_text 
		self.sum_len = config.max_sum 

		for file in files:
			file_name  = os.path.splitext(file)[0]
			print(file_name)
			file = os.path.join(folder, file)
			df   = pd.read_json(file, encoding = 'utf-8')
			df   = self.clean_data(df)
			print('\n--------------------------------------------')

			file = os.path.join(out_folder, file_name + ext)
			print(file)
			df.to_csv(file, index = False)
			print('\n======================================================================================')

