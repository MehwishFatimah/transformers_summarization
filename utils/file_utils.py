#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:08:56 2019
FINALIZED
@author: fatimamh
"""

import os
import chardet
import warnings
import pickle
from csv import writer

'''-----------------------------------------------------------------------
Takes file path with file name, reads it as bite returns the content.
  Args:
    file     : str
  Returns:
    content  : str
'''

def read_content(file):

    with open(file,'rb') as rb:
        content = rb.read()

    code = chardet.detect(content)['encoding']
    #print(code)
    try:
        content = content.decode(code)
    except:
        message = 'This file code {} has some error, and ignored.'.format(code)
        warnings.warn(message)
        content = content.decode(code, 'ignore')

    return content

'''-----------------------------------------------------------------------
Takes folder path reads all files and their contents, returns the content list.
  Args:
    folder          : str
  Returns:
    total_content   : list
'''
def read_folder_content(folder):

    total_content = []
    list_files = os.listdir(folder)

    for f in list_files:
        content = read_content(file =folder + f)
        total_content.append(content)

    return total_content

'''-----------------------------------------------------------------------
Takes folder path reads all files with specific extension, returns the list of all files.
  Args:
    folder          : str
  Returns:
    files           : list
'''
def get_files(folder, ext=".pt"):

    files = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            files.append(file)
    return files
'''-----------------------------------------------------------------
'''
def print_writer(file, content):
    with open(file, 'a+', newline='') as obj:
        headers = ['epoch', 'train_loss', 'eval_loss', 'eval_best_loss', 'time']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)
'''-----------------------------------------------------------------
'''
def plot_writer(file, content):
    with open(file, 'a+', newline='') as obj:
        headers = ['epoch', 'train_loss', 'eval_loss']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)
'''-----------------------------------------------------------------
'''
def build_row(*args):
    row =[]
    for i in range(len(args)):
        row.append(args[i])
    return row