#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:26:46 2016

@author: troyer
"""
from __future__ import division

from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

def count_words(txt):
    # Parse the txt and create a dictionary of word counts
    word_dict = defaultdict(int)
    with open(txt, 'r') as textfile:
        for line in textfile.readlines():
            clean_line = re.sub('[^a-zA-Z0-9_ ]', '', line)
            for word in clean_line.split():
                word_dict[word] += 1    
    return word_dict

def counts_dict_add_prop(dict_):
    n = sum(dict_.values())
    dd = defaultdict(list)
    for k, v in dict_.items():
        dd['word'].append(k)
        dd['count'].append(v)
        dd['proportion'].append(v/n)
        dd['length'].append(len(k))
        
    dataframe = pd.DataFrame(dd)
    return dataframe

def plot_word_length(path, dataframe, x_label='Length',
                                      y_label='Proportion',
                                      title='Word Length and Proportion'): 
    
    word_len_and_prop = dataframe.groupby('length').sum()
    plt.plot(word_len_and_prop.index, word_len_and_prop['proportion'],
             label='file: {}'.format(os.path.basename(path)))
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

def count_words_and_plot(txt):
    counted_words = count_words(txt)
    word_count_df = counts_dict_add_prop(counted_words)
    plot_word_length(txt, word_count_df)
    





