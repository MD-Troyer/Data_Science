#!/usr/bin/env python
# -*- coding: utf-8 -*-


#######################################################################################################################
#
#front matter
#
#######################################################################################################################

"""
Author: Michael D. Troyer

Date:

Purpose:

Comments:
"""


#######################################################################################################################
#
#imports
#
#######################################################################################################################

import datetime, logging, os, re, sys, traceback
#import csv, math
#import arcpy as ap
import numpy as np
#import pandas as pd
#import scipy as sp
#import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c

#from collections import Counter, defaultdict

#pylab


#######################################################################################################################
#
#global variables
#
#######################################################################################################################

date_time = (str(datetime.datetime.now())).split('.')[0]
date_time_stamp = re.sub('[^0-9]', '_', date_time)
filename = os.path.basename(__file__)


#######################################################################################################################
#
#global functions
#
#######################################################################################################################



#######################################################################################################################
#
#initialize logging
#
#######################################################################################################################

#output_process_log = 'logfile_'+filename[:-3]+'_'+date_time_stamp+'.txt'
#
#logging.basicConfig(filename = output_process_log, level=logging.DEBUG, format = '%(message)s')
#
##create the log file header
#logging.debug('LOG FILE:')
#logging.debug('_'*120+'\n\n')
#logging.debug('Start date: '+date_time+'\n')
#logging.debug('Filename: '+filename+'\n')
#logging.debug('Working directory:\n '+os.getcwd()+'\n')
#logging.debug('Running python from:\n '+sys.executable+'\n')
#logging.debug('System Info:\nPython version: '+sys.version+'\n\n')
#logging.debug("Execution:")
#logging.debug('_'*120+'\n')

#uncomment for production code
#logging.disable(logging.CRITICAL)

#use for logging variables
#logging.debug(    'DEBUG:     x = {} and is {}'.format(repr(x), str(type(x))))
#logging.info(     'INFO:      x = {} and is {}'.format(repr(x), str(type(x))))
#logging.warning(  'WARNING:   x = {} and is {}'.format(repr(x), str(type(x))))
#logging.error(    'ERROR:     x = {} and is {}'.format(repr(x), str(type(x))))
#logging.critical( 'CRITICAL:  x = {} and is {}'.format(repr(x), str(type(x))))

#use assert sanity checks
#assert [condition], 'assertionError str'


#######################################################################################################################
#
#execution
#
#######################################################################################################################

#
#        try:
#            pass
iris = datasets.load_iris()
x = iris.data
y = iris.target

x = x[y != 2]
y = y[y != 2]


#        except:
#            errorfile = open(output_process_log, 'a')
#            errorfile.write('\n\n')
#            errorfile.write('Exceptions:\n')
#            errorfile.write('_'*80+'\n\n')
#            errorfile.write(traceback.format_exc())
#            errorfile.close()
#            print '\n'+str((traceback.format_exc()).split('\n')[-2])
#            print '\ntraceback written to:\n{}'.format(output_process_log)


#######################################################################################################################
#if this .py has been called by interpreter directly and not by another module
#__name__ == "__main__":    #will be True, else name of importing module
