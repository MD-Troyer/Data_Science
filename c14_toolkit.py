#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
#
# FRONT MATTER ----------------------------------------------------------------
#
###############################################################################

"""
Author:
    Michael D. Troyer

Date:
    Start: 20161118
    End: 

Purpose:
    A suite of radiocarbon based visualizations and statistical and other
    numeric analysis. fuck yeah.

Comments:
    credit is due to Joel Grus and his awesome book _Data Science from Scratch_
    for examples and inspiration
    
TODO:
    chi-squared cdf - write own
    make a better comparative nomral writer - accecpt mu_1, sig_1 ...
    z-test for 14c with stErr
    
"""

###############################################################################
#
# IMPORTS ---------------------------------------------------------------------
#
###############################################################################

from __future__ import division  # Integer division is lame - use // instead

#from collections import Counter
from collections import defaultdict
from scipy import stats
#import copy
#import csv
import datetime
import getpass
import math
import matplotlib.pyplot as plt
#import numpy as np
import operator
import os
#import pandas as pd
import re
#import scipy
import sys
import textwrap
import traceback

#pylab

###############################################################################
#
# GLOBALS ---------------------------------------------------------------------
#
###############################################################################

###############################################################################
# Variables
###############################################################################

filename = os.path.basename(__file__)

start_time = datetime.datetime.now()

user = getpass.getuser()

header = ('*'*100) # a text header for all the various print functions

working_dir = r''

###############################################################################
# Classes
###############################################################################

class py_log(object):
    """A custom logging class that simultaneously writes to the console,
       an optional logfile, and/or a production report. The methods provide
       three means of observing the tool behavior 1.) console progress updates
       during execution, 2.) tool metadata regarding date/user/inputs/outputs..
       and 3.) an optional logfile where the tool will print messages and
       unpack variables for further inspection"""

    def __init__(self, report_path, log_path, log_active=True):
        self.report_path = report_path
        self.log_path = log_path
        self.log_active = log_active

    def _write_arg(self, arg, path, starting_level=0):
        """Accepts a [path] txt from open(path)
           and unpacks that data like a baller!"""
        level = starting_level
        txtfile = open(path, 'a')
        if level == 0:
            txtfile.write(header)
        if type(arg) == dict:
            txtfile.write("\n"+(level*"\t")+(str(arg))+"\n")
            txtfile.write((level*"\t")+str(type(arg))+"\n")
            for k, v in arg.items():
                txtfile = open(path, 'a')
                txtfile.write('\n'+(level*"\t\t")+(str(k))+": "+(str(v))+"\n")
                if hasattr(v, '__iter__'):
                    txtfile.write((level*"\t\t")+"Values:"+"\n")
                    txtfile.close()
                    for val in v:
                        self._write_arg(val, path, starting_level=level+2)
        else:
            txtfile.write("\n"+(level*"\t")+(str(arg))+"\n")
            txtfile.write((level*"\t")+str(type(arg))+"\n")
            if hasattr(arg, '__iter__'):  # Does not include strings
                txtfile.write((level*"\t")+"Iterables:"+"\n")
                txtfile.close()
                for a in arg:
                    self._write_arg(a, path, starting_level=level+1)
        txtfile.close()
        
    def _writer(self, msg, path, *args):
        """A writer to write the msg, and unpacked variable"""
        with open(path, 'a') as txtfile:
            txtfile.write(msg+"\n")
            txtfile.close()
            if args:
                for arg in args:
                    self._write_arg(arg, path)

    def console(self, msg):
        """Print to console only - progress reports"""
        print(msg)  # Optionally - arcpy.AddMessage()

    def report(self, msg):
        """Write to report only - tool process metadata for the user"""
        path_rep = self.report_path
        self._writer(msg, path_rep)

    def logfile(self, msg, *args):
        """Write to logfile only - use for reporting debugging data
           With an optional shut-off"""
        if self.log_active:
            path_log = self.log_path
            self._writer(msg, path_log, *args)
   
    def logging(self, log_level, msg, *args): 
        assert log_level in [1,2,3], "Incorrect log level"
        if log_level == 1: # Updates - Console, report, and logfile:
            self.console(msg)
            self.report(msg)
            self.logfile(msg, *args)
        if log_level == 2:  # Operational metadata - report and logfile
            self.report(msg)
            self.logfile(msg, *args)
        if log_level == 3:  # Debugging - logfile only
            self.logfile(msg, *args)
            
###############################################################################
# Functions
###############################################################################

def print_exception_full_stack(print_locals=True):
    """Print full stack in a more orderly way
       Optionally print the exception frame local variables"""
    exc = sys.exc_info()  # 3-tuple (type, value, traceback)
    if exc is None:
        return None

    tb_type, tb_value, tb_obj = exc[0], exc[1], exc[2]
    exc_type = str(tb_type).split(".")[1].replace("'>", '')
    lg.logging(1, '\n\n'+header+'\n'+header)
    lg.logging(1,'\nEXCEPTION:\n{}\n{}\n'.format(exc_type, tb_value))
    lg.logging(1, header+'\n'+header+'\n\n')
    lg.logging(1, 'Traceback (most recent call last):')

    # 4-tuple (filename, line no, func name, text)
    tb = traceback.extract_tb(exc[2])
    for tb_ in tb:
        lg.logging(1, "{}\n"
                   "Filename: {}\n"
                   "Line Number: {}\n"
                   "Function Name: {}\n"
                   "Text: {}\n"
                   "Exception: {}"
                   "".format(header, tb_[0], tb_[1], tb_[2],
                             textwrap.fill(tb_[3]), exc[1]))
    if print_locals:
        stack = []
        while tb_obj.tb_next:
            tb_obj = tb_obj.tb_next  # Make sure at end of stack
        f = tb_obj.tb_frame          # Get the frame object(s)

        while f:                     # Append and rewind, reverse order
            stack.append(f)
            f = f.f_back
        stack.reverse()

        lg.logging(3, '\n\nFrames and locals (innermost last):\n'+header)
        for frame in stack:
            if str(frame.f_code.co_filename).endswith(filename):
                lg.logging(3, "{}\n"
                           "FRAME {} IN:\n"
                           "{}\n"
                           "LINE: {}\n"
                           "".format(header,
                                     textwrap.fill(frame.f_code.co_name),
                                     textwrap.fill(frame.f_code.co_filename),
                                     frame.f_lineno))

                if not frame.f_locals.items():
                    lg.logging(3, "No locals\n")

                else:
                    lg.logging(3, "{} LOCALS:\n".format(frame.f_code.co_name))
                    for key, value in sorted(frame.f_locals.items()):
                        # Exclude private and the i/o and header parameters
                        if not str(key).startswith("_"):
                            if not str(key) in ['In', 'Out', 'header']:
                                lg.logging(3, (str(key)+":").strip())
                                
                                try:
                                    lg.logging(3, str(value).strip()+'\n')
                                except:
                                    lg.logging(3, 'Error writing value')
    return

###############################################################################
# Settings
###############################################################################


###############################################################################
#
# EXECUTION -------------------------------------------------------------------
#
###############################################################################

try:
    # Make a directory - or use woring_dir from above
    path = working_dir if working_dir else os.path.join(os.getcwd(), 'Test')
    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    folder_path = os.path.join(dir_name, base_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    date_split = str(datetime.datetime.now()).split('.')[0]
    date_time_stamp = re.sub('[^0-9]', '', date_split)
    name_stamp = filename.split('.')[0]+"_"+date_time_stamp

    # Create the logger
    report_path = os.path.join(folder_path, name_stamp+"_Report.txt")
    logfile_path = os.path.join(folder_path, name_stamp+"_Logfile.txt")
    lg = py_log(report_path, logfile_path)

###
#    lg.log_active = False # Uncomment to disable logfile
###

    # Start logging
    lg.logging(1, "\nExecuting: "+filename+' \nDate: '+date_split)
    lg.logging(2, header)
    lg.logging(2, "Running environment: Python - {}".format(sys.version))
    lg.logging(1, "User: "+user)
    lg.logging(1, "\nLogging to: "+folder_path)

###############################################################################
#
# MAIN PROGRAM ----------------------------------------------------------------
#
###############################################################################

###
#    assert 1/0, "TestTrap" 
###

    # Define the data

    # T-W Diamond Dataset
    twdiamond_c14 = {"Beta-6848": {'rcybp':   860, 'sigma':  100},
                     "Beta-6849": {'rcybp':   930, 'sigma':  230},
                     "Aeon-1272": {'rcybp':   920, 'sigma':   80},
                     "Aeon-1273": {'rcybp':   780, 'sigma':  220},
                     "Aeon-1274": {'rcybp':  1500, 'sigma':  340},
                     "Aeon-2152": {'rcybp':   715, 'sigma':   25},
                     "Aeon-2153": {'rcybp':   750, 'sigma':   30}}
    
    # Killdeer dataset
    killdeer_c14 =  {"Beta-5129": {'rcybp':   360, 'sigma':   80},
                     "Beta-5127": {'rcybp':   150, 'sigma':   50},
                     "Beta-5130": {'rcybp':   260, 'sigma':   50},
                     "Beta-5128": {'rcybp':   300, 'sigma':   90},
                     "Beta-5131": {'rcybp':   170, 'sigma':   50},
                     "Aeon-2150": {'rcybp':   225, 'sigma':   25},
                     "Aeon-2151": {'rcybp':   230, 'sigma':   25}}
    
    ### Define the functions
    
    def dot(v, w):
        """v_1 * w_1 + ... + v_n * w_n"""
        return sum(v_i * w_i for v_i, w_i in zip(v, w))
    
    def sum_of_squares(v):
        """v_1 * v_1 + ... + v_n * v_n"""
        return dot(v, v)
    
    # this isn't right if you don't from __future__ import division
    def mean(x):
        return sum(x) / len(x)
    
    def de_mean(x):
        """translate x by subtracting its mean (so the result has mean 0)"""
        x_bar = mean(x)
        return [x_i - x_bar for x_i in x]
    
    def variance(x):
        """assumes x has at least two elements"""
        n = len(x)
        deviations = de_mean(x)
        return sum_of_squares(deviations) / (n - 1)
    
    def standard_deviation(x):
        return math.sqrt(variance(x))
            
    def normal_pdf(x, mu=0, sigma=1):
        """the normal probabiity density function"""
        sqrt_two_pi = math.sqrt(2 * math.pi)
        return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / 
                (sqrt_two_pi * sigma))
    
    def normal_cdf(x, mu=0,sigma=1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

    def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
        """find approximate inverse using binary search"""
    
        # if not standard, compute standard and rescale
        if mu != 0 or sigma != 1:
            return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
        
        low_z, low_p = -10.0, 0          # normal_cdf(-10) is (very close to) 0
        hi_z,  hi_p  =  10.0, 1          # normal_cdf(10)  is (very close to) 1
        while hi_z - low_z > tolerance:
            mid_z = (low_z + hi_z) / 2   # consider the midpoint
            mid_p = normal_cdf(mid_z)    # and the cdf's value there
            if mid_p < p:
                # midpoint is still too low, search above it
                low_z, low_p = mid_z, mid_p
            elif mid_p > p:
                # midpoint is still too high, search below it
                hi_z, hi_p = mid_z, mid_p
            else:
                break
        
    # the normal cdf _is_ the probability the variable is below a threshold
    normal_probability_below = normal_cdf
    
    # it's above the threshold if it's not below the threshold
    def normal_probability_above(lo, mu=0, sigma=1):
        return 1 - normal_cdf(lo, mu, sigma)
        
    # it's between if it's less than hi, but not less than lo
    def normal_probability_between(lo, hi, mu=0, sigma=1):
        return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)
    
    # it's outside if it's not between
    def normal_probability_outside(lo, hi, mu=0, sigma=1):
        return 1 - normal_probability_between(lo, hi, mu, sigma) 
    
    def normal_upper_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z <= z) = probability"""
        return inverse_normal_cdf(probability, mu, sigma)
        
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)
    
    def normal_two_sided_bounds(probability, mu=0, sigma=1):
        """returns the symmetric (about the mean) bounds 
        that contain the specified probability"""
        tail_probability = (1 - probability) / 2
    
        # upper bound should have tail_probability above it
        upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    
        # lower bound should have tail_probability below it
        lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    
        return lower_bound, upper_bound
    
    def two_sided_p_value(x, mu=0, sigma=1):
        if x >= mu:
            # if x is greater than the mean, the tail is above x
            return 2 * normal_probability_above(x, mu, sigma)
        else:
            # if x is less than the mean, the tail is below x
            return 2 * normal_probability_below(x, mu, sigma) 
            
    def plot_normal_pdf(mu=0, sigma=1, lbl="Sample", x_lower=-50, x_upper=50):
        """plot function normal_pdf within +- 5 Z"""
        xs = [(x/10) for x in range(10*x_lower, 10*x_upper)]
        plt.plot(xs,[normal_pdf(x, mu, sigma) for x in xs],
                 label=('{}: {} +/-{}'.format(lbl, mu, sigma)))
        return
    
    def plot_normal_dfs(mu=0, sigma=1, plot_type='pdf'):
        if not plot_type in ['pdf', 'cdf']:
            return "Invalid plot_type - use 'pdf' or 'cdf'"
    
        lower_x = int(mu-(5*sigma))
        upper_x = int(mu+(5*sigma))
        xs = [x for x in range(lower_x, upper_x)]
        if plot_type == 'pdf':
            plt.plot(xs,[normal_pdf(x, mu, sigma) for x in xs],
                         '-', label='mu={},sigma={}'.format(mu, sigma))
    
        if plot_type =='cdf':
            plt.plot(xs,[normal_cdf(x, mu, sigma) for x in xs],
                         '-', label='mu={},sigma={}'.format(mu, sigma))
        return
    
    def plot_14c_pdfs(date_dict=twdiamond_c14, plot_type='plot', title_=None):
        """Plot 14c - as single plot or stacked subplots"""
        # Highest at the peak
        y_upper = max(normal_pdf(c14['rcybp'], c14['rcybp'], c14['sigma'])
                        for c14 in date_dict.values())
        
        x_lower = min([date['rcybp']-5*date['sigma']
                        for date in date_dict.values()])
        
        x_upper = max([date['rcybp']+5*date['sigma']
                        for date in date_dict.values()])
    
        if plot_type.lower() == 'plot':
            for sample, c14 in sorted(date_dict.items(),
                                      key=operator.itemgetter(1)):
                
                plot_normal_pdf(c14['rcybp'], c14['sigma'],
                                sample, x_lower, x_upper)
                
            plt.xlim(xmin=x_lower, xmax=x_upper)
            plt.ylim(ymin=0, ymax=(y_upper*1.1))
            plt.xlabel('Radiocarbon Years BP')
            plt.ylabel('Probability')
            plt.legend()
            if title_:
                plt.title(str(title_))
            plt.show()
            return
    
        elif not plot_type.lower() == 'subplot':
            return "Incorrect plot type - use 'plot' / 'subplot'"
    
        else:
            nrows = str(len(killdeer_c14.items()))
            row_count = 1
            for sample, c14 in sorted(date_dict.items(),
                                      key=operator.itemgetter(1)):
                
                plt.subplot(nrows, 1, row_count)
                plot_normal_pdf(c14['rcybp'], c14['sigma'],
                                sample, x_lower, x_upper)
                
                plt.legend()
                cur_axes = plt.gca()
                #cur_axes.axes.get_xaxis().set_ticklabels([])
                cur_axes.axes.get_yaxis().set_ticklabels([])
                #cur_axes.axes.get_xaxis().set_ticks([])
                cur_axes.axes.get_yaxis().set_ticks([])
                row_count += 1
                plt.xlim(xmin=x_lower, xmax=x_upper)
                plt.ylim(ymin=0, ymax=(y_upper*1.4))
            plt.xlabel('Radiocarbon Years BP')
            plt.show()
        return
    
    def f_test(mu_1, sigma_1, mu_2, sigma_2):
        """Performs the standard F-test for homogeneous variances"""
        max_sigma = max(sigma_1**2, sigma_2**2)
        min_sigma = min(sigma_1**2, sigma_2**2)
        print 'f-test: '+str(max_sigma/min_sigma)
        return max_sigma/min_sigma
    
    def cohens_d(mu_1, sigma_1, mu_2, sigma_2):
        """Performs Cohen's d for effect size for difference between two means
           Assumes very large homogeneous sample sizes and doesn't weight 
           pooled sigmas"""
        cohens_d =  abs(mu_1 - mu_2)/math.sqrt((sigma_1**2 + sigma_2**2) / 2)
        if cohens_d <= 0.01:
            effect_size = 'Very Small'
        elif cohens_d <= 0.2:
            effect_size = 'Small'
        elif cohens_d <= 0.5:
            effect_size = 'Medium'
        elif cohens_d <= 0.8:
            effect_size = 'Large'
        else:
            effect_size = 'Very Large'
    
        print "Cohen's d: {}\nEffect Size: {}".format(cohens_d, effect_size)
        return (cohens_d, effect_size)

    def chi_square(array_1, array_2):  # Make sure these are sorted!
        return sum([((o_n - e_n) ** 2) / e_n
                      for o_n, e_n in zip(array_1, array_2)])
    
    def chi_distribution(x, df):
        return 1 / (2 * math.gamma(df/2)) * (x/2)**(df/2-1) * math.exp(-x/2)
        
    def chi_sq_test_14c(pdf_1, mu_1, sig_1, pdf_2, mu_2, sig_2, num_bins=None):
        """Pearson's chi-squared goodness-of-fit test for 14c Dates"""
        num_bins = num_bins if num_bins else 10
        dof = num_bins-1
        x_lower = min(mu_1 - 5.0 * sig_1, mu_2 - 5.0 * sig_2)
        x_upper = max(mu_1 + 5.0 * sig_1, mu_2 + 5.0 * sig_2)
        bin_size = (x_upper - x_lower)/num_bins
        chi_bins = [bin_size * i + x_lower for i in range(0, num_bins)]
        sample_interval = bin_size / 10.0  # Can change interval here
        pdf_1_dd = defaultdict(int)
        pdf_2_dd = defaultdict(int)
        
        # Sample each function within each bin 9 times and return the mean
        for bin_ in chi_bins:
            tst_pts = [float(sample_interval * i + bin_) for i in range(1, 10)]         
            
            pdf_1_dd[bin_] = mean([pdf_1(test_x, mu_1, sig_1) 
                                        for test_x in tst_pts])
            
            pdf_2_dd[bin_] = mean([pdf_2(test_x, mu_2, sig_2) 
                                        for test_x in tst_pts])
        
        pdf_1_array = [item[1] for item in sorted(pdf_1_dd.items())]
        pdf_2_array = [item[1] for item in sorted(pdf_2_dd.items())]
        
        chi_square_score = chi_square(pdf_1_array, pdf_2_array)
        p_value = stats.chi2.cdf(chi_square_score, dof)
        
        print "chi-squared: {}\np-value: {}\ndegrees of freedom: {}"\
                "".format(chi_square_score, p_value, dof)
        return (chi_square_score, p_value, dof)
     
    
    m1 = 1000
    s1 =  80
    m2 = 910
    s2 =  70
    f_test(m1, s1, m2, s2)
    cohens_d(m1, s1, m2, s2)
    chi_sq_test_14c(normal_pdf, m1, s1, normal_pdf, m2, s2)
    plot_14c_pdfs({'1':{'rcybp':m1, 'sigma': s1}, '2': {'rcybp': m2, 'sigma': s2}})
    lg.logging(1, '\n\nAll functions successfully loaded..\n')

###############################################################################
#
# EXCEPTIONS ------------------------------------------------------------------
#
###############################################################################

except:
    print_exception_full_stack(print_locals=True)  # Or print_locals=False

    # Don't create exceptions in the except block!
    try:
        lg.logging(1, '\n\n{} did not sucessfully complete'.format(filename))
        lg.console('See logfile for details')
        
    except:
        pass

###############################################################################
#
# CLEAN-UP --------------------------------------------------------------------
#
###############################################################################

finally:
    end_time = datetime.datetime.now()
    
    try:
        lg.logging(1, "End Time: "+str(end_time))
        lg.logging(1, "Time Elapsed: {}".format(str(end_time - start_time)))
        
    except:
        pass

###############################################################################
# if this .py has been called by interpreter directly and not by another module
# __name__ == "__main__":  #will be True, else name of importing module
if __name__ == "__main__":
    lg.logging(1, "{} was called directly"
               "".format(filename))
    