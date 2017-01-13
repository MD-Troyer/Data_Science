#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:
Michael D. Troyer

Date:
12/31/16

Purpose:
Model the sklearn breast cancer dataset

Comments:
Nailed it!
"""

import data_science_tools as dst
from itertools import combinations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts

# Get the data
cancer = load_breast_cancer()

# Create a dataframe
cancer_data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_data_df['target'] = cancer.target

features = list(cancer_data_df.columns)
features.remove('target')

## Add the transformations - the usuals
transformations = {'1/x'      : lambda x: 1/x,
                   'x**2'     : lambda x: x**2,
                   'x**3'     : lambda x: x**3,
                   'log(x)'   : lambda x: np.log(x),
                   'sqrt(x)'  : lambda x: math.sqrt(x),
                   'exp(x)'   : lambda x: np.exp(x),
                   'log(1/x)' : lambda x: np.log(1/x)
                   }

# Transform all the predicter variables
for ft in features:
    for t in transformations:
        transform_c = [transformations[t](i) for i in cancer_data_df[ft]]
        name = ft +'_'+ t
        cancer_data_df[name] = transform_c

# Reset the list of features
features = list(cancer_data_df.columns)
features.remove('target')

## Create all the product features
#for ft1, ft2 in combinations(features, 2):
#    name = str(ft1)+'_x_'+str(ft2)
#    cancer_data_df[name] = cancer_data_df[ft1] * cancer_data_df[ft2]

# Reset the list of features
features = list(cancer_data_df.columns)

cancer_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns with NaNs
cancer_data_df.dropna(axis=1, inplace=True)

# Update the features list
features = list(cancer_data_df.columns)
features.remove('target')

#Drop rows with NaNs
cancer_data_df.dropna(subset=features, inplace=True)

# Get the target column
target_lbls = cancer_data_df['target']

# Delete the target_column from df
cancer_data_df.drop('target', axis=1, inplace=True)

# Get and split the X and y data
X_train, X_test, y_train, y_test = tts(cancer_data_df,
                                       target_lbls,
                                       stratify=target_lbls,
                                       train_size=0.5)

test_results = []

print
print "Breast Cancer Tumor Data Analysis".center(80, '=')
for C in [0.1, 1, 10, 100]:
    # The Logistic Regression
    cancer_logreg = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)

    # The results
    train_score = cancer_logreg.score(X_train, y_train)
    test_score  = cancer_logreg.score(X_test, y_test)

    coefficients = np.ndarray.tolist(cancer_logreg.coef_)[0]

    included_list = []
    excluded_list = []

    # Sort the coefficients
    for fea, coef in zip(list(features), coefficients):
        if coef == 0:
            excluded_list.append(fea)
        else:
            included_list.append((fea, coef))

    sorted_coef = sorted(included_list, key=lambda x: x[1], reverse=True)

    # Predict the labels for test_y
    predicted_labels = cancer_logreg.predict(X_test)

    # Compare the prediction against the actuals to measure model performance
    actual_labels = y_test
    comparison = zip(predicted_labels, actual_labels)

    # Count the corerct and incorrect predictions
    tp = fp = fn = tn = 0

    for item in comparison:
        if item[0] == 0 and item[0] == item[1]: tn += 1
        if item[0] == 0 and item[0] <> item[1]: fp += 1
        if item[0] == 1 and item[0] == item[1]: tp += 1
        if item[0] == 1 and item[0] <> item[1]: fn += 1

    model_performance = dst.assess_model_performance(tp, fp, fn, tn)

    model_results = [C, tp, fp, fn, tn,
                     model_performance['accuracy  ([tp+tn]/all)'],
                     model_performance['precision (tp/[tp+tn])'],
                     model_performance['recall    (tp/[tp+fn])'],
                     model_performance['f1-score  (2pr/[p+r])']]

    test_results.append(model_results)

    # Prep the data for later plotting - just get the top and bottom 5 coef
    top_five = sorted_coef[ :5]
    bot_five = sorted_coef[-5:]
    top_five.extend(bot_five)
    disp_set = sorted(list(set(top_five)), key=lambda x: x[1], reverse=True)
    midpoint = int(math.ceil(len(disp_set) / 2))

    # Get a new index
    sc_plot_x = [i  for i, _ in enumerate(disp_set)]
    sc_plot_y = [v for _, v in disp_set]
    sc_lbls_x = [i for i, _ in disp_set]

    # Print everything
    print
    print
    print " C = {} ".format(C).center(80, '-')
    print
    print "Train score:".ljust(35, '.') + str(round(train_score, 4)).rjust(8)
    print "Test score:".ljust(35, '.') + str(round(test_score, 4)).rjust(8)
    print
    print "Coefficients:"
    print

    for fea, coef in disp_set[:midpoint]:
            print str(fea).ljust(45, '.'), str(round(coef, 4)).rjust(8)
    print "....."
    for fea, coef in disp_set[midpoint:]:
            print str(fea).ljust(45, '.'), str(round(coef, 4)).rjust(8)

    # plot coefficients
    plt.bar(sc_plot_x, sc_plot_y, tick_label=sc_lbls_x, align='center')
    plt.xlim(xmin=min(sc_plot_x)-0.4, xmax=max(sc_plot_x)+0.4)
    plt.xticks(rotation=90)
    plt.ylim(ymin=min(sc_plot_y)*1.1, ymax=max(sc_plot_y)*1.1)
    plt.ylabel("Regression Coefficients")
    print
    plt.show()
    print

#    print "Excluded:"
#    for item in excluded_list:
#        print item

# Print the summary data
print
print 'Results'.center(80, '-')
print

for result in sorted(test_results, key=lambda x: x[8], reverse=True):
    result_names =['C - inverse regularization strength',
                   'true positives  (tp)',
                   'false positive  (fp)',
                   'false negative  (fn)',
                   'true negative   (tn)',
                   'accuracy        ([tp+tn]/all)',
                   'precision       (tp/[tp+tn])',
                   'recall          (tp/[tp+fn])',
                   'f1-score        (2pr/[p+r])']

    for name_, score_ in zip(result_names, list(result)):
        print name_.ljust(45, '.'), str(round(score_, 4)).rjust(8)
    print
