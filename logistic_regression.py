#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic Regression
"""
from __future__ import division

import numpy as np
import pandas as pd
import random
import sklearn

from collections import Counter
from data_science_tools import assess_model_performance
from sklearn.linear_model import LogisticRegression

path = r'/home/michael/Documents/Python/Meds_to_bed_fake_test.csv'

# Read the excel file
df = pd.read_excel(path, 'EDT')

col_index = df.columns
#attributes = ['pv_rx', 'pharm', 'ach02', 'ama07', 'alf01', 'exp20',
#              'h01', 'hhh06', 'hpmc06', 'nhm03', 'pmc03']

attributes = ['pv_rx', 'pharm']

label_field = 'readmit'
label_index = df.columns.get_loc(label_field)

split_prop = 0.5  # train below, test above

# get a list and count of the output labels
target_label_ct = dict(pd.value_counts(df[label_field]))

# Split the data such that a split_prop proportion of each class
# is captured by the test and train sets
# Often issues when using a pure random sort function whe the frequenct of any
# given class is low

train = []
test  = []

for item in target_label_ct:  # the list of label classes
    # for each output label, iterate rows and sample according to split_prop
    for row in df.get_values():
        # Search the output variable only
        if row[label_index] == item:
            if random.random() < split_prop:
                train.append(row)
            else:
                test.append(list(row))

print
print 'number of rows: ', len(df)
print
print 'training records: {}\ntesting records: {}'.format(len(train), len(test))
print

# Get the resulting set label counts
train_label_cts = Counter([row[label_index] for row in train])
test_label_cts  = Counter([row[label_index] for row in test])

print 'training labels: ', train_label_cts
print 'testing labels: ', test_label_cts
print

# Turn them back into dataframes
train = pd.DataFrame(train, columns=col_index)
test  = pd.DataFrame(test, columns=col_index)

# Parcel up the dataframe for analysis
train_x = train[attributes]
train_y = train[label_field]
test_x  = test[attributes]
test_y  = test[label_field]

# Fit the model
logistic = LogisticRegression()
logistic.fit(train_x, train_y)

# test the model

# Predict the labels for test_y
predicted_labels = logistic.predict(test_x)
predicted_probs = logistic.predict_proba(test_x)

# Compare the prediction against the actuals to measure model performance
actual_labels = test_y
comparison = zip(predicted_labels, actual_labels)

# Count the corerct and incorrect predictions

tp = sum([1 for pred, actu in comparison if pred == 1 and actu == 1])
tn = sum([1 for pred, actu in comparison if pred == 0 and actu == 0])
fp = sum([1 for pred, actu in comparison if pred == 1 and actu == 0])
fn = sum([1 for pred, actu in comparison if pred == 0 and actu == 1])

correct = tp+tn
incorrect = fp+fn
capture = correct / (correct + incorrect)

print "Predictions:"
print 'True: {0}\nFalse {1}\nCapture: {2:.3}'\
      ''.format(correct, incorrect, capture)

print
print 'tp: {}\ntn: {}\nfp: {}\nfn: {}'.format(tp, tn, fp, fn)

if not 0 in [tp, tn, fp, fn]:
    performance = assess_model_performance(tp, fp, fn, tn)
    for k, v in performance.items():
        print k, v
else: print '\nat least one prediction class == 0\nthis in not a good model'

# Get the intercept and coefficients
intercept = logistic.intercept_
dataframe = pd.DataFrame(logistic.coef_, columns=attributes)

dataframe['intercept'] = intercept
print
print dataframe
