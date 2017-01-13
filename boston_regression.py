#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression

Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive

    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None
"""
from __future__ import division
import copy
import data_science_tools as dst
import numpy as np
import math
import pandas as pd
import random
import re
from collections import Counter, defaultdict
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression

# Load the predictor data and get the column names, load the target array
boston = load_boston()
targets_ = boston.target

# Set the target field name
target_col = 'MEDV_target'

# Bucketize the points standardize to units of 5k - median house prince in 5k
targets = [dst.bucketize(point, 5)/5 for point in targets_]

columns_ = [str(item)+'_X' for item in boston.feature_names]

# Create the dataframe
df = pd.DataFrame(boston.data, columns=columns_)
df[target_col] = targets

features = list(df.columns)

# Add the transformations - the usuals
transformations = {'1/x'      : lambda x: 1/x,
                   'x**2'     : lambda x: x**2,
                   'x**3'     : lambda x: x**3,
                   'log(x)'   : lambda x: np.log(x),
                   'sqrt(x)'  : lambda x: math.sqrt(x),
                   'exp(x)'   : lambda x: np.exp(x),
                   'log(1/x)' : lambda x: np.log(1/x)}

# Transform all the predicter variables
for ft in features:
    if not ft == target_col:
        for t in transformations:
            v = [transformations[t](i) for i in df[ft]]
            name = re.sub('_X', '_'+t, ft)
            df[name] = v

# reset the list of features
features = list(df.columns)

print 'raw\nrows: {}\ncols: {}'.format(len(df), len(features))

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns with NaNs
df.dropna(axis=1, inplace=True)

# Update the features list
features = list(df.columns)

#Drop rows with NaNs
df.dropna(subset=features, inplace=True)

print 'clean\nrows: {}\ncols: {}'.format(len(df), len(features))


# Describe data
#desc_df = pd.DataFrame(df[features])
#list_of_rows = list(desc_df.values)
#print list_of_rows
#dst.n_d_numeric_description(list_of_rows)

# Get the target field index
target_idx = df.columns.get_loc(target_col)

# Set the split proportion
split_prop = 0.7  # train below, test above

# Get a list and count of the output labels
target_label_ct = dict(pd.value_counts(df[target_col]))

## Split the data such that a split_prop proportion of each class
## is captured by the test and train sets
## There are often issues when using a pure random sort function whe the
## frequency of any given class is low
## All classes should be represented in the train y

train = []
test  = []

for item in target_label_ct:  # the list of label classes
    # for each output label, iterate rows and sample according to split_prop
    for row in df.get_values():
        # Search the output variable only
        if row[target_idx] == item:
            if random.random() < split_prop:
                train.append(row)
            else:
                test.append(list(row))

print
print 'training records: {}\ntesting records: {}'.format(len(train), len(test))
print

# Turn them back into dataframes
train = pd.DataFrame(train, columns=features)
test  = pd.DataFrame(test, columns=features)

# Parcel up the dataframe for analysis
feature_sort_list =  transformations.keys()
feature_sort_list.append('_X')

def test_model(train, test, features, target_col, test_trans):
    test_features = [item for item in features if test_trans in str(item)]

    train_x = train[test_features]
    train_y = train[target_col]
    test_x  = test[test_features]
    test_y  = test[target_col]

    # Fit the model
    logistic = LogisticRegression()
    logistic.fit(train_x, train_y)

    # Predict the labels for test_y
    predicted_labels = logistic.predict(test_x)
#    predicted_probs = logistic.predict_proba(test_x)

    # Compare the prediction against the actuals to measure model performance
    actual_labels = test_y
    comparison = zip(predicted_labels, actual_labels)

    # Count the corerct and incorrect predictions
    correct = 0
    incorrect = 0

    for item in comparison:
        if item[0] == item[1]:
            correct += 1
        else:
            incorrect += 1

    capture = correct / (correct + incorrect)

    print "Predictions "+test_trans+":"
    print 'True: {0}\nFalse {1}\nCapture: {2:.3}'\
          ''.format(correct, incorrect, capture)

    # Get the intercept and coefficients
    intercept = logistic.intercept_
    dataframe = pd.DataFrame(logistic.coef_, columns=test_features)
    dataframe['intercept'] = intercept

    return correct, incorrect, capture, dataframe

for feature in features:
    if 'exp(x)' in str(feature):
        dst.one_d_numeric_description(train[feature])

test_model(train, test, features, target_col, 'exp(x)')