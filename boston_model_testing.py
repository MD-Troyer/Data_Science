#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:
Michael D. Troyer

Date:
01/02/17

Purpose:
Model testing

Comments:

"""

#--- Imports -------------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#--- Data Frames ---------------------------------------------------------------

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

fea_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
tar_df = pd.DataFrame(cancer.target, columns=['target'])

fea_df['target'] = tar_df

#path = r''
#fea_df = pd.read_csv(os.path.join(path, 'train.csv'), header=0)
#tar_df = pd.read_csv(os.path.join(path, 'test.csv'),  header=0)

#--- Variables -----------------------------------------------------------------

### Drop the useless colunms
##fea_df.drop([], axis=1, inplace=True)
##tar_df.drop([], axis=1, inplace=True)

# Create the derived features
target_id = 'target'
features = list(fea_df.columns)
features.remove(target_id)

## Add the transformations
#transformations = {'1/x'      : lambda x: 1/float(x),
#                   'x**2'     : lambda x: float(x)**2,
#                   'x**3'     : lambda x: float(x)**3,
#                   'log(x)'   : lambda x: np.log(float(x)),
#                   'sqrt(x)'  : lambda x: math.sqrt(float(x)),
#                   'log(1/x)' : lambda x: np.log(1/float(x))}
#
#for ft in features:
#    for t in transformations:
#        try:
#            fea_transformed = [transformations[t](f) for f in fea_df[ft]]
#            fea_name = ft +'_'+ f
#            fea_df[fea_name] = fea_transformed
#
#        except:
#            pass
#
## 2-part product
#for f1, f2 in combinations(features, 2):
#    name = f1+'_'+f2
#    fea_df[name] = fea_df[f1] * fea_df[f2]
#
## 3-part product
#for f1, f2, f3 in combinations(features, 3):
#    name = f1+'_'+f2+'_'+f3
#    fea_df[name] = fea_df[f1] * fea_df[f2] * fea_df[f3]

# Remove the target column from df
target_lbls = fea_df[target_id]
fea_df.drop(target_id, axis=1, inplace=True)

#--- Train Models --------------------------------------------------------------

# Get and split the X and y data
X_train, X_test, y_train, y_test = tts(fea_df,
                                       target_lbls,
                                       train_size=0.7,
                                       stratify=target_lbls)

# Rescale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# Apply PCA transformation
pca = PCA()
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test  = pca.transform(X_test)


LogReg = LogisticRegression(
                                       penalty='l1',         # Default is 'l2'
                                       C=0.1,                # Default is 1.0
                                       n_jobs=-1,
                                       random_state=42
                                       )

LinSVC = LinearSVC(
                                       penalty='l2',         # Default is 'l2'
                                       dual=False,           # Default is 'True'
                                       C=0.1,                # Default is 1.0
                                       random_state=42
                                       )

svmSVC = SVC(
                                       kernel='rbf',         # Default is 'rbf'
                                       C=5,                  # Default is 1.0
                                       gamma='auto',         # Default is 'auto'
                                       random_state=42)

RandFr = RandomForestClassifier(
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,         # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       n_jobs=-1,
                                       random_state=42
                                       )

GrdtBC = GradientBoostingClassifier(
                                       loss='deviance',      # Default is 'deviance'
                                       learning_rate=0.1,    # Default is 0.1
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,       # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       random_state=42
                                       )

KNrNgb = KNeighborsClassifier(
                                       n_neighbors=1,        # Default is 5
                                       n_jobs=-1
                                       )

nnwMLP = MLPClassifier(max_iter=1000)

models_list = [LogReg, LinSVC, svmSVC, RandFr, GrdtBC, KNrNgb, nnwMLP]
model_names = ['LogReg', 'LinSVC', 'svmSVC', 'RandFr', 'GrdtBC', 'KNrNgb', 'nnwMLP']

for model in models_list:
    model.fit(X_train, y_train)

#--- Make the Predictions ------------------------------------------------------

predictions = [(m.score(X_train, y_train), m.score(X_test, y_test))
                for m in models_list]

results = zip(model_names, predictions)

print
for r in sorted(results, key=lambda x: x[1][1], reverse=True):
    print "{}:\tTrain: {:.4f},\tTest: {:.4f},\tProd: {:.4f}"\
          "".format(r[0], r[1][0], r[1][1], r[1][0] * r[1][1])
print

#--- Metadata ------------------------------------------------------------------

#fea_df.info()
