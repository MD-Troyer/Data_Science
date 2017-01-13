#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Author:
Michael D. Troyer

Date:
01/02/17

Purpose:
Machine Learning

Comments:

"""


#--- Imports -------------------------------------------------------------------


# The Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Transformation
from sklearn.preprocessing   import Imputer
from sklearn.preprocessing   import OneHotEncoder
from sklearn.preprocessing   import PolynomialFeatures
from data_science_tools      import PolynomialFeatures_labeled

# Scaling
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import RobustScaler
from sklearn.preprocessing   import Normalizer

# Feature Extraction
from sklearn.decomposition   import PCA
from sklearn.decomposition   import IncrementalPCA
from sklearn.decomposition   import RandomizedPCA
from sklearn.decomposition   import FactorAnalysis
from sklearn.decomposition   import NMF

# Classification
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import LinearSVC
from sklearn.svm             import SVC
from sklearn.naive_bayes     import GaussianNB
from sklearn.naive_bayes     import BernoulliNB
from sklearn.naive_bayes     import MultinomialNB

# Regression
from sklearn.linear_model    import LinearRegression
from sklearn.linear_model    import Lasso
from sklearn.linear_model    import Ridge

# Clustering
from sklearn.cluster         import KMeans
from sklearn.cluster         import AgglomerativeClustering
from sklearn.cluster         import DBSCAN

# Model Tuning and Evaluation
from sklearn.grid_search     import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Pipelining


#--- Create Data Frames --------------------------------------------------------


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

fea_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
tar_df = pd.DataFrame(cancer.target, columns=['target'])

fea_df['target'] = tar_df

target_id = 'target'

# TODO: Raw visualizations

print
print ' Raw Input Data '.center(60, '-')
print
fea_df.info()
if len(fea_df.columns) < 10:
    pd.scatter_matrix(fea_df)


#--- Prepare Variables ---------------------------------------------------------

print
print ' Data Transformations '.center(60, '-')
print

# Remove the target column from df
target_lbls = fea_df[target_id]
fea_df.drop(target_id, axis=1, inplace=True)

# Add polynomials and interactions
poly_thrshld =  1
fea_df = PolynomialFeatures_labeled(fea_df, poly_thrshld)

# Get an updated feature list
features = fea_df.columns

# Rescale the data
scalers = {
          'Standard Scaler' : StandardScaler,
          'Min/Max Scaler'  : MinMaxScaler,
          'Robust Scaler'   : RobustScaler,
          'Normalizer'      : Normalizer
          }

scaler = 'Min/Max Scaler'
print
print "Scaler: {}".format(scaler if scaler else 'None')
print

sclr = scalers[scaler]()
sclr.fit(fea_df)
fea_df = pd.DataFrame(sclr.transform(fea_df), columns=features)

# Apply a decomposition
decomps = {
           'PCA' : PCA,
           'Iterative PCA' : IncrementalPCA,
           'Randomized PCA' : RandomizedPCA,
           'NMF' : NMF
           }

decomposition = 'Iterative PCA'
print
print "Decomposition: {}".format(decomposition if decomposition else 'None')
print

dcmp = decomps[decomposition]()
dcmp.fit(fea_df)
fea_df = pd.DataFrame(dcmp.transform(fea_df), columns=features)

# TODO: Transformed visualizations


#--- Exploratory Analysis ------------------------------------------------------


kmeans      = KMeans(
                     n_clusters=2
                                       )

agglom      = AgglomerativeClustering(
                                       )

dbscan      = DBSCAN(
                                       )

clusterers  = {
              'k-Means Clustering'       : kmeans,
              'Agglomerative Clustering' : agglom,
              'DBSCAN'                   : dbscan
               }

print
print ' Clustering Results '.center(60, '-')
print

for name, model in clusterers.items():
    model.fit(fea_df)
    score = 'Pending'
    print "\n{}:\n\n\tScore: {}".format(name, score)
    print
    print '\t\t', 'Params'.center(45, '-')
    for k, v in sorted(model.get_params().items()):
        if len(str(v)) > 15:
            v = str(v)[:15]
        print '\t\t|', k.ljust(25, '.'), str(v).rjust(15, '.'), '|'
    print '\t\t', ''.center(45, '-')
    print

#TODO: Cluster visualizations


#--- Train Classifiers----------------------------------------------------------


KNrNgb = KNeighborsClassifier(
                                       n_neighbors=1,        # Default is 5
                                       n_jobs=-1
                                       )

RandFr = RandomForestClassifier(
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,       # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       n_jobs=-1,
                                       random_state=42
                                       )

GrdtBC = GradientBoostingClassifier(
                                       loss='deviance',      # D: 'deviance'
                                       learning_rate=0.1,    # Default is 0.1
                                       n_estimators=100,     # Default is 100
                                       max_features='auto',  # Default is 'auto'
                                       max_depth=None,       # Default is None
                                       min_samples_split=2,  # Default is 2
                                       min_samples_leaf=1,   # Default is 1
                                       max_leaf_nodes=None,  # Default is None
                                       random_state=42
                                       )

LogReg = LogisticRegression(
                                       penalty='l1',         # Default is 'l2'
                                       C=250,                # Default is 1.0
                                       n_jobs=-1,
                                       random_state=42
                                       )

nnwMLP = MLPClassifier(
                                       max_iter=1000         # Default is 100
                                       )

LinSVC = LinearSVC(
                                       penalty='l2',         # Default is 'l2'
                                       dual=False,           # Default is 'True'
                                       C=0.1,                # Default is 1.0
                                       random_state=42
                                       )

svmSVC = SVC(
                                       kernel='rbf',         # Default is 'rbf'
                                       C=1.0,                # Default is 1.0
                                       gamma='auto',         # Default is 'auto'
                                       random_state=42
                                       )

#GausNB = GaussianNB()

#BernNB = BernoulliNB()

#MltiNB = MultinomialNB()

classifiers = {
#              'K Nearest Neighbors'                   : KNrNgb,
#              'Random Forest'                         : RandFr,
              'Gradient Boosted Decision Trees'       : GrdtBC,
              'Logistic Regression'                   : LogReg,
              'Neural Network Multi-layer Perceptron' : nnwMLP,
              'Linear Support Vector Classifier'      : LinSVC,
              'Support Vector Machine'                : svmSVC
#              'Gaussian Naive Bayes'                  : GausNB,
#              'Bernoulli Naive Bayes'                 : BernNB,
#              'Multinomial Naive Bayes'               : MltiNB
              }

# TODO: Raw classifier visualizations


#--- Tune Models -----------------------------------------------------------


# Grid Search

# TODO: OPtimized classifier visualizations

cv = 10
print
print ' Model Cross-Validation (cv={}) '.format(cv).center(60, '-')
print
print 'Coming Soon'
print


#--- Evaluate Models -----------------------------------------------------------


print
print ' Model Results '.center(60, '-')
print

#TODO: Implement own K-fold with model performance evaluation

for name, model in classifiers.items():
    model.fit(fea_df, target_lbls)
    scores = cross_val_score(model, fea_df, target_lbls, cv=cv)
    print "\n{}:\n\n\tAccuracy: {:.3f} (+/- {:.3f})"\
          "".format(name, scores.mean(), scores.std() * 2)
    print
    print '\t\t', 'Params'.center(45, '-')
    for k, v in sorted(model.get_params().items()):
        if len(str(v)) > 15:
            v = str(v)[:15]
        print '\t\t|', k.ljust(25, '.'), str(v).rjust(15, '.'), '|'
    print '\t\t', ''.center(45, '-')
    print

# TODO: Evaluation visualizations