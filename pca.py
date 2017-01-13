#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Michael D. Troyer

Date:

Purpose:

Comments:
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

data, target = iris.data, iris.target

pca = PCA().fit(data)

print "Explained variance by component: %s" % pca.explained_variance_ratio_
print pd.DataFrame(pca.components_, columns=iris.feature_names)