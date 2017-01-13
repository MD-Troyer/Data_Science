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
from sklearn.decomposition import FactorAnalysis

iris = load_iris()

data, target = iris.data, iris.target

factor = FactorAnalysis(n_components=2, random_state=101).fit(data)

factor_pd = pd.DataFrame(factor.components_, columns=iris.feature_names)

print factor_pd