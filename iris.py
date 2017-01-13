#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Michael D. Troyer

Date:

Purpose:

Comments:
"""

from sklearn.datasets import load_iris

iris = load_iris()

data, target = iris.data, iris.target

print iris['feature_names']

print type(data), data

print type(target), target