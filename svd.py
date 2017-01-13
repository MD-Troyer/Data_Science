#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Michael D. Troyer

Date:

Purpose:

Comments:
"""

import numpy as np

#m = np.array([[1,2,3,2,3],[2,1,3,2,3],[3,4,5,6,7],[4,5,6,5,5],[6,7,5,3,4]])
m = np.array([[1,2,3,2,3],[2,1,3,2,3],[3,4,5,6,7],[4,5,6,5,5]])
U, s, Vh = np.linalg.svd(m)

print
print m
print
print U
print
print s
print
print Vh
