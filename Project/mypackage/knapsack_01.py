# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:11:49 2024

@author: r96325jj
"""

import numpy as np

n = 50

weights = np.ones(n)
values = np.random.rand(n)
# print(weights)
# print(values)
knapsack_capacity = n

def knapsack(values, weights, n, W):
    # Initialize the matrix with zeros
    m = np.zeros((n + 1, W + 1), dtype=int)
    
    # Fill in the matrix according to the 0/1 knapsack logic
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if weights[i] > j:
                m[i, j] = m[i - 1, j]
            else:
                m[i, j] = max(m[i - 1, j], m[i - 1, j - weights[i]] + values[i])
                
    return m[n, W] 


print(knapsack(values,weights,n,knapsack_capacity))