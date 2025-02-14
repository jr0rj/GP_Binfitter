# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:20:41 2025

@author: r96325jj
"""

from numpy.polynomial import Chebyshev
import gpytorch
import torch
from .maximums import GP_logsumexp, GP_hardmax, hardmax, softmax
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import wasserstein_distance
from .ExactGP_class import ExactGPModel


class GPModelND(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelND, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)