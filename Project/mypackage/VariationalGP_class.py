# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:16:47 2024

@author: r96325jj
"""

from numpy.polynomial import Chebyshev
import gpytorch
import torch
from .maximums import GP_logsumexp, GP_hardmax
import numpy as np
import matplotlib.pyplot as plt


class VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGP,self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
