# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:20:45 2024

@author: r96325jj
"""

import gpytorch


class ZeroNoiseGaussianLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self):
        super().__init__(noise_constraint=gpytorch.constraints.GreaterThan(0.0))