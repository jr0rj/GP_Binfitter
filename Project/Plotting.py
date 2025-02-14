# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:13:49 2024

@author: r96325jj
"""

from mypackage import *
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from scipy.special import kv, gamma





#Cosine curve with noise plot
"""
torch.manual_seed(0)
x = np.linspace(0,8,1000)
y = np.cos(x)
train_x = torch.linspace(0, 8, 100)
train_y =  torch.cos(train_x) + torch.randn(train_x.size()) * math.sqrt(0.04)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
#model.plot(title = 'Unoptimized GP of Noisy Samples From Cosine Function')
print(model)
model.plot_kernel()
model.optimize(100)
model.plot(title = 'Optimized GP of Noisy Samples From Cosine Curve')
#model.plot(False, title = 'Comparison of Optimized GP to True Function')
print(model)
model.plot_kernel()
#lengthscale: 1.895943  outputscale: 0.982507  noise: 0.042957

"""

#Gaussin/sqrdexp kernel with sigma=1

# Define a 2D Gaussian kernel function
def sqrdexp_kernel(x, y, sigma=1.0,l=1.0):
    return (sigma**2)*np.exp(-((x-y)**2)/(2*(l**2)))

def scale_kernel(x,y,sigma=1.0):
    return sigma

# Generate a grid of x and y values
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Evaluate the kernel on the grid
Z = sqrdexp_kernel(X, Y)

plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Kernel Value')
plt.xlabel('x')
plt.ylabel('x\'')
plt.title('2D Squared Exponential Kernel')
plt.show()


#periodic kernel with period 1
"""
def periodic_kernel(x,y,sigma=1.0,p=1.0,l=1.0):
    return (sigma**2)*np.exp(-(2*np.sin(np.pi*abs(x-y)/p)**2)/(l**2))

# Generate a grid of x and y values
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Evaluate the kernel on the grid
Z = periodic_kernel(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Kernel Value')
plt.xlabel('x')
plt.ylabel('x\'')
plt.title('2D Periodic Kernel')
plt.show()
"""

#matern kernel with shown parameters
"""
def matern_kernel(x1, x2, nu=1.5, length_scale=1.0, variance=1.0):
    # Convert inputs to arrays
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    
    # Compute the Euclidean distance between points
    if x1.ndim == 1 and x2.ndim == 1:
        dist = np.abs(x1[:, None] - x2[None, :])  # Pairwise distances
    else:
        dist = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=2)

    # Rescale distances by the length scale
    scaled_dist = dist / length_scale
    
    if nu == 0.5:
        # Special case: Matérn kernel becomes the Exponential kernel
        kernel = variance * np.exp(-scaled_dist)
    elif nu == 1.5:
        # Special case: Simplified Matérn with ν = 3/2
        kernel = variance * (1 + np.sqrt(3) * scaled_dist) * np.exp(-np.sqrt(3) * scaled_dist)
    elif nu == 2.5:
        # Special case: Simplified Matérn with ν = 5/2
        kernel = variance * (1 + np.sqrt(5) * scaled_dist + (5/3) * scaled_dist**2) * np.exp(-np.sqrt(5) * scaled_dist)
    else:
        # General case for arbitrary ν
        factor = 2**(1-nu) / gamma(nu)
        kernel = variance * factor * (scaled_dist**nu) * kv(nu, scaled_dist)
        kernel[np.isclose(scaled_dist, 0)] = variance  # Handle singularity at 0 distance

    return kernel


# Generate a grid of x and y values
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Evaluate the kernel on the grid
Z = matern_kernel(x, y)
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Kernel Value')
plt.xlabel('x')
plt.ylabel('x\'')
plt.title('2D Matérn Kernel')
plt.show()
"""

#linear kernel
"""
def linear_kernel(x,y,sigma1=1.0,sigma2=1.0,c=0.1):
    return sigma1**2 + sigma2**2*(x-c)*(y-c)

# Generate a grid of x and y values
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Evaluate the kernel on the grid
Z = linear_kernel(X,Y)
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Kernel Value')
plt.xlabel('x')
plt.ylabel('x\'')
plt.title('2D Linear Kernel')
plt.show()
"""

#Normal dist

# def normal_pdf(x, mean, variance):
#     return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

# x_train = -4+8 * torch.rand(10)
# y_train = normal_pdf(x_train,0,1)
# likelihood = ZeroNoiseGaussianLikelihood()
# model = ExactGPModel(x_train,y_train, likelihood)
# model.plot()
# model.optimize(50)
# model.plot()
# model.optimize(100)
# model.plot(title='GP Regression of Exact Standard Gaussian Distribution Samples',limit = 'logsumexp')




