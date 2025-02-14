# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:11:51 2024

@author: r96325jj
"""

from mypackage import *
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import time

import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pandas as pd
#from ot_pytorch import sink


data = pd.read_csv("rwmh_results.csv")
X_hist = torch.tensor(data['Sample'].values, dtype=torch.float32)
y_hist = torch.tensor(data['Density'].values, dtype=torch.float32)
X_GPtrain = torch.tensor(data['Proposed Sample'].values, dtype=torch.float32)
y_GPtrain = torch.tensor(data['Proposed Density'].values, dtype=torch.float32)



# print(torch.mean(X_hist))
# print(torch.var(X_hist))


"""
likelihood = ZeroNoiseGaussianLikelihood()
model = ExactGP_manysample(X_GPtrain, y_GPtrain, likelihood, 0.001, 0.1)
#model.plot(limit='logsumexp')


model_hist = histogram(model,X_hist,100)
model_hist.plot()
# model_hist.initial_optimization()
# model_hist.plot()
model_hist.adam()
model_hist.plot()
"""
"""
model_hist2 = histogram(model,X_hist,100)
model_hist.initial_optimization()
init_weights = torch.tensor(model_hist.sample_weights)

init_mean = torch.sum(init_weights * X_hist) / torch.sum(init_weights)
init_variance = torch.sum(init_weights * (X_hist - init_mean) ** 2) / torch.sum(init_weights)
print(init_mean, init_variance)


model_hist2.adam(1000)
adam_weights = torch.tensor(model_hist.sample_weights)
adam_mean = torch.sum(adam_weights * X_hist) / torch.sum(adam_weights)
adam_variance = torch.sum(adam_weights * (X_hist - adam_mean) ** 2) / torch.sum(adam_weights)
print(adam_mean, adam_variance)
# model_hist.plot()
"""


"""
train_x = torch.rand(100, 2)  # 100 points in 2D space
train_y = torch.sin(train_x[:, 0] * 3) + torch.cos(train_x[:, 1] * 3)  # A function of both x1 and x2
train_y += 0.1 * torch.randn_like(train_y)  # Add some noise

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModelND(train_x, train_y, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(50):  # Number of training iterations
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

print("Training complete.")

import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define a 2D GP Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  # Zero mean function
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())  # RBF kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 2: Generate Training Data
train_x = torch.rand(100, 2) * 2 - 1  # 20 random points in [-1,1]²
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])  # Some function of x,y
train_y += 0.1 * torch.randn(train_y.shape)  # Add noise

# Step 3: Train the GP Model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x, train_y, likelihood)
model.train()
likelihood.train()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
for i in range(200):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

# Step 4: Evaluate GP Mean on a Grid
model.eval()
likelihood.eval()

# Create a meshgrid of (x,y) points
grid_size = 50
x1, x2 = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size), indexing="ij")
test_x = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)

# Predict mean and variance
with torch.no_grad():
    pred = model(test_x)
    mean = pred.mean.reshape(grid_size, grid_size)  # Reshape into 2D grid

# Step 5: Plot the Mean Function as a Heatmap
plt.figure(figsize=(8, 6))
plt.contourf(x1.numpy(), x2.numpy(), mean.numpy(), levels=30, cmap="viridis")
plt.colorbar(label="Mean Function Value")
plt.scatter(train_x[:, 0], train_x[:, 1], c="red", marker="x", label="Train Points")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D Gaussian Process Mean Function")
plt.legend()
plt.show()

# Optional: Plot in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x1.numpy(), x2.numpy(), mean.numpy(), cmap="viridis")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Mean Value")
ax.set_title("3D Plot of GP Mean Function")
plt.show()





"""










"""
n=50
sample_points = torch.linspace(-5,30,n)#change to endpoints
GP_opt_points = model.eval_mean_vec(sample_points)
model.histogram(50)
model.binfitter(30,True)
hist_opt_points = model.eval_hist(sample_points)
cost_matrix = ot.dist(sample_points.reshape((n, 1)), sample_points.reshape((n, 1)))
cost_matrix /= cost_matrix.max()
plt.figure(1, figsize=(6.4, 3))
plt.plot(sample_points, GP_opt_points, "b", label="Source distribution")
plt.plot(sample_points, hist_opt_points, "r", label="Target distribution")
plt.legend()

plt.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(GP_opt_points, hist_opt_points, cost_matrix, "Cost matrix M")

lambd = 1e-3
Gs = ot.sinkhorn2(GP_opt_points, hist_opt_points, cost_matrix.numpy(), lambd, verbose=False)
print(Gs)
"""

train_x = torch.rand(100, 2) * 2 - 1  # 20 random points in [-1,1]²
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])  # Some function of x,y
train_y += 0.1 * torch.randn(train_y.shape)  # Add noise

likelihood = ZeroNoiseGaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
model.optimize()
model.plot_kernel()















    