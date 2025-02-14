# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:57:43 2024

@author: r96325jj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:01:31 2024

@author: r96325jj
"""



from numpy.polynomial import Chebyshev
import gpytorch
import torch
from .maximums import GP_logsumexp, GP_hardmax
import numpy as np
import matplotlib.pyplot as plt


class ExactGPModel2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        
    def __str__(self):
        return 'lengthscale: %f  outputscale: %f  noise: %f' % (
            self.covar_module.base_kernel.lengthscale.item(),
            self.covar_module.outputscale.item(),
            self.likelihood.noise.item(),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def plot(self,observed = True,limit = 'none'):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            xrange = torch.linspace(min(self.train_x), max(self.train_x), 1000,dtype=torch.float32)
            observed_pred = self(xrange)
        mean = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.numpy()
        upper = upper.numpy()
        if limit == 'logsumexp':
            mean,upper,lower = GP_logsumexp(mean,upper,lower,1000)
        if limit =='hardmax':
            mean,upper,lower = GP_hardmax(mean,upper,lower)
        with torch.no_grad():#torch.nograd skips gradianet calcs to save time, don'tneed grads in forward calcs
            f, ax = plt.subplots(1, 1)
            ax.grid()
            if observed == True:
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'x',color='green')
            ax.plot(xrange, mean, 'b')
            ax.fill_between(xrange, lower, upper, alpha=0.4,color='orange')
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            
    
    def eval_mean(self,x):
        self.eval()
        x = torch.tensor([[x]])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        return mean
    
    def eval_mean_vec(self,x):
        self.eval()
        x = torch.tensor(x,dtype = torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        return mean
    
    def optimize(self,tol = 1e-4, training_iter = 100,output = False):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if output ==True:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %f   lengthscale: %f  outputscale: %f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.covar_module.outputscale.item(),
                ))
                optimizer.step()
        else:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()
        

    
    def mean_RWMH(self,start_point = 0, beta = 2.3, iterations = 1000, full = True):
        self.eval()
        samples = np.zeros(iterations)
        samples[0] = start_point
        current_sample = start_point
        #accep_prob = np.zeros(iterations)
        acep = 0
        densities = np.zeros(iterations)
        densities[0] = self.eval_mean(start_point)
        ind_accep = np.array(0)
        for i in range(1,iterations):
            proposal_sample = np.random.normal(current_sample, beta*beta)
            U = np.random.uniform(0,1)
            proposal_density = self.eval_mean(proposal_sample)
            current_density = self.eval_mean(current_sample)
            alpha = min(1, proposal_density/current_density)
            #accep_prob[i] = alpha
            if U <= alpha:
                current_sample = proposal_sample
                #samples[i] = current_X
                densities[i] = proposal_density
                acep+=1
                ind_accep = np.append(ind_accep, i)
            else:
                densities[i] = current_density
            samples[i] = current_sample
            #densities[i] = self.chebyshev(current_sample)
        print("Acceptance rate: ", acep/iterations)
        if full == True:
            return samples, densities#, accep_prob
        if full == False:
            return samples[ind_accep], densities[ind_accep]#, accep_prob[ind_accep]
        
    def mean_RWMH_chebyshev(self,start_point = 0, beta = 2.3, iterations = 1000, full = True,nodes = 100):
        self.chebyshev_init(nodes)
        self.eval()
        samples = np.zeros(iterations)
        samples[0] = start_point
        current_sample = start_point
        #accep_prob = np.zeros(iterations)
        acep = 0
        densities = np.zeros(iterations)
        densities[0] = self.chebyshev(start_point)
        ind_accep = np.array(0)
        for i in range(1,iterations):
            proposal_sample = np.random.normal(current_sample, beta*beta)
            U = np.random.uniform(0,1)
            proposal_density = self.chebyshev(proposal_sample)
            current_density = self.chebyshev(current_sample)
            alpha = min(1, proposal_density/current_density)
            #accep_prob[i] = alpha
            if U <= alpha:
                current_sample = proposal_sample
                #samples[i] = current_X
                densities[i] = proposal_density
                acep+=1
                ind_accep = np.append(ind_accep, i)
            else:
                densities[i] = current_density
            samples[i] = current_sample
            #densities[i] = self.chebyshev(current_sample)
        print("Acceptance rate: ", acep/iterations)
        if full == True:
            return samples, densities#, accep_prob
        if full == False:
            return samples[ind_accep], densities[ind_accep]#, accep_prob[ind_accep]
    
    def chebyshev_init(self,nodes):
         chebyshev_nodes = np.cos((2 * np.arange(nodes) + 1) / (2 * nodes) * np.pi)
         lims = max(abs(self.train_x.numpy()))
         chebyshev_nodes = 0.5 * (chebyshev_nodes + 1) * (2*lims) - lims
         y_vals = self.eval_mean_vec(chebyshev_nodes)
         chebyshev_coeffs = Chebyshev.fit(chebyshev_nodes, y_vals, deg=nodes-1)
         self.chebyshev = chebyshev_coeffs
         
         # x = np.linspace(-40,40,1000)
         # y = self.chebyshev(x)
         # plt.plot(x,y)
    
    def binfitter(self,bins = 50):
        bin_dens, bin_edges=np.histogram(self.train_x,density = True,range=(min(self.train_x.numpy())-0.1,max(self.train_x.numpy())+0.1),bins=bins)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        GP_heights = self.eval_mean_vec(bin_midpoints)
        bindex = np.digitize(self.train_x.numpy(),bin_edges)
        weights = GP_heights/bin_dens
        new_vector = weights[bindex-1]
        plt.hist(self.train_x, density = True, weights = new_vector,color = 'green',alpha = 0.7,bins=bins)
        plt.hist(self.train_x,density = True,alpha = 0.5, color = 'red',bins='auto')
    
    def binfitter_plus(self,bins=50,samples=1000,nodes=200):
        X_mcmc, y_mcmc = self.mean_RWMH_chebyshev(nodes=nodes,iterations = samples)#, full = False)
        #plt.plot(X_mcmc)
        X_mcmc = torch.tensor(X_mcmc,dtype = torch.float32)
        new_x = torch.cat((self.train_x,X_mcmc),dim=0)
        bin_density, bin_edges=np.histogram(new_x,density = True,range=(min(new_x.numpy())-0.1,max(new_x.numpy())+0.1),bins=bins)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        GP_heights = self.eval_mean_vec(bin_midpoints)
        bindex = np.digitize(new_x.numpy(),bin_edges)
        weights = GP_heights/bin_density
        new_vector = weights[bindex-1]
        plt.hist(new_x, density = True,color = 'red',alpha = 0.5,bins=bins)
        plt.hist(new_x, density = True, weights = new_vector,color = 'green',alpha = 0.5,bins=bins)
        
         



