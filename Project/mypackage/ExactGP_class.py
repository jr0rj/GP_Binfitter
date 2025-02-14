# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:01:31 2024

@author: r96325jj
"""



from numpy.polynomial import Chebyshev
import gpytorch
import torch
from .maximums import GP_logsumexp, GP_hardmax, hardmax, softmax, softmax2
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import wasserstein_distance





class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        #define GP and save features to be used in other functions
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        
    def __str__(self):
        #print(model) output, gives hyperparameters
        return 'lengthscale: %f  outputscale: %f  noise: %f' % (
            self.covar_module.base_kernel.lengthscale.item(),
            self.covar_module.outputscale.item(),
            self.likelihood.noise.item()
        )

    def forward(self, x):
        #forward calcs, used to predict mean and variance
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def plot(self,observed = True,limit = 'none',**kwargs):#xlab = 'x',ylab = 'y', title = ''):
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
            mean,upper,lower = GP_logsumexp(mean,upper,lower,10000)
        if limit =='hardmax':
            mean,upper,lower = GP_hardmax(mean,upper,lower)
        with torch.no_grad():#torch.nograd skips gradianet calcs to save time, don'tneed grads in forward calcs
            f, ax = plt.subplots(1, 1)
            ax.grid()
            if observed == True:
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'x',color='green')
            ax.plot(xrange, mean, 'b')
            ax.fill_between(xrange, lower, upper, alpha=0.4,color='orange')
            ax.legend(['Observed Data','Mean', 'Confidence'])
            ax.set_xlabel(kwargs.get("ylabel", "y"))
            ax.set_ylabel(kwargs.get("xlabel", "x"))
            ax.set_title(kwargs.get("title",""))
            #plt.show()

    
    def evaluate(self,x,limit = 'none'):
        self.eval()
        self.likelihood.eval()
        x = torch.tensor(x,dtype = torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        std = np.sqrt(observed_pred.variance.numpy())
        if limit == 'logsumexp':
            return softmax(mean,10000),std
        elif limit == 'hardmax':
            return hardmax(mean),std
        return mean, std
    
    
    def eval_mean(self,x, limit = 'none'):
        self.eval()
        self.likelihood.eval()
        x = torch.tensor([[x]],dtype = torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        if limit == 'logsumexp':
            return softmax(mean,10000)
        if limit == 'hardmax':
            return hardmax(mean)
        return mean
    
    def eval_mean_vec(self,x, limit = 'none'):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean
        if limit == 'logsumexp':
            return softmax2(mean,10000)
        if limit == 'hardmax':
            return hardmax(mean)
        return mean
    
    def optimize(self, training_iter = 100,output = False):
        self.train()
        self.likelihood.train()
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
                print('Iter %d/%d - Loss: %f   lengthscale: %f  outputscale: %f  noise: %f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.covar_module.outputscale.item(),
                    self.likelihood.noise.item(),
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
        

    
    def mean_RWMH(self,start_point = 0, beta = 2.3, iterations = 1000, full = True,limit='none'):
        self.eval()
        self.likelihood.eval()
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
            proposal_density = self.eval_mean(proposal_sample,limit)
            current_density = self.eval_mean(current_sample,limit)
            alpha = min(1, proposal_density/current_density)
            #accep_prob[i] = alpha
            if U <= alpha:
                current_sample = proposal_sample
                densities[i] = proposal_density
                acep+=1
                ind_accep = np.append(ind_accep, i)
            else:
                densities[i] = current_density
            samples[i] = current_sample
        print("Acceptance rate: ", acep/iterations)
        if full == True:
            return samples, densities#, accep_prob
        if full == False:
            return samples[ind_accep], densities[ind_accep]#, accep_prob[ind_accep]
    # def mean_RWMH(self,start_point = 0, beta = 2.3, iterations = 1000, full = True):
    #     samples = torch.empty(iterations)
    #     zero = torch.tensor([0])
    #     samples[0] = start_point
    #     current_sample = start_point
    #     #accep_prob = np.zeros(iterations)
    #     acep = 0
    #     densities = torch.empty(iterations)
    #     densities[zero] = self.eval_mean(samples[zero])
    #     ind_accep = np.array(0)
    #     for i in range(1,iterations):
    #         proposal_sample = torch.normal(mean=current_sample,std=beta)
    #         proposal_density = self.eval_mean(proposal_sample)
    #         current_density = self.eval_mean(current_sample)
    #         alpha = min(1, proposal_density/current_density)
    #         #accep_prob[i] = alpha
    #         if torch.rand(1) <= alpha:
    #             current_sample = proposal_sample
    #             densities[i] = proposal_density
    #             acep+=1
    #             ind_accep = np.append(ind_accep, i)
    #         else:
    #             densities[i] = current_density
    #         samples[i] = current_sample
    #     print("Acceptance rate: ", acep/iterations)
    #     if full == True:
    #         return samples, densities#, accep_prob
    #     if full == False:
    #         return samples[ind_accep], densities[ind_accep]#, accep_prob[ind_accep]
        
    def mean_RWMH_chebyshev(self,start_point = 0, beta = 2.3, iterations = 1000, full = True,nodes = 100):
        self.chebyshev_init(nodes)
        self.eval()
        self.likelihood.eval()
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
    
    def histogram(self, bins):
        bin_dens, bin_edges=np.histogram(self.train_x.numpy(),density = True,range=(min(self.train_x.numpy())-1e-5,max(self.train_x.numpy())+1e-5),bins=bins)
        self.bin_dens = bin_dens
        self.bin_edges = bin_edges
        self.bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
    def eval_hist(self, points):
        # Digitize points to find their bin indices
        bin_indices = np.digitize(points, self.bin_edges, right=False) - 1
        
        # Handle edge cases: points exactly at the rightmost edge should belong to the last bin
        bin_indices = np.clip(bin_indices, 0, len(self.bin_dens) - 1)
        
        # Map bin indices to heights
        dens_of_points = self.bin_dens[bin_indices]
        
        return dens_of_points
    
    def plot_hist(self,**kwargs):
        plt.bar(self.bin_midpoints,self.bin_dens,width = np.diff(self.bin_edges),color='green',alpha=1)
    
    def binfitter(self,bins = 50,plot=False):
        #find heights and edges of each bin with current weight 1 samples
        bin_dens, bin_edges=np.histogram(self.train_x.numpy(),density = True,range=(min(self.train_x.numpy())-1e-5,max(self.train_x.numpy())+1e-5),bins=bins)
        #i=16        
        #print((bin_dens*200*np.diff(bin_edges))[i])
        
        #find midpoints of bins and GP mean at these points
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        GP_heights = self.eval_mean_vec(bin_midpoints,limit='logsumexp')
        
        #place points into bins so the weights can be assigned binwise
        bindex = np.digitize(self.train_x.numpy(),bin_edges)
        
        #find weights - to be replced with some sort of optimization
        weights = GP_heights/bin_dens
        
        #change weight to zero where there is an empty bin
        div_zero = np.where(np.isinf(weights))
        weights[div_zero] = 0.0
        #print(weights)
        #print(len(weights))
        
        #finx index of weights to be the same as bins
        weights = weights[bindex-1]
        
        #find heights, edges and midpoints of new weighted histogram
        new_bin_dens,new_bin_edges=np.histogram(self.train_x.numpy(), density = True, weights = weights,bins=bins,range=(min(self.train_x.numpy())-1e-5,max(self.train_x.numpy())+1e-5))
        new_bin_midpoints = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
        #TODO: check if below actually does anything
        print(sum(new_bin_dens*np.diff(new_bin_edges)))
        new_bin_dens = new_bin_dens*np.sum(weights)/len(self.train_x.numpy())
        print(sum(new_bin_dens*np.diff(new_bin_edges)))
        if plot==True:
            #plt.hist(self.train_x.numpy(), density=True,weights = weights,color = 'green',alpha = 0.7,bins=bins)
            plt.bar(new_bin_midpoints,new_bin_dens,width = np.diff(new_bin_edges),color='green',alpha=0.5)
        print(np.sum(abs(GP_heights - new_bin_dens)))
        # self.binfitter_edge = new_bin_edges
        # self.binfitter_dens = new_bin_dens
        return new_bin_dens, new_bin_edges
    
    # def binfitter_plus(self,bins=50,samples=1000,nodes=200):
    #     X_mcmc, y_mcmc = self.mean_RWMH_chebyshev(nodes=nodes,iterations = samples,beta = 2.0, full = False)
    #     #plt.plot(X_mcmc)
    #     X_mcmc = torch.tensor(X_mcmc,dtype = torch.float32)
    #     new_x = torch.cat((self.train_x,X_mcmc),dim=0)
    #     bin_density, bin_edges=np.histogram(new_x,density = True,range=(min(new_x.numpy())-0.1,max(new_x.numpy())+0.1),bins=bins)
    #     bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     GP_heights = self.eval_mean_vec(bin_midpoints)
    #     bindex = np.digitize(new_x.numpy(),bin_edges)
    #     weights = GP_heights/bin_density
    #     new_vector = weights[bindex-1]
    #     plt.hist(new_x, density = True,color = 'red',alpha = 0.5,bins=bins)
    #     plt.hist(new_x, density = True, weights = new_vector,color = 'green',alpha = 0.5,bins=bins)
    #     return new_vector, new_x.numpy()
    
    
    def quadrature(self,lower_lim=-1,upper_lim=1):
        pdf = lambda x: self.eval_mean(x,limit='logsumexp')
        prob, error = quad(pdf, lower_lim,upper_lim, limit = 1000)
        return prob, error
        
    def KullLeib(self,pdf_x,pdf_heights):
        gp_heights = self.eval_mean_vec(pdf_x,limit='logsumexp')
        
        kl_divergences = gp_heights*np.log(gp_heights/pdf_heights)
        index = np.where(np.isnan(kl_divergences))[0]
        if np.all(gp_heights[index] == pdf_heights[index]):
            print('yay')
        return np.sum(kl_divergences)
    
    def EucDist(self,x,y):
        gp_heights = self.eval_mean_vec(x,limit='logsumexp')
        errors = abs(y-gp_heights)
        return np.sum(errors)
    
    def Wasserstein(self,x,y):
        gp_heights = self.eval_mean_vec(x,limit='logsumexp')
        wass_dis = wasserstein_distance(y,gp_heights)
        return wass_dis
    
    def plot_kernel(self, lower_lim=-3, upper_lim=3, bins=100, **kwargs):
        x = torch.linspace(lower_lim, upper_lim, bins).view(-1, 1)
        # Evaluate the kernel on the grid
        Z = self.covar_module(x,).evaluate()
        plt.imshow(Z.detach().numpy(), cmap="viridis", extent=[lower_lim,upper_lim,lower_lim,upper_lim])
        plt.colorbar(label="Kernel Value")
        plt.title(kwargs.get("title", "Kernel Matrix Heatmap"))
        plt.xlabel(kwargs.get("xlabel", "x"))
        plt.ylabel(kwargs.get("ylabel", "x'"))
        plt.show()
    


