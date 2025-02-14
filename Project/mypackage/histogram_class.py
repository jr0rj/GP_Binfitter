# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:22:15 2025

@author: r96325jj
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import ot
import ot.plot
import scipy.optimize
from .maximums import GP_logsumexp, GP_hardmax, hardmax, softmax, softmax2


class histogram():
    def __init__(self, GP_model,samples,nbins):
        bin_dens, bin_edges=np.histogram(samples.numpy(),density = True,range=(min(samples.numpy())-1e-5,max(samples.numpy()+1e-5)),bins=nbins)
        
        #Fixed parameters
        self.samples = samples
        self.bin_edges = torch.tensor(bin_edges)
        self.bin_midpoints = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2     
        self.nbins = nbins
        self.GP_model = GP_model        
        self.bindex = torch.bucketize(self.samples,self.bin_edges, right = True) 
        
        #Parameters to be optimized
        self.bin_dens = torch.tensor(bin_dens)
        self.bin_weights = torch.ones(nbins)/len(samples)
        self.sample_weights = self.bin_weights[self.bindex-1]
        
        
    def plot(self, plot_GP=False,**kwargs):
        
        plt.xlabel(kwargs.get("xlabel", "x"))
        plt.ylabel(kwargs.get("ylabel", "Density"))
        plt.title(kwargs.get("title", "Probability Distribution Estimate"))  
        plt.bar(self.bin_midpoints, self.bin_dens,width = torch.diff(self.bin_edges),color='green',**kwargs)
        current_ymin, current_ymax = plt.ylim() 
        plt.ylim(0, current_ymax)
        if plot_GP == True:
            self.GP_model.plot(limit = 'logsumexp')
        def normal_pdf(x, mean, variance):
            """Calculate the probability density function (PDF) of a normal distribution."""
            return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)
        def multimodal_pdf(x, mean1, variance1, mean2, variance2, weight1=0.5):
            """Calculate the PDF of a multimodal distribution formed by two normal distributions."""
            weight2 = 1 - weight1  # Ensure the weights sum to 1
            pdf_1 = normal_pdf(x, mean1, variance1)  # First normal distribution
            pdf_2 = normal_pdf(x, mean2, variance2)  # Second normal distribution
            return weight1 * pdf_1 + weight2 * pdf_2  # Weighted sum of the two PDFs

        mean1 = 0.0
        variance1 = 1.0

        mean2 = 25
        variance2 = 2.0

        # Range of x values
        x_vals = np.linspace(-10, 40, 3000)

        # Calculate the multimodal PDF
        multimodal_values = multimodal_pdf(x_vals, mean1, variance1, mean2, variance2, weight1=0.5)
        plt.plot(x_vals, multimodal_values, label="Multimodal PDF", color="blue")
        plt.show()
    
    
    def initial_optimization(self):
        GP_midpoint_densities = self.GP_model.eval_mean_vec(self.bin_midpoints, limit = 'logsumexp')
        assert(torch.is_tensor(GP_midpoint_densities))
        #find weights - to be replaced with some sort of optimization (or not based on work)
        weights = GP_midpoint_densities/self.bin_dens#len or .numel()
        assert(torch.is_tensor(weights))
        #change weight to zero where there is an empty bin
        div_zero = torch.isinf(weights)
        weights[div_zero] = 0.0
        #print(sum(weights))
        assert(torch.is_tensor(weights))
        self.update(weights)
        
        
        # #find index of weights to be the same as bins
        # sample_weights = weights[self.bindex-1]
        # #find heights, edges and midpoints of new weighted histogram
        # new_bin_dens,new_bin_edges=np.histogram(self.samples.numpy(), density = True, weights = sample_weights,bins=self.nbins,range=(min(self.samples.numpy())-1e-5,max(self.samples.numpy())+1e-5))
        # new_bin_midpoints = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
        # print(new_bin_dens)
        # self.bin_dens = new_bin_dens
        # self.bin_edges = new_bin_edges
        # self.bin_midpoints = new_bin_midpoints  
        # self.bin_weights = weights
        # self.sample_weights = sample_weights
        # # print(sum(weights))
        # # print(sum(sample_weights))
        # # print(sum(self.bin_dens*np.diff(self.bin_edges)))
        
    def update(self,weights):
        new_sample_weights = weights[self.bindex - 1]
        self.sample_weights = new_sample_weights
        self.bin_weights = weights
        new_bin_dens = torch.histc(self.samples, bins=self.nbins)
        new_bin_dens *= self.bin_weights
        new_bin_dens/=sum(new_bin_dens*torch.diff(self.bin_edges))
        self.bin_dens = new_bin_dens

        

    
    def adam(self,threshold = 1e-8, max_iters=10000, lr=0.1):
        #Define parameters and optimizer
        weights = torch.nn.Parameter(self.bin_weights.clone().detach().requires_grad_(True))
        optimizer = torch.optim.Adam([weights],lr = lr)
    
        GP_densities = self.GP_model.eval_mean_vec(self.bin_midpoints)
        GP_densities_normalised = GP_densities#/sum(GP_densities))

        
        for i in range(0,max_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            #print(weights)
            
            # cost_matrix = ot.dist(hist_positions[:, None], GP_densities_normalised[:, None], metric="sqeuclidean")
            # plt.figure(2, figsize=(5, 5))
            # ot.plot.plot1D_mat(GP_densities_normalised, hist_positions, cost_matrix, "Cost matrix M")
            # plt.show()
            #output = self(self.train_x)
            # Calc loss and backprop gradients
            #loss = ot.sinkhorn2(self.bin_weights, continuous_weights, cost_matrix, 1e-3,verbose=False)
            #loss = torch.nn.functional.kl_div(hist_positions,torch.tensor(GP_densities_normalised))
            weights_transformed = softmax2(weights,10000)
            #weights_transformed = weights_transformed/weights_transformed.sum()

            # Normalize weights so they sum to 1
            #weights_transformed /= weights_transformed.sum()
            new_bin_dens = torch.histc(self.samples, bins=self.nbins)
            #new_bin_dens = torch.tensor(new_bin_dens, dtype=torch.float32)
            #loss  = custom_mse_loss(new_bin_dens,torch.tensor(GP_densities_normalised))
            
            new_bin_dens = new_bin_dens.float() / new_bin_dens.sum()  # Normalize
            new_bin_dens *= weights_transformed  # Apply weights
            loss = torch.nn.functional.mse_loss(new_bin_dens, GP_densities_normalised)
            #print(loss)
            loss.backward()
            optimizer.step()
            if i%100==0:
                if loss.item()<threshold:
                    print("Stopping Criteria Me at loss = "+ str(loss.item()))
                    break
                # self.update(weights.detach())
                # self.plot()
                
            
            
            #hist_positions.data = new_bin_dens
        self.update(weights.detach())
        # weights = weights/(torch.sum(weights)*500)
        # print(weights)
        # print(sum(weights))
        # self.bin_weights = weights_transformed.detach().numpy()
        # self.sample_weights = self.bin_weights[self.bindex - 1]
        # self.bin_dens = new_bin_dens.detach().numpy()

    def sinkhorn(self,n=1000):
        evaluation_points = torch.linspace(min(self.samples), max(self.samples), n)
        GP_densities_normalised = self.GP_model.eval_mean_vec(evaluation_points)
        GP_densities_normalised = GP_densities_normalised/sum(GP_densities_normalised)
        print(sum(GP_densities_normalised))
        hist_dens = self.bin_weights*self.samples.numpy()
        hist_dens = hist_dens/sum(hist_dens)
        print(sum(hist_dens))
        cost_matrix = np.outer(GP_densities_normalised,hist_dens)
        cost_matrix /= cost_matrix.max()
        plt.figure(1, figsize=(6.4, 3))
        plt.plot(evaluation_points, GP_densities_normalised, "b", label="Source distribution")
        plt.plot(np.linspace(0,1,len(self.samples)), hist_dens, "r", label="Target distribution")
        plt.legend()

        plt.figure(2, figsize=(5, 5))
        ot.plot.plot1D_mat(GP_densities_normalised, hist_dens, cost_matrix, "Cost matrix M")

        lambd = 1e-3
        Gs = ot.sinkhorn(GP_densities_normalised,hist_dens, cost_matrix, lambd, verbose=False)
        plt.figure(4, figsize=(5, 5))
        ot.plot.plot1D_mat(GP_densities_normalised,hist_dens, Gs, "OT matrix Sinkhorn")
        for i in range(0,5):
            val = ot.sinkhorn2(GP_densities_normalised,hist_dens, cost_matrix, lambd, verbose=False)
            print(val)
        plt.show()
    
    def sinkhorn2(self,n=1000):
        evaluation_points = torch.linspace(min(self.samples), max(self.samples), n)
        GP_densities_normalised = self.GP_model.eval_mean_vec(evaluation_points)
        GP_densities_normalised = GP_densities_normalised/sum(GP_densities_normalised)

        
        hist_dens = self.bin_dens/sum(self.bin_dens)
        
        print(sum(hist_dens))
        cost_matrix = np.outer(GP_densities_normalised,hist_dens)
        cost_matrix /= cost_matrix.max()
        plt.figure(1, figsize=(6.4, 3))
        plt.plot(evaluation_points, GP_densities_normalised, "b", label="Source distribution")
        plt.plot(np.linspace(0,1,len(self.bin_dens)), hist_dens, "r", label="Target distribution")
        plt.legend()

        plt.figure(2, figsize=(5, 5))
        ot.plot.plot1D_mat(GP_densities_normalised, hist_dens, cost_matrix, "Cost matrix M")

        lambd = 1e-3
        Gs = ot.sinkhorn(GP_densities_normalised,hist_dens, cost_matrix, lambd, verbose=False)
        plt.figure(4, figsize=(5, 5))
        ot.plot.plot1D_mat(GP_densities_normalised,hist_dens, Gs, "OT matrix Sinkhorn")
        for i in range(0,1):
            val = ot.sinkhorn2(GP_densities_normalised,hist_dens, cost_matrix, lambd, verbose=False)
            print(val)
        plt.show()     
                
        
        

    

    
    
        
    