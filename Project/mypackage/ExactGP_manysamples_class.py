# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:54:39 2024

@author: r96325jj
"""




from numpy.polynomial import Chebyshev
import gpytorch
import torch
from .maximums import GP_logsumexp, GP_hardmax, hardmax, softmax
import numpy as np
import matplotlib.pyplot as plt
from .ExactGP_class import ExactGPModel
from scipy.integrate import quad


class ExactGP_manysample(ExactGPModel):
    def __init__(self, full_train_x, full_train_y, likelihood, epsilon=0.001, k=0.1):
        #Making sure samples are unique, repeated samples don't improve gp model
        train_x, unique_index = np.unique(full_train_x.numpy(), return_index=True) 
        train_y = full_train_y.numpy()[unique_index]
        #TODO: No more random choice? Just doesn't seem to work well
        #TODO: new optimize function which uses full_train so only have to do it once
        #Choose some random initial samples
        init_samples = np.random.choice(range(0,len(train_x)),size=100,replace=False)
        init_x = train_x[init_samples]
        init_y = train_y[init_samples]
        #delete initial samples from vectors to choose new samples from
        train_x = np.delete(train_x,init_samples)
        train_y = np.delete(train_y,init_samples)
        #define intial model and optimize 'fully'
        model = ExactGPModel(torch.tensor(init_x,dtype = torch.float32),torch.tensor(init_y,dtype = torch.float32),likelihood)
        model.optimize(200)
        #Save hyperparameters to imrpove optimisation speed when new samples added
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        
        for i in range(0,len(train_x)):
            #evaluate aquisition function and find maximal sample
            mean,std = model.evaluate(train_x,limit='logsumexp')
            improvement = abs(mean-train_y)+k*std
            ind = np.argmax(improvement)
            # plt.plot(train_x,improvement)
            # plt.scatter(train_x[ind],improvement[ind],marker='x',color = 'green')
            # plt.show()
            #remove samples with low aquisition values
            low_improvement = np.where(improvement<epsilon)
            remove = np.append(low_improvement, ind)
            
            #define new model with added samples
            init_x = np.append(init_x,train_x[ind])
            init_y = np.append(init_y,train_y[ind])
            model = ExactGPModel(torch.tensor(init_x,dtype = torch.float32),torch.tensor(init_y,dtype = torch.float32),likelihood)
            #set hyperparamaters to saved values and do a few more optimization steps
            model.covar_module.base_kernel.lengthscale = lengthscale
            model.covar_module.outputscale = outputscale
            model.likelihood.noise = 0.0
            model.optimize(5)
            #save new optimized hyperparameters
            lengthscale = model.covar_module.base_kernel.lengthscale.item()
            outputscale = model.covar_module.outputscale.item()
            
            if max(improvement)<epsilon:
                print(i)
                model.optimize(100)
                print(model)
                #model.plot(limit = 'logsumexp')
                break   
            train_x = np.delete(train_x,remove)
            train_y = np.delete(train_y,remove)
            if len(train_x)<=0:
                print(i)
                model.optimize(100)
                print(model)
                #model.plot(limit = 'logsumexp')
                break                
        
        #define final model to be used and save features to be used in other functions
        inducing_points_x = torch.tensor(init_x)
        inducing_points_y = torch.tensor(init_y)
        super(ExactGPModel, self).__init__(inducing_points_x,inducing_points_y , likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = inducing_points_x
        self.train_y = inducing_points_y
        self.full_train_x = full_train_x.clone().detach()
        self.full_train_y = full_train_y.clone().detach()
        self.likelihood = likelihood
        self.likelihood.noise = 0.0
        self.covar_module.base_kernel.lengthscale =  model.covar_module.base_kernel.lengthscale.item()
        self.covar_module.outputscale =  model.covar_module.outputscale.item()


            
    def plot(self,points='inducing',limit = 'none',**kwargs):#xlab = 'x',ylab = 'y', title = ''):
        self.eval()
        self.likelihood.eval()
        #for speed don't calculate gradients and predict the gp at chosen points
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            xrange = torch.linspace(min(self.train_x), max(self.train_x), 1000,dtype=torch.float32)
            observed_pred = self(xrange)
        #split into vectors for plotting
        mean = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.numpy()
        upper = upper.numpy()
        #choose what limit to use usually to prevent negative values
        if limit == 'logsumexp':
            mean,upper,lower = GP_logsumexp(mean,upper,lower,10000)
        if limit =='hardmax':
            mean,upper,lower = GP_hardmax(mean,upper,lower)
        #plot GP
        with torch.no_grad():
            f, ax = plt.subplots(1, 1)
            ax.grid()
            if points == 'inducing':
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'x',color='green')
            elif points == 'full':
                ax.plot(self.full_train_x.numpy(), self.full_train_y.numpy(), 'x', color = 'green')
            ax.plot(xrange, mean, 'b')
            ax.fill_between(xrange, lower, upper, alpha=0.4,color='orange')
            ax.legend(['Observed Data','Mean', 'Confidence'])
            ax.set_xlabel(kwargs.get("ylabel", "y"))
            ax.set_ylabel(kwargs.get("xlabel", "x"))
            ax.set_title(kwargs.get("title",""))
            plt.show()
   
            
    def quadrature(self,lower_lim=-1,upper_lim=1):
        #define gp mean as function and calculate integral
        pdf = lambda x: self.eval_mean(x,limit='logsumexp')
        prob, error = quad(pdf, lower_lim,upper_lim)
        return prob, error
    
    def binfitter(self,bins = 50,plot=False):
        bin_dens, bin_edges=np.histogram(self.train_x.numpy(),density = True,range=(min(self.full_train_x.numpy())-1e-5,max(self.full_train_x.numpy())+1e-5),bins=bins)
        #i=16        
        #print((bin_dens*200*np.diff(bin_edges))[i])
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        GP_heights = self.eval_mean_vec(bin_midpoints,limit='logsumexp')
        bindex = np.digitize(self.full_train_x.numpy(),bin_edges)
        weights = GP_heights/bin_dens
        div_zero = np.where(np.isinf(weights))
        weights[div_zero] = 0.0
        #print(weights)
        #print(len(weights))
        weights = weights[bindex-1]

        # print(bin_dens[i]*weights[i])
        # print(weights[i]/(0.48884964*np.sum(weights)))
        # print(GP_heights[i]*200/np.sum(weights)
        new_bin_dens,new_bin_edges=np.histogram(self.full_train_x.numpy(), density = True, weights = weights,bins=bins,range=(min(self.full_train_x.numpy())-1e-5,max(self.full_train_x.numpy())+1e-5))
        new_bin_midpoints = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
        #TODO: check if below actually does anything
        print(sum(new_bin_dens*np.diff(new_bin_edges)))
        new_bin_dens = new_bin_dens*np.sum(weights)/len(self.full_train_x.numpy())
        print(sum(new_bin_dens*np.diff(new_bin_edges)))
        if plot==True:
            #plt.hist(self.train_x.numpy(), density=True,weights = weights,color = 'green',alpha = 0.7,bins=bins)
            plt.bar(new_bin_midpoints,new_bin_dens,width = np.diff(new_bin_edges),color='green',alpha=0.5)
        #print(new_bin_dens[i])
        print(np.sum(abs(GP_heights - new_bin_dens)))
        self.bin_dens = new_bin_dens
        self.bin_edges = new_bin_edges
        self.bin_midpoints = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2       
        
        #return new_bin_dens, new_bin_edges
    
    
    def histogram(self, bins):
        bin_dens, bin_edges=np.histogram(self.full_train_x.numpy(),density = True,range=(min(self.full_train_x.numpy())-1e-5,max(self.full_train_x.numpy())+1e-5),bins=bins)
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





###################################################################################################        
###################################################################################################



            
            
class ExactGP_manysample2(ExactGPModel):
    def __init__(self, full_train_x, full_train_y, likelihood, epsilon, k):
        np.random.seed(42) #TODO: remove this
        train_x, unique_index = np.unique(full_train_x.numpy(), return_index=True) 
        train_y = full_train_y.numpy()[unique_index]
        print(len(train_x))
        #TODO: No more random choice?
        #TODO: new optimize function which uses full_train so only have to do it once
        init_samples = np.random.choice(range(0,len(train_x)),size=100,replace=False)
        init_x = train_x[init_samples]
        init_y = train_y[init_samples]
        train_x = np.delete(train_x,init_samples)
        train_y = np.delete(train_y,init_samples)
        
        model = ExactGPModel(torch.tensor(init_x,dtype = torch.float32),torch.tensor(init_y,dtype = torch.float32),likelihood)
        model.set_train_data(full_train_x,full_train_y,strict = False)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        # Optimization loop
        for i in range(100):
            optimizer.zero_grad()
            output = model(full_train_x)
            loss = -mll(output,full_train_y)
            loss.backward()
            optimizer.step()
            
        model.plot(limit = 'logsumexp')
        plt.show()
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        outputscale = model.covar_module.outputscale.item()
        
        for i in range(0,len(train_x)):
            mean,std = model.evaluate(train_x,limit='logsumexp')
            improvement = abs(mean-train_y)+k*std
            ind = np.argmax(improvement)
            # plt.plot(train_x,improvement)
            # plt.scatter(train_x[ind],improvement[ind],marker='x',color = 'green')
            # plt.show()
            low_improvement = np.where(improvement<epsilon)
            #print(len(improvement))
            print(len(train_x),len(low_improvement[0]))
            
            remove = np.append(low_improvement, ind)
            #ind = np.argsort(improvement)[-100:]
            init_x = np.append(init_x,train_x[ind])
            init_y = np.append(init_y,train_y[ind])
            model = ExactGPModel(torch.tensor(init_x,dtype = torch.float32),torch.tensor(init_y,dtype = torch.float32),likelihood)
            #model.optimize(200)
            model.covar_module.base_kernel.lengthscale = lengthscale
            model.covar_module.outputscale = outputscale
            model.likelihood.noise = 0.0
            #model.optimize(5)
            #model.plot()
            # lengthscale = model.covar_module.base_kernel.lengthscale.item()
            # outputscale = model.covar_module.outputscale.item()
            if max(improvement)<epsilon:
                print(i)
                #model.optimize(100)
                print(model)
                model.plot(limit = 'logsumexp')
                break   
            train_x = np.delete(train_x,remove)
            train_y = np.delete(train_y,remove)
            #print(max(improvement))
            if len(train_x)<=0:
                print(i)
                #model.optimize(100)
                print(model)
                model.plot(limit = 'logsumexp')
                break                

        inducing_points_x = torch.tensor(init_x)
        inducing_points_y = torch.tensor(init_y)
        super(ExactGPModel, self).__init__(inducing_points_x,inducing_points_y , likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.train_x = inducing_points_x
        self.train_y = inducing_points_y
        self.full_train_x = full_train_x.clone().detach()
        self.full_train_y = full_train_y.clone().detach()
        self.likelihood = likelihood
        self.likelihood.noise = 0.0
        self.covar_module.base_kernel.lengthscale =  model.covar_module.base_kernel.lengthscale.item()
        self.covar_module.outputscale =  model.covar_module.outputscale.item()



    def plot(self,points = 'inducing',limit = 'none'):
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
            if points == 'inducing':
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'x',color='green')
            elif points == 'full':
                ax.plot(self.full_train_x.numpy(), self.full_train_y.numpy(), 'x', color = 'green')
            ax.plot(xrange, mean, 'b')
            ax.fill_between(xrange, lower, upper, alpha=0.4,color='orange')
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
   
            
    def quadrature(self,lower_lim=-1,upper_lim=1):
        pdf = lambda x: self.eval_mean(x,limit='logsumexp')
        prob, error = quad(pdf, lower_lim,upper_lim)
        return prob, error
        
    # def density(self,lower_lim,upper_lim):
    #     if hasattr(self,'weights'):
    #         print(self.weights)
    #         print(self.bin_widths)
    #         print(np.sum(self.weights*self.bin_widths))
    #         indices = np.where((self.bin_edges[:-1] >= lower_lim) & (self.bin_edges[1:] <= upper_lim))[0]
    #         print(indices)
    #         # Compute integral for the range
    #         partial_integral = np.sum(self.weights[indices] * self.bin_widths[indices])
    #         print(partial_integral)
    #     else:
    #         self.binfitter()
    #         self.density(lower_lim,upper_lim)
    
    def optimize(self,model, full_train_x, full_train_y, training_iter=100, output = False):
        model.train()
        model.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        if output ==True:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(full_train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, full_train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %f   lengthscale: %f  outputscale: %f  noise: %f' % (
                    i + 1, training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.covar_module.outputscale.item(),
                    model.likelihood.noise.item(),
                ))
                optimizer.step()
                
            
        else:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(full_train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, full_train_y)
                loss.backward()
                optimizer.step()            
        return model.covar_module.base_kernel.lengthscale.item(),model.covar_module.outputscale.item(),model.likelihood.noise.item()

        
            
            