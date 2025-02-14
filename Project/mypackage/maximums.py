# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:30:14 2024

@author: r96325jj
"""


from scipy.special import logsumexp
import numpy as np
import torch


def GP_logsumexp(mean,upper,lower,beta):
    for i in range(0,len(mean)):
        lower[i] = 1/beta*logsumexp([beta*lower[i],0])
        upper[i] = 1/beta*logsumexp([beta*upper[i],0])
        mean[i] = 1/beta*logsumexp([mean[i]*beta,0])
    return mean, upper, lower

def GP_hardmax(mean,upper,lower):#TODO: numpy.maximum with np.zeros???
    for i in range(0,len(mean)):
        lower[i] = max(lower[i],0)
        upper[i] = max(upper[i],0)
        mean[i] = max(mean[i],0)
    return mean, upper, lower



def hardmax(x):
    return max(x,np.zeros(len(x)))

def softmax(x,beta):
    for i in range(0,len(x)):
        x[i] = 1/beta*logsumexp([beta*x[i],0])
    return x


def softmax2(x,beta):
    return (1 / beta) * torch.special.logsumexp(torch.stack([beta * x, torch.zeros_like(x)]), dim=0)


    
        


