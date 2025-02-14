# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:17:01 2024

@author: r96325jj
"""
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import csv

def normal_pdf(x, mean, variance):
    """Calculate the probability density function (PDF) of a normal distribution."""
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

def multimodal_pdf(x, mean1, variance1, mean2, variance2, weight1=0.5):
    """Calculate the PDF of a multimodal distribution formed by two normal distributions."""
    weight2 = 1 - weight1  # Ensure the weights sum to 1
    pdf_1 = normal_pdf(x, mean1, variance1)  # First normal distribution
    pdf_2 = normal_pdf(x, mean2, variance2)  # Second normal distribution
    return weight1 * pdf_1 + weight2 * pdf_2  # Weighted sum of the two PDFs





def write_vector(vector, vector_name):
    """
    Writes each element of a NumPy vector to a .txt file.
    The file name will be based on the provided vector_name.
    
    :param vector: The NumPy array to be written to file.
    :param vector_name: The name of the vector to create the filename.
    """
    file_name = f"{vector_name}.txt"
    
    # Use NumPy's savetxt to write the vector to a file
    np.savetxt(file_name, vector, fmt='%s')
    
    print(f"Vector written to {file_name}")
# Parameters for the two normal distributions
mean1 = 0.0
variance1 = 1.0

mean2 = 25
variance2 = 2.0

# Range of x values
x_vals = np.linspace(-20, 60, 3000)

# Calculate the multimodal PDF
multimodal_values = multimodal_pdf(x_vals, mean1, variance1, mean2, variance2, weight1=0.5)

# Plot the multimodal PDF
#plt.plot(x_vals, multimodal_values, label="Multimodal PDF", color="blue")




def mean_RWMH(func,start_point = 0, beta = 2.3, iterations = 50000, full = True):

    samples = np.zeros(iterations)
    samples[0] = start_point
    current_sample = start_point
    #accep_prob = np.zeros(iterations)
    acep = 0
    densities = np.zeros(iterations)
    densities[0] = func(start_point)
    ind_accep = np.array(0)
    for i in range(1,iterations):
        proposal_sample = np.random.normal(current_sample, beta*beta)
        U = np.random.uniform(0,1)
        proposal_density = func(proposal_sample)
        current_density = func(current_sample)
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


# d,t = mean_RWMH(target_func,beta=2.6,iterations = 50000)
# e,f = mean_RWMH(target_func,beta=2.6,iterations = 50000, full=False)
# plt.hist(d,density = True,bins = 100)
# plt.hist(e,density = True,bins=100)
# plt.scatter(d,t,color='green')
# plt.show()



def rwmh_normal_numpy(func,n_samples=1000, proposal_std = 2.3,start_point=0):
    # Initialize variables
    current = 0.0  # Start at x = 0
    accep_samples = np.zeros(n_samples)
    accep_samples[0] = start_point
    prop_points = np.zeros(n_samples)
    prop_points[0] = start_point
    prop_densities = np.zeros(n_samples)
    prop_densities[0] = func(start_point)
    accepted = 0

    for i in range(1,n_samples):
        # Propose a new sample
        proposal = np.random.normal(current, proposal_std**2)
        prop_points[i]= proposal

        # Compute densities
        current_density = func(current)
        proposal_density = func(proposal)
        prop_densities[i] = proposal_density
        # Compute acceptance probability
        acceptance_ratio = proposal_density / current_density
        acceptance_prob = min(1, acceptance_ratio)

        # Accept or reject the proposed sample
        if np.random.uniform(0, 1) < acceptance_prob:
            current = proposal  # Accept the proposal
            accepted += 1

        # Store the current state
        accep_samples[i] = current

    # Calculate the acceptance rate
    print(accepted / n_samples)
    
    return accep_samples, prop_points, prop_densities





"""
#samples, prop_points, prop_densities = rwmh_normal_numpy(target_func,50000,2.3,0)
samples, densities = rwmh(target_func,5.5,0,50000)
plt.hist(samples,density = True,bins = 100)
plt.scatter(samples,densities,color='green')
plt.show()
plt.plot(samples)

write_vector(samples, 'multimodal_samples')
write_vector(densities, 'multimodal_densities')
# plt.scatter(prop_points,prop_densities,marker = 'x',color = 'green')
# plt.plot(samples)
# plt.plot(prop_points)
# write_vector(prop_points,'normal_proposals5')
# write_vector(prop_densities,'normal_densities5')
# write_vector(samples, 'normal_accepted5')
"""







def rwmh(target_density, proposal_std, initial_point, num_samples, output_csv):
    """
    Perform Random Walk Metropolis-Hastings and save results to a CSV file.

    Parameters:
    - target_density: callable, target probability density function (not necessarily normalized).
    - proposal_std: float, standard deviation of the proposal distribution (Gaussian).
    - initial_point: float, starting point of the Markov Chain.
    - num_samples: int, number of samples to generate in the Markov Chain.
    - output_csv: str, name of the output CSV file to save results.

    Returns:
    - chain: np.ndarray, array of accepted samples.
    """
    # Initialize Markov Chain
    current_point = initial_point
    current_density = target_density(current_point)
    
    # Containers for results
    chain = [current_point]
    chain_densities = [current_density]
    proposals = [current_point]
    proposal_densities = [current_density]
    i=0
    for _ in range(num_samples):
        # Propose a new point
        proposed_point = np.random.normal(loc=current_point, scale=proposal_std)
        proposed_density = target_density(proposed_point)
        
        # Compute acceptance probability
        acceptance_ratio = proposed_density / current_density
        acceptance_prob = min(1, acceptance_ratio)
        
        # Accept or reject
        if np.random.rand() < acceptance_prob:
            # Accept the proposed point
            current_point = proposed_point
            current_density = proposed_density
            i+=1
        # Record results
        chain.append(current_point)
        chain_densities.append(current_density)
        proposals.append(proposed_point)
        proposal_densities.append(proposed_density)
    print(i/num_samples)
    plt.hist(chain,density=True,bins=100)
    plt.scatter(chain,chain_densities,color='green',marker='x')
    plt.show()
    plt.hist(proposals,density=True,bins=100)
    plt.scatter(proposals,proposal_densities,color='green',marker='x')
    #Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sample", "Density", "Proposed Sample", "Proposed Density"])
        for i in range(num_samples):
            writer.writerow([chain[i], chain_densities[i], proposals[i], proposal_densities[i]])
    
    return np.array(chain)

# Example Usage
target_func = partial(multimodal_pdf, mean1=mean1,variance1=variance1, mean2=mean2,variance2=variance2)
#target_func = partial(normal_pdf,mean=1,variance=1)

# Parameters
proposal_std = 5.5
initial_point = 0.0
num_samples = 50000
output_csv = "rwmh_results.csv"

# Run RWMH
chain = rwmh(target_func, proposal_std, initial_point, num_samples, output_csv)

