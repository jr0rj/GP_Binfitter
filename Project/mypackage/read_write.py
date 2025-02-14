# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:23:36 2024

@author: r96325jj
"""
import os
import numpy as np

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
    
    
    
def read_vector(file_name):
    """
    Reads the contents of a .txt file and outputs a NumPy vector (array).
    
    :param file_name: The name of the file to read the vector from.
    :return: NumPy array of elements read from the file.
    """
    if not os.path.exists(file_name):
        print(f"Error: {file_name} does not exist.")
        return None

    # Use NumPy's loadtxt to read the file into a NumPy array
    vector = np.loadtxt(file_name, dtype=float)
    
    return vector