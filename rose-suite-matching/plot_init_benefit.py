#!/usr/bin/env python

"""
plot_init_benefit.py
--------------------

A script which plots a series of subplots.
Each row shows the total skill of the initialized ensemble mean and 
the benefit of the initialization relative to the uninitialized ensemble mean.

Usage:

    $ python plot_init_benefit.py <variable> <region> <season> <forecast_range>
                                <method> <no_bootstraps>
    Args:                            
        <variable>      :   variable to plot
        <region>        :   region to plot
        <season>        :   season to plot
        <forecast_range>:   forecast range to plot
        <method>        :   method to plot
        <no_bootstraps> :   number of bootstraps to plot

    Returns:
        None
                                
"""

# Import general Python modules
import argparse, os, sys, glob, re

# Import additional modules
import numpy as np

# Define a function to extract the values from the txt file
def extract_values_from_txt(path, variable):
    """
    Extract values from a txt file.

    Args:
        path (str): path to the txt file
        variable (str): variable to extract values for

    Returns:
        values (dict): dictionary of values
            Contains:
                - 'nens1' (int): number of ensemble members in the first 
                                    ensemble
                - 'nens2' (int): number of ensemble members in the second
                                    ensemble
                - 'sigo' (float): standard deviation of the observations
                - 'sigo_resid' (float): standard deviation of the observed
                                            residuals
                - 'start_year' (int): start year of the forecast
                - 'end_year' (int): end year of the forecast
    
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the dictionary
    values = {
        'nens1': mdi,
        'nens2': mdi,
        'sigo': mdi,
        'sigo_resid': mdi,
        'start_year': mdi,
        'end_year': mdi
    }

    # Extract the values from the txt file
    files = glob.glob(f'{path}/*.txt')

    # Print the files
    print(files)

    # find the file containing nens1
    nens1_file = [file for file in files if 'nens1' in file][0]
    print(nens1_file)

    # find the file containing nens2
    nens2_file = [file for file in files if 'nens2' in file][0]
    print(nens2_file)

    # find the file containing sigo
    sigo_file = [file for file in files if f'sigo_{variable}' in file][0]
    print(sigo_file)

    # find the file containing sigo_resid
    sigo_resid_file = [file for file in files if 'sigo_resid' in file][0]
    print(sigo_resid_file)

    # find the file containing start_year
    start_end_file = [file for file in files if 'start_end_years' in file][0]
    print(start_end_file)

    # Load the values from the files
    values['nens1'] = np.loadtxt(nens1_file).astype(int)
    values['nens2'] = np.loadtxt(nens2_file).astype(int)
    values['sigo'] = np.loadtxt(sigo_file)
    values['sigo_resid'] = np.loadtxt(sigo_resid_file)
    values['start_year'] = (np.loadtxt(start_end_file))[0].astype(int)
    values['end_year'] = (np.loadtxt(start_end_file))[1].astype(int)

    # Return the values
    return values

# Now define a function to extract the numpy arrays within the path
def load_arrays_from_npy(path, variable):
    """
    Loads the numpy arrays from a path into a dictionary.
    
    Args:
        path (str): path to the numpy arrays
        variable (str): variable to extract values for
        
    Returns:
        arrays (dict): dictionary of numpy arrays
            Contains:
                - 'corr1' (np.ndarray): correlation of the first ensemble
                                        (initialized) with the observations
                - 'corr1_p' (np.ndarray): p-value of the correlation of the
                                            first ensemble (initialized) with
                                            the observations
                - 'partial_r' (np.ndarray): the bias corrected correlation of 
                                            the first ensemble (initialized)
                                            with the observations (after 
                                            accounting for uninitialized trend)
                - 'partial_r_p' (np.ndarray): p-value of the bias corrected
                                                correlation of the first 
                                                ensemble (initialized) with
                                                the observations (after 
                                                accounting for uninitialized
                                                trend)
    
    """

    # Set up the dictionary
    arrays = {
        'corr1': None,
        'corr1_p': None,
        'partial_r': None,
        'partial_r_p': None
    }

    # extract the .npy files using glob
    files = glob.glob(f'{path}/*.npy')

    # Print the files
    print(files)

    # find the file containing corr1
    corr1_file = [file for file in files if f'corr1_{variable}' in file]

    # find the file containing corr1_p
    corr1_p_file = [file for file in files if 'corr1_p' in file]

    # find the file containing partial_r
    partial_r_file = [file for file in files if f'partial_r_{variable}' 
                      in file]

    # find the file containing partial_r_p
    partial_r_p_file = [file for file in files if 'partial_r_p' in file]

    # Load the arrays from the files
    arrays['corr1'] = np.load(corr1_file)
    arrays['corr1_p'] = np.load(corr1_p_file)
    arrays['partial_r'] = np.load(partial_r_file)
    arrays['partial_r_p'] = np.load(partial_r_p_file)

    # Return the arrays
    return arrays

