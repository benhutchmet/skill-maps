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
                - 'start_year' (int): start year of the forecast
                - 'end_year' (int): end year of the forecast
    
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the dictionary
    values = {
        'nens1': mdi,
        'nens2': mdi,
        'start_year': mdi,
        'end_year': mdi
    }

    # Extract the values from the txt file
    files = glob.glob(f'{path}/*.txt')

    # Print the files
    print(files)

    # find the file containing nens1
    nens1_file = [file for file in files if 'nens1' in file][0]
    # print(nens1_file)

    # find the file containing nens2
    nens2_file = [file for file in files if 'nens2' in file][0]
    # print(nens2_file)

    # find the file containing start_year
    start_end_file = [file for file in files if 'start_end_years' in file][0]
    # print(start_end_file)

    # Load the values from the files
    values['nens1'] = np.loadtxt(nens1_file).astype(int)
    values['nens2'] = np.loadtxt(nens2_file).astype(int)
    values['start_year'] = (np.loadtxt(start_end_file))[0].astype(int)
    values['end_year'] = (np.loadtxt(start_end_file))[1].astype(int)

    # Return the values
    return values

# Now define a function to extract the numpy arrays within the path
def load_arrays_from_npy(path, variable, timeseries=False):
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
                - 'sigo (np.ndarray)': standard deviation of the observations
                - 'sigo_resid (np.ndarray)': standard deviation of the
                                                observed residuals
                - 'fcst1_ts (np.ndarray)': timeseries of the first ensemble
                                            (initialized)
                - 'fcst2_ts (np.ndarray)': timeseries of the second ensemble
                                            (uninitialized)
                - 'obs_ts (np.ndarray)': timeseries of the observations
                - 'fcst1_em_resid (np.ndarray)': residuals of the first
                                                    ensemble (initialized)
                - 'obs_resid (np.ndarray)': residuals of the observations
                - 'corr_diff (np.ndarray)': difference in correlation between
                                                the first ensemble (initialized)
                                                and the second ensemble
                                                (uninitialized)
                - 'corr_diff_p (np.ndarray)': p-value of the difference in
                                                correlation between the first
                                                ensemble (initialized) and the
                                                second ensemble (uninitialized)
            
    """

    # Set up the dictionary
    arrays = {
        'corr1': None,
        'corr1_p': None,
        'partial_r': None,
        'partial_r_p': None,
        'sigo': None,
        'sigo_resid': None,
        'fcst1_ts': None,
        'fcst2_ts': None,
        'obs_ts': None,
        'fcst1_em_resid': None,
        'obs_resid': None,
        'corr_diff': None,
        'corr_diff_p': None
    }

    # extract the .npy files using glob
    files = glob.glob(f'{path}/*.npy')

    # Print the files
    print(files)

    # find the file containing corr1
    corr1_file = [file for file in files if f'corr1_{variable}' in file][0]
    # print("corr1_file: ", corr1_file)

    # find the file containing corr1_p
    corr1_p_file = [file for file in files if 'corr1_p' in file][0]

    # find the file containing partial_r
    partial_r_file = [file for file in files if f'partial_r_{variable}'
                        in file][0]

    # find the file containing partial_r_p
    partial_r_p_file = [file for file in files if 'partial_r_p' in file][0]

    # # find the file containing sigo
    # sigo_file = [file for file in files if f'sigo_{variable}' in file][0]

    # # find the file containing sigo_resid
    # sigo_resid_file = [file for file in files if 'sigo_resid' in file][0]

    # find the file containing fcst1_ts
    # FIXME: not processed for 1000 bootstrap run - comment out for now
    if timeseries:
        fcst1_ts_file = [file for file in files if f'fcst1_ts_{variable}' in file][0]

        # find the file containing fcst2_ts
        fcst2_ts_file = [file for file in files if f'fcst2_ts_{variable}' in file][0]

        # find the file containing obs_ts
        obs_ts_file = [file for file in files if f'obs_ts_{variable}' in file][0]

    # find the file containing fcst1_em_resid
    fcst1_em_resid_file = [file for file in files if f'fcst1_em_resid_{variable}' in file][0]

    # find the file containing obs_resid
    obs_resid_file = [file for file in files if f'obs_resid_{variable}' in file][0]

    # find the file containing corr_diff
    corr_diff_file = [file for file in files if f'corr_diff_{variable}' in file][0]

    # find the file containing corr_diff_p
    corr_diff_p_file = [file for file in files if f'corr_diff_p_{variable}' in file][0]

    # Load the arrays from the files
    arrays['corr1'] = np.load(corr1_file)
    arrays['corr1_p'] = np.load(corr1_p_file)
    arrays['partial_r'] = np.load(partial_r_file)
    arrays['partial_r_p'] = np.load(partial_r_p_file)
    # arrays['sigo'] = np.load(sigo_file)
    # arrays['sigo_resid'] = np.load(sigo_resid_file)

    # FIXME: not processed for 1000 bootstrap run - comment out for now
    if timeseries:
        arrays['fcst1_ts'] = np.load(fcst1_ts_file)
        arrays['fcst2_ts'] = np.load(fcst2_ts_file)
        arrays['obs_ts'] = np.load(obs_ts_file)
    arrays['fcst1_em_resid'] = np.load(fcst1_em_resid_file)
    arrays['obs_resid'] = np.load(obs_resid_file)
    arrays['corr_diff'] = np.load(corr_diff_file)
    arrays['corr_diff_p'] = np.load(corr_diff_p_file)

    # Return the arrays
    return arrays

# Now define a function to calculate the benefit of initialization
# As the ratio of the predicted signal arising from initialization
# divided by the total predicted signal
def calculate_init_benefit(partial_r, sigo_resid, corr1, sigo):
    """
    Calculates the benefit of initialization as the ratio of the predicted
    signal arising from initialization divided by the total predicted signal.
    
    Numerator is the predicted signal arising from initialization:
        - partial_r * sigo_resid

    Denominator is the total predicted signal:
        - corr1 * sigo

    Args:
        partial_r (np.ndarray): bias corrected correlation of the first 
                                    ensemble (initialized) with the 
                                    observations (after accounting for 
                                    uninitialized trend)
        sigo_resid (float): standard deviation of the observed residuals
        corr1 (np.ndarray): correlation of the first ensemble (initialized)
                                with the observations
        sigo (float): standard deviation of the observations

    Returns:
        init_impact (np.ndarray): benefit of initialization as the ratio of
                                    the predicted signal arising from 
                                    initialization divided by the total 
                                    predicted signal
    """

    # Extract the nlats from the partial_r
    nlats = partial_r.shape[0]
    
    # Extract the nlons from the partial_r
    nlons = partial_r.shape[1]

    # Set up a new array for the init_impact
    init_impact = np.zeros([nlats, nlons])

    # Loop over the lats
    for lat in range(nlats):
        for lon in range(nlons):
            # extract the values
            partial_r_cell = partial_r[lat, lon]

            sigo_resid_cell = sigo_resid[lat, lon]

            corr1_cell = corr1[lat, lon]

            sigo_cell = sigo[lat, lon]

            pred_sig_init = partial_r_cell * sigo_resid_cell

            total_pred_sig = corr1_cell * sigo_cell

            # Calculate the init_impact
            init_impact[lat, lon] = pred_sig_init / total_pred_sig

    # Return the benefit of initialization
    return init_impact