"""
Functions which use xarray to see how the individual models perform in terms of NAO skill over the first and second winter.
"""

# Local imports
import os
import sys
import glob
import random

# Third party imports
import xesmf as xe
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats, signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Sys.path.append
sys.path.append('/home/users/benhutch/skill-maps/')

# Local imports
import dictionaries as dicts

# Define a function for preprocessing the model data
# TODO: Add regridding in here
def preprocess(ds: xr.Dataset,
               first_fcst_year_idx: int,
               last_fcst_year_idx: int,
               lat1: float,
               lat2: float,
               lon1: float,
               lon2: float,
               start_month: int = 1,
               end_month: int = 12):
    """
    Preprocess the model data using xarray
    """

    # Expand the dimensions of the dataset
    ds = ds.expand_dims('ensemble_member')

    # Set the ensemble member
    ds['ensemble_member'] = [ds.attrs['variant_label']]

    # Extract the years from the data
    years = ds.time.dt.year.values

    # Find the unique years
    unique_years = np.unique(years)

    # Extract the first year
    first_year = int(unique_years[first_fcst_year_idx])

    # Extract the last year
    last_year = int(unique_years[last_fcst_year_idx])

    # If the start or end month is a single digit
    if start_month < 10:
        start_month = f"0{start_month}"

    if end_month < 10:
        end_month = f"0{end_month}"

    # Form the strings for the start and end dates
    start_date = f"{first_year}-{start_month}-01" ; end_date = f"{last_year}-{end_month}-30"

    # Find the centre of the period between start and end date
    mid_date = pd.to_datetime(start_date) + (pd.to_datetime(end_date) - pd.to_datetime(start_date)) / 2

    # Take the mean over the time dimension
    ds = ds.sel(time=slice(start_date, end_date)).mean(dim='time')

    # Take the mean over the lat and lon dimensions
    ds = ds.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(dim=('lat', 'lon'))

    # Set the time to the mid date
    ds['time'] = mid_date

    # Return the dataset
    return ds

# Write a new function for loading the model data using xarray
def load_model_data_xarray(model_variable: str,
                           model: str,
                           experiment: str,
                           start_year: int,
                           end_year: int,
                           grid: dict,
                           first_fcst_year: int,
                           last_fcst_year: int,
                           start_month: int = 12,
                           end_month: int = 3,
                           csv_dir: str = "/home/users/benhutch/unseen_multi_year/paths/"):
    """
    Function for loading each of the ensemble members for a given model using xarray
    
    Parameters
    ----------
    
    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation
        
    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'
    
    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    first_fcst_year: int
        The first forecast year for taking the time average
        E.g. 1960

    last_fcst_year: int
        The last forecast year for taking the time average
        E.g. 1962

    start_month: int
        The start month for the time average
        E.g. 12 for December

    end_month: int
        The end month for the time average
        E.g. 3 for March

    csv_dir: str
        The directory containing the csv files with the paths to the model data

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Extract the lat and lon bounds
    lon1, lon2, lat1, lat2 = grid['lon1'], grid['lon2'], grid['lat1'], grid['lat2']

    # Set up the path to the csv file
    csv_path = f"{csv_dir}/*.csv"

    # Find the csv file
    csv_file = glob.glob(csv_path)[0]

    # Assert that the csv file exists
    assert os.path.exists(csv_file), "The csv file does not exist"

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the given model and experiment and variable
    model_path = csv_data.loc[(csv_data['model'] == model) & (csv_data['experiment'] == experiment) & (csv_data['variable'] == model_variable), 'path'].values[0]

    # # print the model path
    # print("model path:", model_path)
    
    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Extract the first part of the model_path
    model_path_root = model_path.split('/')[1]

    # If the model path root is gws
    if model_path_root in ['gws', 'work']:
        print("The model path root is gws")

        # List the files in the model path
        model_files = os.listdir(model_path)

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Find all of the files for the given year
            year_files = [file for file in model_files if f"s{year}" in file]

            # Split the year files by '/'
            year_files_split = [file.split('/')[-1] for file in year_files]

            # Split the year files by '_'
            year_files_split = [file.split('_')[4] for file in year_files_split]

            # Split the year files by '-'
            year_files_split = [file.split('-')[1] for file in year_files_split]

            # Find the unique combinations
            unique_combinations = np.unique(year_files_split)

            # Assert that the len unique combinations is the same as the no members
            assert len(unique_combinations) == len(year_files), "The number of unique combinations is not the same as the number of members"

    elif model_path_root == 'badc':
        print("The model path root is badc")

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # Extract the number of members
            # as the number of unique combinations of r*i*p?f?
            # here f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
            # List the directories in model_path
            dirs = os.listdir(model_path)

            # Split these by the delimiter '-'
            dirs_split = [dir.split('-') for dir in dirs]

            # Find the unique combinations of r*i*p?f?
            unique_combinations = np.unique(dirs_split)

            # Assert that the number of files is the same as the number of members
            assert len(year_files) == len(unique_combinations), "The number of files is not the same as the number of members"
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Set up unique variant labels
    unique_variant_labels = np.unique(unique_combinations)

    # Print the number of unique variant labels
    print("Number of unique variant labels:", len(unique_variant_labels))
    print("For model:", model)

    # print the unique variant labels
    print("Unique variant labels:", unique_variant_labels)

    # Create an empty list for forming the list of files for each ensemble member
    member_files = []

    # If the model path root is gws
    if model_path_root in ['gws', 'work']:
        print("Forming the list of files for each ensemble member for gws")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Find the file for the given year and member
                file = [file for file in model_files if f"s{year}" in file and variant_label in file][0]

                # Append the model path to the file
                file = f"{model_path}/{file}"

                # Append the file to the member files
                variant_label_files.append(file)

            # Append the member files to the member files
            member_files.append(variant_label_files)
    elif model_path_root == 'badc':
        print("Forming the list of files for each ensemble member for badc")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Form the path to the files for this year
                path = f"{model_path}/s{year}-r{variant_label}i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
    
                # Find the files which match the path
                year_files = glob.glob(path)

                # Assert that the number of files is 1
                assert len(year_files) == 1, "The number of files is not 1"

                # Append the file to the variant label files
                variant_label_files.append(year_files[0])

            # Append the variant label files to the member files
            member_files.append(variant_label_files)
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Assert that member files is a list withiin a list
    assert isinstance(member_files, list), "member_files is not a list"

    # Assert that member files is a list of lists
    assert isinstance(member_files[0], list), "member_files is not a list of lists"

    # Assert that the length of member files is the same as the number of unique variant labels
    assert len(member_files) == len(unique_variant_labels), "The length of member_files is not the same as the number of unique variant labels"

    # Initialize the model data
    dss = []

    # Find the index of the forecast first year
    first_fcst_year_idx = start_year - first_fcst_year ; last_fcst_year_idx = last_fcst_year - first_fcst_year

    # Loop over the member files
    for member_file in tqdm(member_files, desc="Processing members"):
        # print("Processing member:", member_file)

        # Open the files
        ds = xr.open_mfdataset(member_file,
                               preprocess=lambda ds: preprocess(ds, first_fcst_year_idx, last_fcst_year_idx, lat1, lat2, lon1, lon2, start_month, end_month),
                               combine='nested',
                               concat_dim='time',
                               join='override',
                               coords='minimal',
                               parallel=True)
        
        # Append the dataset to the model data
        dss.append(ds)

    # Concatenate the datasets
    ds = xr.concat(dss, dim='ensemble_member')

    # Return the model data
    return ds
