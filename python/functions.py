# functions for the main program
# these should be tested one by one
# before being used in the main program
#
# Usage: python functions.py <variable> <model> <region> <forecast_range> <season>
#
# Example: python functions.py "psl" "BCC-CSM2-MR" "north-atlantic" "2-5" "DJF"
#

# Imports
import argparse
import os
import sys
import glob
import re

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import scipy.stats as stats
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image

# Import CDO
from cdo import *
cdo = Cdo()


# Install imageio
# ! pip install imageio
import imageio.v3 as iio

# Set the path to imagemagick
rcParams['animation.convert_path'] = r'/usr/bin/convert'

# Local imports
sys.path.append('/home/users/benhutch/skill-maps')
import dictionaries as dic

# We want to write a function that takes a data directory and list of models
# which loads all of the individual ensemble members into a dictionary of datasets /
# grouped by models
# the arguments are:
# base_directory: the base directory where the data is stored
# models: a list of models to load
# variable: the variable to load, extracted from the command line
# region: the region to load, extracted from the command line
# forecast_range: the forecast range to load, extracted from the command line
# season: the season to load, extracted from the command line

def load_data(base_directory, models, variable, region, forecast_range, season):
    """Load the data from the base directory into a dictionary of datasets.
    
    This function takes a base directory and a list of models and loads
    all of the individual ensemble members into a dictionary of datasets
    grouped by models.
    
    Args:
        base_directory: The base directory where the data is stored.
        models: A list of models to load.
        variable: The variable to load, extracted from the command line.
        region: The region to load, extracted from the command line.
        forecast_range: The forecast range to load, extracted from the command line.
        season: The season to load, extracted from the command line.
        
    Returns:
        A dictionary of datasets grouped by models.
    """
    
    # Create an empty dictionary to store the datasets.
    datasets_by_model = {}
    
    # Loop over the models.
    for model in models:
        
        # Create an empty list to store the datasets for this model.
        datasets_by_model[model] = []
        
        # create the path to the files for this model
        files_path = base_directory + "/" + variable + "/" + model + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + "outputs" + "/" + "mergetime" + "/" + "*.nc"

        # #print the path to the files
        #print("Searching for files in ", files_path)

        # Create a list of the files for this model.
        files = glob.glob(files_path)

        # if the list of files is empty, #print a warning and
        # exit the program
        if len(files) == 0:
            #print("No files found for " + model)
            sys.exit()
        
        # #print the files to the screen.
        #print("Files for " + model + ":", files)

        # Loop over the files.
        for file in files:

            # #print the file to the screen.
            # #print(file)
            
            # check that the file exists
            # if it doesn't exist, #print a warning and
            # exit the program
            if not os.path.exists(file):
                #print("File " + file + " does not exist")
                sys.exit()

            # Load the dataset.
            dataset = xr.open_dataset(file, chunks = {"time":50})

            # Append the dataset to the list of datasets for this model.
            datasets_by_model[model].append(dataset)
            
    # Return the dictionary of datasets.
    return datasets_by_model

# Write a function to process the data
# this includes an outer function that takes datasets by model
# and an inner function that takes a single dataset
# the outer function loops over the models and calls the inner function
# the inner function processes the data for a single dataset
# by extracting the variable and the time dimension
def process_data(datasets_by_model, variable):
    """Process the data.
    
    This function takes a dictionary of datasets grouped by models
    and processes the data for each dataset.
    
    Args:
        datasets_by_model: A dictionary of datasets grouped by models.
        variable: The variable to load, extracted from the command line.
        
    Returns:
        variable_data_by_model: the data extracted for the variable for each model.
        model_time_by_model: the model time extracted from each model for each model.
    """
    
    #print(f"Dataset type: {type(datasets_by_model)}")

    def process_model_dataset(dataset, variable):
        """Process a single dataset.
        
        This function takes a single dataset and processes the data.
        
        Args:
            dataset: A single dataset.
            variable: The variable to load, extracted from the command line.
            
        Returns:
            variable_data: the extracted variable data for a single model.
            model_time: the extracted time data for a single model.
        """
        
        if variable == "psl":
           

            # #print the variable data
            # #print("Variable data: ", variable_data)
            # # #print the variable data type
            # #print("Variable data type: ", type(variable_data))

            # # #print the len of the variable data dimensions
            # #print("Variable data dimensions: ", len(variable_data.dims))
            
            # Convert from Pa to hPa.
            # Using try and except to catch any errors.
            try:
                # Extract the variable.
                variable_data = dataset["psl"]

                # #print the values of the variable data
                # #print("Variable data values: ", variable_data.values)

            except:
                #print("Error converting from Pa to hPa")
                sys.exit()

        elif variable == "tas":
            # Extract the variable.
            variable_data = dataset["tas"]
        elif variable == "rsds":
            # Extract the variable.
            variable_data = dataset["rsds"]
        elif variable == "sfcWind":
            # Extract the variable.
            variable_data = dataset["sfcWind"]
        else:
            #print("Variable " + variable + " not recognised")
            sys.exit()

        # If variable_data is empty, #print a warning and exit the program.
        if variable_data is None:
            #print("Variable " + variable + " not found in dataset")
            sys.exit()

        # Extract the time dimension.
        model_time = dataset["time"].values
        # Set the type for the time dimension.
        model_time = model_time.astype("datetime64[Y]")

        # If model_time is empty, #print a warning and exit the program.
        if model_time is None:
            #print("Time not found in dataset")
            sys.exit()

        return variable_data, model_time
    
    # Create empty dictionaries to store the processed data.
    variable_data_by_model = {}
    model_time_by_model = {}
    for model, datasets in datasets_by_model.items():
        try:
            # Create empty lists to store the processed data.
            variable_data_by_model[model] = []
            model_time_by_model[model] = []
            # Loop over the datasets for this model.
            for dataset in datasets:
                # Process the dataset.
                variable_data, model_time = process_model_dataset(dataset, variable)
                # Append the processed data to the lists.
                variable_data_by_model[model].append(variable_data)
                model_time_by_model[model].append(model_time)
        except Exception as e:
            #print(f"Error processing dataset for model {model}: {e}")
            #print("Exiting the program")
            sys.exit()

    # Return the processed data.
    return variable_data_by_model, model_time_by_model

# Functions to process the observations.
# Broken up into smaller functions.
# ---------------------------------------------
def check_file_exists(file_path):
    """
    Check if a file exists in the given file path.

    Parameters:
    file_path (str): The path of the file to be checked.

    Returns:
    None

    Raises:
    SystemExit: If the file does not exist in the given file path.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        #print(f"File {file_path} does not exist")
        sys.exit()

def regrid_observations(obs_dataset):
    """
    Regrids an input dataset of observations to a standard grid.

    Parameters:
    obs_dataset (xarray.Dataset): The input dataset of observations.

    Returns:
    xarray.Dataset: The regridded dataset of observations.

    Raises:
    SystemExit: If an error occurs during the regridding process.
    """
    try:

        regrid_example_dataset = xr.Dataset({
            "lon": (["lon"], np.arange(0.0, 359.9, 2.5)),
            "lat": (["lat"], np.arange(90.0, -90.1, -2.5)),
        })
        regridded_obs_dataset = obs_dataset.interp(
            lon=regrid_example_dataset.lon,
            lat=regrid_example_dataset.lat
        )
        return regridded_obs_dataset
    
    except Exception as e:
        #print(f"Error regridding observations: {e}")
        sys.exit()


def select_region(regridded_obs_dataset, region_grid):
    """
    Selects a region from a regridded observation dataset based on the given region grid.

    Parameters:
    regridded_obs_dataset (xarray.Dataset): The regridded observation dataset.
    region_grid (dict): A dictionary containing the region grid with keys 'lon1', 'lon2', 'lat1', and 'lat2'.

    Returns:
    xarray.Dataset: The regridded observation dataset for the selected region.

    Raises:
    SystemExit: If an error occurs during the region selection process.
    """
    try:

        # Echo the dimensions of the region grid
        #print(f"Region grid dimensions: {region_grid}")

        # Define lon1, lon2, lat1, lat2
        lon1, lon2 = region_grid['lon1'], region_grid['lon2']
        lat1, lat2 = region_grid['lat1'], region_grid['lat2']

        # dependent on whether this wraps around the prime meridian
        if lon1 < lon2:
            regridded_obs_dataset_region = regridded_obs_dataset.sel(
                lon=slice(lon1, lon2),
                lat=slice(lat1, lat2)
            )
        else:
            # If the region crosses the prime meridian, we need to do this in two steps
            # Select two slices and concatenate them together
            regridded_obs_dataset_region = xr.concat([
                regridded_obs_dataset.sel(
                    lon=slice(0, lon2),
                    lat=slice(lat1, lat2)
                ),
                regridded_obs_dataset.sel(
                    lon=slice(lon1, 360),
                    lat=slice(lat1, lat2)
                )
            ], dim='lon')

        return regridded_obs_dataset_region
    except Exception as e:
        #print(f"Error selecting region: {e}")
        sys.exit()

# Using cdo to do the regridding and selecting the region
def regrid_and_select_region(observations_path, region, obs_var_name):
    """
    Uses CDO remapbil and a gridspec file to regrid and select the correct region for the obs dataset. Loads for the specified variable.
    
    Parameters:
    observations_path (str): The path to the observations dataset.
    region (str): The region to select.

    Returns:
    xarray.Dataset: The regridded and selected observations dataset.
    """
    
    # First choose the gridspec file based on the region
    gridspec_path = "/home/users/benhutch/gridspec"

    # select the correct gridspec file
    if region == "north-atlantic":
        gridspec = gridspec_path + "/" + "gridspec-north-atlantic.txt"
    elif region == "global":
        gridspec = gridspec_path + "/" + "gridspec-global.txt"
    elif region == "azores":
        gridspec = gridspec_path + "/" + "gridspec-azores.txt"
    elif region == "iceland":
        gridspec = gridspec_path + "/" + "gridspec-iceland.txt"
    else:
        print("Invalid region")
        sys.exit()

    # Check that the gridspec file exists
    if not os.path.exists(gridspec):
        print("Gridspec file does not exist")
        sys.exit()

    # Create the output file path
    regrid_sel_region_file = "/home/users/benhutch/ERA5/" + region + "_" + "regrid_sel_region.nc"

    # Check if the output file already exists
    # If it does, then exit the program
    if os.path.exists(regrid_sel_region_file):
        print("File already exists")
        print("Loading ERA5 data")
    else:
        print("File does not exist")
        print("Processing ERA5 data using CDO")

        # Regrid and select the region using cdo 
        cdo.remapbil(gridspec, input=observations_path, output=regrid_sel_region_file)

    # Load the regridded and selected region dataset
    # for the provided variable
    # check whether the variable name is valid
    if obs_var_name not in ["psl", "tas", "sfcWind", "rsds", "tos"]:
        print("Invalid variable name")
        sys.exit()

    # Translate the variable name to the name used in the obs dataset
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    else:
        print("Invalid variable name")
        sys.exit()

    # Load the regridded and selected region dataset
    # for the provided variable
    try:
        # Load the dataset for the selected variable
        regrid_sel_region_dataset = xr.open_mfdataset(regrid_sel_region_file, combine='by_coords', parallel=True, chunks={"time": 100, 'lat': 100, 'lon': 100})[obs_var_name]

        # Combine the two expver variables
        regrid_sel_region_dataset_combine = regrid_sel_region_dataset.sel(expver=1).combine_first(regrid_sel_region_dataset.sel(expver=5))

        return regrid_sel_region_dataset_combine

    except Exception as e:
        print(f"Error loading regridded and selected region dataset: {e}")
        sys.exit()


def select_season(regridded_obs_dataset_region, season):
    """
    Selects a season from a regridded observation dataset based on the given season string.

    Parameters:
    regridded_obs_dataset_region (xarray.Dataset): The regridded observation dataset for the selected region.
    season (str): A string representing the season to select. Valid values are "DJF", "MAM", "JJA", "SON", "SOND", "NDJF", and "DJFM".

    Returns:
    xarray.Dataset: The regridded observation dataset for the selected season.

    Raises:
    ValueError: If an invalid season string is provided.
    """

    try:
        # Extract the months from the season string
        if season == "DJF":
            months = [12, 1, 2]
        elif season == "MAM":
            months = [3, 4, 5]
        elif season == "JJA":
            months = [6, 7, 8]
        elif season == "JJAS":
            months = [6, 7, 8, 9]
        elif season == "SON":
            months = [9, 10, 11]
        elif season == "SOND":
            months = [9, 10, 11, 12]
        elif season == "NDJF":
            months = [11, 12, 1, 2]
        elif season == "DJFM":
            months = [12, 1, 2, 3]
        else:
            raise ValueError("Invalid season")

        # Select the months from the dataset
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(
            time=regridded_obs_dataset_region["time.month"].isin(months)
        )

        return regridded_obs_dataset_region_season
    except:
        #print("Error selecting season")
        sys.exit()

def calculate_anomalies(regridded_obs_dataset_region_season):
    """
    Calculates the anomalies for a given regridded observation dataset for a specific season.

    Parameters:
    regridded_obs_dataset_region_season (xarray.Dataset): The regridded observation dataset for the selected region and season.

    Returns:
    xarray.Dataset: The anomalies for the given regridded observation dataset.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        obs_climatology = regridded_obs_dataset_region_season.mean("time")
        obs_anomalies = regridded_obs_dataset_region_season - obs_climatology
        return obs_anomalies
    except:
        #print("Error calculating anomalies for observations")
        sys.exit()

def calculate_annual_mean_anomalies(obs_anomalies, season):
    """
    Calculates the annual mean anomalies for a given observation dataset and season.

    Parameters:
    obs_anomalies (xarray.Dataset): The observation dataset containing anomalies.
    season (str): The season for which to calculate the annual mean anomalies.

    Returns:
    xarray.Dataset: The annual mean anomalies for the given observation dataset and season.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-1)
        else:
            obs_anomalies_shifted = obs_anomalies

        # Calculate the annual mean anomalies
        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        #print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

def select_forecast_range(obs_anomalies_annual, forecast_range):
    """
    Selects the forecast range for a given observation dataset.

    Parameters:
    obs_anomalies_annual (xarray.Dataset): The observation dataset containing annual mean anomalies.
    forecast_range (str): The forecast range to select.

    Returns:
    xarray.Dataset: The observation dataset containing annual mean anomalies for the selected forecast range.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        
        forecast_range_start, forecast_range_end = map(int, forecast_range.split("-"))
        #print("Forecast range:", forecast_range_start, "-", forecast_range_end)
        
        rolling_mean_range = forecast_range_end - forecast_range_start + 1
        #print("Rolling mean range:", rolling_mean_range)
        
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time=rolling_mean_range, center = True).mean()
        
        return obs_anomalies_annual_forecast_range
    except Exception as e:
        #print("Error selecting forecast range:", e)
        sys.exit()


def check_for_nan_values(obs):
    """
    Checks for NaN values in the observations dataset.

    Parameters:
    obs (xarray.Dataset): The observations dataset.

    Raises:
    SystemExit: If there are NaN values in the observations dataset.
    """
    try:
        if obs['msl'].isnull().values.any():
            #print("Error: NaN values in observations")
            sys.exit()
    except Exception as e:
        #print("Error checking for NaN values in observations:", e)
        sys.exit()

# Function for checking the model data for NaN values
# For individual years
def check_for_nan_timesteps(ds):
    """
    Checks for NaN values in the given dataset and #prints the timesteps that contain NaN values.

    Parameters:
    ds (xarray.Dataset): The dataset to check for NaN values.

    Returns:
    None
    """
    try:
        # Get the time steps in the dataset
        time_steps = ds.time.values

        # Loop over the time steps
        for time_step in time_steps:
            # Check for NaN values in the dataset for the current time step
            if ds.sel(time=time_step).isnull().values.any():
                print(f"Time step {time_step} contains NaN values")
    except Exception as e:
        print("Error checking for NaN values:", e)

# Define a new function to load the observations
# selecting a specific variable
def load_observations(observations_path, obs_var_name):
    """
    Loads the observations dataset and selects a specific variable.
    
    Parameters:
    variable (str): The variable to load.
    obs_var_name (str): The name of the variable in the observations dataset.

    Returns:
    xarray.Dataset: The observations dataset for the given variable.
    """

    # Check if the observations file exists
    check_file_exists(observations_path)

    # check whether the variable name is valid
    if obs_var_name not in ["psl", "tas", "sfcWind", "rsds"]:
        #print("Invalid variable name")
        sys.exit()

    try:
        # Load the observations dataset
        obs_dataset = xr.open_dataset(observations_path, chunks={"time": 50})[obs_var_name]

        ERA5 = xr.open_mfdataset(observations_path, combine='by_coords', chunks={"time": 50})[obs_var_name]
        ERA5_combine =ERA5.sel(expver=1).combine_first(ERA5.sel(expver=5))
        ERA5_combine.load()
        ERA5_combine.to_netcdf(observations_path + "_copy.nc")

        
        # #print the dimensions of the observations dataset
        # #print("Observations dataset:", obs_dataset.dims)

        # Check for NaN values in the observations dataset
        # check_for_nan_values(obs_dataset)

        return obs_dataset, ERA5_combine

    except Exception as e:
        #print(f"Error loading observations dataset: {e}")
        sys.exit()

# Call the functions to process the observations
def process_observations(variable, region, region_grid, forecast_range, season, observations_path, obs_var_name):
    """
    Processes the observations dataset by regridding it to the model grid, selecting a region and season,
    calculating anomalies, calculating annual mean anomalies, selecting the forecast range, and returning
    the processed observations.

    Args:
        variable (str): The variable to process.
        region (str): The region to select.
        region_grid (str): The grid to regrid the observations to.
        forecast_range (str): The forecast range to select.
        season (str): The season to select.
        observations_path (str): The path to the observations dataset.
        obs_var_name (str): The name of the variable in the observations dataset.

    Returns:
        xarray.Dataset: The processed observations dataset.
    """

    # Check if the observations file exists
    check_file_exists(observations_path)

    # # set up the file name for the processed observations dataset
    # processed_obs_file = dic.home_dir + "/" + "sm_processed_obs" + "/" + variable + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + "outputs" + "/" + variable + "_" + region + "_" + f"years_{forecast_range}" + "_" + season + "_processed_obs_da.nc"
    # # make the directory if it doesn't exist
    # if not os.path.exists(os.path.dirname(processed_obs_file)):
    #     os.makedirs(os.path.dirname(processed_obs_file))

    # # #print the processed observations file name
    # print("Processed observations file name:", processed_obs_file)

    # Process the observations using try and except to catch any errors
    try:
        # Regrid using CDO, select region and load observation dataset
        # for given variable
        obs_dataset = regrid_and_select_region(observations_path, region, obs_var_name)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_dataset")
        # check_for_nan_values(obs_dataset)

        # Select the season
        # --- Although will already be in DJFM format, so don't need to do this ---
        regridded_obs_dataset_region_season = select_season(obs_dataset, season)

        # # #print the dimensions of the regridded and selected region dataset
        #print("Regridded and selected region dataset:", regridded_obs_dataset_region_season.time)

        # # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in regridded_obs_dataset_region_season")
        # check_for_nan_values(regridded_obs_dataset_region_season)
        
        # Calculate anomalies
        obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_anomalies")
        # check_for_nan_values(obs_anomalies)

        # Calculate annual mean anomalies
        obs_annual_mean_anomalies = calculate_annual_mean_anomalies(obs_anomalies, season)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_annual_mean_anomalies")
        # check_for_nan_values(obs_annual_mean_anomalies)

        # Select the forecast range
        obs_anomalies_annual_forecast_range = select_forecast_range(obs_annual_mean_anomalies, forecast_range)
        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_anomalies_annual_forecast_range")
        # check_for_nan_values(obs_anomalies_annual_forecast_range)

        # if the forecast range is "2-2" i.e. a year ahead forecast
        # then we need to shift the dataset by 1 year
        # where the model would show the DJFM average as Jan 1963 (s1961)
        # the observations would show the DJFM average as Dec 1962
        # so we need to shift the observations to the following year
        # if the forecast range is "2-2" and the season is "DJFM"
        # then shift the dataset by 1 year
        if forecast_range == "2-2" and season == "DJFM":
            obs_anomalies_annual_forecast_range = obs_anomalies_annual_forecast_range.shift(time=1)

        # Save the processed observations dataset as a netCDF file
        # print that the file is being saved
        # Save the processed observations dataset as a netCDF file
        # Convert the variable to a DataArray object before saving
        # print("Saving processed observations dataset")
        # obs_anomalies_annual_forecast_range.to_netcdf(processed_obs_file)

        return obs_anomalies_annual_forecast_range

    except Exception as e:
        #print(f"Error processing observations dataset: {e}")
        sys.exit()



def plot_data(obs_data, variable_data, model_time):
    """
    Plots the observations and model data as two subplots on the same figure.
    One on the left and one on the right.

    Parameters:
    obs_data (xarray.Dataset): The processed observations data.
    variable_data (xarray.Dataset): The processed model data for a single variable.
    model_time (str): The time dimension of the model data.

    Returns:
    None
    """

    # #print the dimensions of the observations data
    # #print("Observations dimensions:", obs_data.dims)

    # Take the time mean of the observations
    obs_data_mean = obs_data.mean(dim='time')

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Plot the observations on the left subplot
    obs_data_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=-2, vmax=2)
    ax1.set_title('Observations')

    # Plot the model data on the right subplot
    variable_data.mean(dim=model_time).plot(ax=ax2, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=-2, vmax=2)
    ax2.set_title('Model Data')

    # Set the title of the figure
    # fig.suptitle(f'{obs_data.variable.long_name} ({obs_data.variable.units})\n{obs_data.region} {obs_data.forecast_range} {obs_data.season}')

    # Show the plot
    plt.show()

def plot_obs_data(obs_data):
    """
    Plots the first timestep of the observations data as a single subplot.

    Parameters:
    obs_data (xarray.Dataset): The processed observations data.

    Returns:
    None
    """

    # #print the dimensions of the observations data
    # #print("Observations dimensions:", obs_data.dims)
    # #print("Observations variables:", obs_data)

    # #print all of the latitude values
    # #print("Observations latitude values:", obs_data.lat.values)
    # #print("Observations longitude values:", obs_data.lon.values)

    # Select the first timestep of the observations
    obs_data_first = obs_data.isel(time=-1)

    # Select the variable to be plotted
    # and convert to hPa
    obs_var = obs_data_first["var151"]/100

    # #print the value of the variable
    # #print("Observations variable:", obs_var.values)

    # #print the dimensions of the observations data
    # #print("Observations dimensions:", obs_data_first)

    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the observations on the subplot
    c = ax.contourf(obs_data_first.lon, obs_data_first.lat, obs_var, transform=ccrs.PlateCarree(), cmap='coolwarm')

    # Add coastlines and gridlines to the plot
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add a colorbar to the plot
    fig.colorbar(c, ax=ax, shrink=0.6)

    # Set the title of the figure
    # fig.suptitle(f'{obs_data.variable.long_name} ({obs_data.variable.units})\n{obs_data.region} {obs_data.forecast_range} {obs_data.season}')

    # Show the plot
    plt.show()

# Define a function to make gifs
def make_gif(frame_folder):
    """
    Makes a gif from a folder of images.

    Parameters:
    frame_folder (str): The path to the folder containing the images.
    """

    # Set up the frames to be used
    frames = [Image.open(os.path.join(frame_folder, f)) for f in os.listdir(frame_folder) if f.endswith("_anomalies.png")]
    frame_one = frames[0]
    # Save the frames as a gif
    frame_one.save(os.path.join(frame_folder, "animation.gif"), format='GIF', append_images=frames, save_all=True, duration=300, loop=0)

def plot_model_data(model_data, observed_data, models, gif_plots_path):
    """
    Plots the first timestep of the model data as a single subplot.

    Parameters:
    model_data (dict): The processed model data.
    observed_data (xarray.Dataset): The processed observations data.
    models (list): The list of models to be plotted.
    gif_plots_path (str): The path to the directory where the plots will be saved.
    """

    # if the gif_plots_path directory does not exist
    if not os.path.exists(gif_plots_path):
        # Create the directory
        os.makedirs(gif_plots_path)

    # # #print the values of lat and lon
    # #print("lat values", ensemble_mean[0, :, 0])
    # #print("lon values", ensemble_mean[0, 0, :])

    # lat_test = ensemble_mean[0, :, 0]
    # lon_test = ensemble_mean[0, 0, :]

    # Initialize filepaths
    filepaths = []

    # Extract the years from the model data
    # #print the values of the years
    # #print("years values", years)
    # #print("years shape", np.shape(years))
    # #print("years type", type(years))

    # Process the model data and calculate the ensemble mean
    ensemble_mean, lat, lon, years = process_model_data_for_plot(model_data, models)

    # #print the dimensions of the model data
    #print("ensemble mean shape", np.shape(ensemble_mean))

    # set the vmin and vmax values
    vmin = -5
    vmax = 5

    # process the observed data
    # extract the lat and lon values
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values
    obs_years = observed_data.time.dt.year.values

    # Do we need to convert the lons in any way here?
    # #print the values of lat and lon
    # #print("obs lat values", obs_lat)
    # #print("obs lon values", obs_lon) 
    # #print("obs lat shape", np.shape(obs_lat))
    # #print("obs lon shape", np.shape(obs_lon))
    # #print("model lat shape", np.shape(lat))
    # #print("model lon shape", np.shape(lon))
    # #print("model lat values", lat)
    # #print("model lon values", lon)
    # #print("years values", years)
    # #print("obs years values", obs_years)
    # #print("obs years shape", np.shape(obs_years))
    # #print("obs years type", type(obs_years))
    # #print("model year shape", np.shape(years))

    # Make sure that the obs and model data are for the same time period
    # Find the years which are in both the obs and model data
    years_in_both = np.intersect1d(obs_years, years)

    # Select the years which are in both the obs and model data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))
    ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))

    # remove the years with NaN values from the model data
    observed_data, ensemble_mean = remove_years_with_nans(observed_data, ensemble_mean)

    # convert to numpy arrays
    # and convert from pa to hpa
    obs_array = observed_data['var151'].values / 100
    model_array = ensemble_mean.values / 100

    # Check that these have the same shape
    if np.shape(obs_array) != np.shape(model_array):
        raise ValueError("The shapes of the obs and model arrays do not match")
    else:
        print("The shapes of the obs and model arrays match")

    # assign the obs and model arrays to the same variable
    obs = obs_array
    model = model_array

    # Loop over the years array
    for year in years:
        # #print the year
        # #print("year", year)

        # Set up the figure
        # modify for three subplots
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the ensemble mean on the subplot
        # for the specified year
        # Check that the year index is within the range of the years array
        if year < years[0] or year > years[-1]:
            continue

        # Find the index of the year in the years array
        year_index = np.where(years == year)[0][0]

        
        # #print the values of the model and obs arrays
        # #print("model values", model[year_index, :, :])
        # #print("obs values", obs[year_index, :, :])

        # Plot the ensemble mean on the subplot
        # for the specified year
        c1 = axs[0].contourf(lon, lat, model[year_index, :, :], transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # Add coastlines and gridlines to the plot
        axs[0].coastlines()
        # axs[0].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[0].annotate(f"{year}", xy=(0.01, 0.92), xycoords='axes fraction', fontsize=16)
        # annotate the plot with model
        # in the top right corner
        axs[0].annotate(f"{models[0]}", xy=(0.8, 0.92), xycoords='axes fraction', fontsize=16)

        # Plot the observations on the subplot
        c2 = axs[1].contourf(lon, lat, obs[year_index, :, :], transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # Add coastlines and gridlines to the plot
        axs[1].coastlines()
        # axs[1].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[1].annotate(f"{year}", xy=(0.01, 0.92), xycoords='axes fraction', fontsize=16)
        # annotate the plot with obs
        # in the top right corner
        axs[1].annotate(f"obs", xy=(0.8, 0.92), xycoords='axes fraction', fontsize=16)

        # Plot the anomalies on the subplot
        c3 = axs[2].contourf(lon, lat, model[year_index, :, :] - obs[year_index, :, :], transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # Add coastlines and gridlines to the plot
        axs[2].coastlines()
        # axs[2].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[2].annotate(f"{year}", xy=(0.01, 0.92), xycoords='axes fraction', fontsize=16)
        axs[2].annotate(f"anoms", xy=(0.8, 0.92), xycoords='axes fraction', fontsize=16)

        # Set up the filepath for saving
        filepath = os.path.join(gif_plots_path, f"{year}_obs_model_anoms.png")
        # Save the figure
        fig.savefig(filepath)

        # Add the filepath to the list of filepaths
        filepaths.append(filepath)

    # Create the gif
    # Using the function defined above
    make_gif(gif_plots_path)

    # Show the plot
    # plt.show()

# Define a function to constrain the years to the years that are in all of the model members
def constrain_years(model_data, models):
    """
    Constrains the years to the years that are in all of the models.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    constrained_data (dict): The model data with years constrained to the years that are in all of the models.
    """
    # Initialize a list to store the years for each model
    years_list = []

    # #print the models being proces
    # #print("models:", models)
    
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # Append the years to the list of years
            years_list.append(years)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))

    # #print the common years for debugging
    # #print("Common years:", common_years)
    # #print("Common years type:", type(common_years))
    # #print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # #print the years extracted from the model
            # #print('model years', years)
            # #print('model years shape', np.shape(years))
            
            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(years, common_years)

            # #print("years in both shape", np.shape(years_in_both))
            # #print("years in both", years_in_both)
            
            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_data:
                constrained_data[model] = []
            constrained_data[model].append(member)

    # # #print the constrained data for debugging
    # #print("Constrained data:", constrained_data)

    return constrained_data


# Define a function which processes the model data for spatial correlations
def process_model_data_for_plot(model_data, models):
    """
    Processes the model data and calculates the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    ensemble_mean (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    """
    # Initialize a list for the ensemble members
    ensemble_members = []

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # #print
        #print("extracting data for model:", model)

        # Set the ensemble members count to zero
        # if the model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0
        
        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Try to #print values for each member
            # #print("trying to #print values for each member for debugging")
            # #print("values for model:", model)
            # #print("values for members:", member)
            # #print("member values:", member.values)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            years = member.time.dt.year.values

            # #print statements for debugging
            # #print('shape of years', np.shape(years))
            # # #print('years', years)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # #print the dimensions of the ensemble members
    # #print("ensemble members shape", np.shape(ensemble_members))

    # #print the ensemble members count
    print("ensemble members count", ensemble_members_count)

    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # #print the dimensions of the ensemble mean
    # #print(np.shape(ensemble_mean))
    # #print(type(ensemble_mean))
    # #print(ensemble_mean)
        
    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(ensemble_mean, coords=member.coords, dims=member.dims)

    return ensemble_mean, lat, lon, years, ensemble_members_count

def calculate_spatial_correlations(observed_data, model_data, models, variable):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. Before calculating the spatial correlations between the two datasets.
    
    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    # try:
    # Process the model data and calculate the ensemble mean
    ensemble_mean, lat, lon, years, ensemble_members_count = process_model_data_for_plot(model_data, models)

    # Debug the model data
    # #print("ensemble mean within spatial correlation function:", ensemble_mean)
    # #print("shape of ensemble mean within spatial correlation function:", np.shape(ensemble_mean))
    
    # Extract the lat and lon values
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values
    # And the years
    obs_years = observed_data.time.dt.year.values

    # Initialize lists for the converted lons
    obs_lons_converted, lons_converted = [], []

    # Transform the obs lons
    obs_lons_converted = np.where(obs_lon > 180, obs_lon - 360, obs_lon)
    # add 180 to the obs_lons_converted
    obs_lons_converted = obs_lons_converted + 180

    # For the model lons
    lons_converted = np.where(lon > 180, lon - 360, lon)
    # # add 180 to the lons_converted
    lons_converted = lons_converted + 180

    # #print the observed and model years
    # #print('observed years', obs_years)
    # #print('model years', years)
    
    # Find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, years)

    # #print('years in both', years_in_both)

    # Select only the years that are in both the observed and model data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))
    ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))

    # Remove years with NaNs
    observed_data, ensemble_mean = remove_years_with_nans(observed_data, ensemble_mean, variable)

    # #print the ensemble mean values
    # #print("ensemble mean value after removing nans:", ensemble_mean.values)

    # # set the obs_var_name
    # obs_var_name = variable
    
    # # choose the variable name for the observed data
    # # Translate the variable name to the name used in the obs dataset
    # if obs_var_name == "psl":
    #     obs_var_name = "msl"
    # elif obs_var_name == "tas":
    #     obs_var_name = "t2m"
    # elif obs_var_name == "sfcWind":
    #     obs_var_name = "si10"
    # elif obs_var_name == "rsds":
    #     obs_var_name = "ssrd"
    # elif obs_var_name == "tos":
    #     obs_var_name = "sst"
    # else:
    #     #print("Invalid variable name")
    #     sys.exit()

    # variable extracted already
    # Convert both the observed and model data to numpy arrays
    observed_data_array = observed_data.values / 100
    ensemble_mean_array = ensemble_mean.values / 100

    # #print the values and shapes of the observed and model data
    # #print("observed data shape", np.shape(observed_data_array))
    # #print("model data shape", np.shape(ensemble_mean_array))
    # #print("observed data", observed_data_array)
    # #print("model data", ensemble_mean_array)

    # Check that the observed data and ensemble mean have the same shape
    if observed_data_array.shape != ensemble_mean_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    rfield, pfield = calculate_correlations(observed_data_array, ensemble_mean_array, obs_lat, obs_lon)

    return rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count

    # except Exception as e:
    #     #print(f"An error occurred when calculating spatial correlations: {e}")

def calculate_correlations(observed_data, model_data, obs_lat, obs_lon):
    """
    Calculates the spatial correlations between the observed and model data.

    Parameters:
    observed_data (numpy.ndarray): The processed observed data.
    model_data (numpy.ndarray): The processed model data.
    obs_lat (numpy.ndarray): The latitude values of the observed data.
    obs_lon (numpy.ndarray): The longitude values of the observed data.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    try:
        # Initialize empty arrays for the spatial correlations and p-values
        rfield = np.empty([len(obs_lat), len(obs_lon)])
        pfield = np.empty([len(obs_lat), len(obs_lon)])

        # #print the dimensions of the observed and model data
        # #print("observed data shape", np.shape(observed_data))
        # #print("model data shape", np.shape(model_data))

        # Loop over the latitudes and longitudes
        for y in range(len(obs_lat)):
            for x in range(len(obs_lon)):
                # set up the obs and model data
                obs = observed_data[:, y, x]
                mod = model_data[:, y, x]

                # #print the obs and model data
                # #print("observed data", obs)
                # #print("model data", mod)

                # Calculate the correlation coefficient and p-value
                r, p = stats.pearsonr(obs, mod)

                # #print the correlation coefficient and p-value
                # #print("correlation coefficient", r)
                # #print("p-value", p)

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p

        # #print the range of the correlation coefficients and p-values
        # to 3 decimal places
        #print(f"Correlation coefficients range from {rfield.min():.3f} to {rfield.max():.3f}")
        #print(f"P-values range from {pfield.min():.3f} to {pfield.max():.3f}")

        # Return the correlation coefficients and p-values
        return rfield, pfield

    except Exception as e:
        #print(f"Error calculating correlations: {e}")
        sys.exit()

# checking for Nans in observed data
def remove_years_with_nans(observed_data, ensemble_mean, variable):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        ensemble_mean (xarray.Dataset): The ensemble mean (model data).
        variable (str): the variable name.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
    """

    # # Set the obs_var_name == variable
    obs_var_name = variable

    # Set up the obs_var_name
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    else:
        #print("Invalid variable name")
        sys.exit()

    #print("var name for obs", obs_var_name)
    
    for year in observed_data.time.dt.year.values[::-1]:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # print("data type", (type(data)))
        # print("data vaues", data)
        # print("data shape", np.shape(data))

        
        # If there are any Nan values in the data
        if np.isnan(data.values).any():
            # #print the year
            # #print(year)

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

            # for the model data
            ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year != year)

        # if there are no Nan values in the data for a year
        # then #print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no nan for this year")

            # exit the loop
            break

    return observed_data, ensemble_mean

# plot the correlations and p-values
def plot_correlations(models, rfield, pfield, obs, variable, region, season, forecast_range, plots_dir, 
                        obs_lons_converted, lons_converted, azores_grid, iceland_grid, uk_n_box, 
                            uk_s_box, ensemble_members_count = None, p_sig = 0.05):
    """Plot the correlation coefficients and p-values.
    
    This function plots the correlation coefficients and p-values
    for a given variable, region, season and forecast range.
    
    Parameters
    ----------
    model : str
        Name of the models.
    rfield : array
        Array of correlation coefficients.
    pfield : array
        Array of p-values.
    obs : str
        Observed dataset.
    variable : str
        Variable.
    region : str
        Region.
    season : str
        Season.
    forecast_range : str
        Forecast range.
    plots_dir : str
        Path to the directory where the plots will be saved.
    obs_lons_converted : array
        Array of longitudes for the observed data.
    lons_converted : array
        Array of longitudes for the model data.
    azores_grid : array
        Array of longitudes and latitudes for the Azores region.
    iceland_grid : array
        Array of longitudes and latitudes for the Iceland region.
    uk_n_box : array
        Array of longitudes and latitudes for the northern UK index box.
    uk_s_box : array
        Array of longitudes and latitudes for the southern UK index box.
    p_sig : float, optional
        Significance level for the p-values. The default is 0.05.
    """

    # Extract the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Extract the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # Extract the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box['lon1'], uk_n_box['lon2']
    uk_n_lat1, uk_n_lat2 = uk_n_box['lat1'], uk_n_box['lat2']

    # Extract the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box['lon1'], uk_s_box['lon2']
    uk_s_lat1, uk_s_lat2 = uk_s_box['lat1'], uk_s_box['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # subtract 180 from all of the uk lons
    uk_n_lon1, uk_n_lon2 = uk_n_lon1 - 180, uk_n_lon2 - 180
    uk_s_lon1, uk_s_lon2 = uk_s_lon1 - 180, uk_s_lon2 - 180

    # set up the converted lons
    # Set up the converted lons
    lons_converted = lons_converted - 180

    # Set up the lats and lons
    # if the region is global
    if region == 'global':
        lats = obs.lat
        lons = lons_converted
    # if the region is not global
    elif region == 'north-atlantic':
        lats = obs.lat
        lons = lons_converted
    else:
        #print("Error: region not found")
        sys.exit()

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Set the projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines
    ax.coastlines()

    # Add gridlines with labels for the latitude and longitude
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False
    # gl.xlabel_style = {'size': 12}
    # gl.ylabel_style = {'size': 12}

    # Add green lines outlining the Azores and Iceland grids
    ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Add green lines outlining the northern and southern UK index boxes
    ax.plot([uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1], [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Add filled contours
    # Contour levels
    clevs = np.arange(-1, 1.1, 0.1)
    # Contour levels for p-values
    clevs_p = np.arange(0, 1.1, 0.1)
    # Plot the filled contours
    cf = plt.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=ccrs.PlateCarree())

    # If the variables is 'tas'
    # then we want to invert the stippling
    # so that stippling is plotted where there is no significant correlation
    if variable == 'tas':
        # replace values in pfield that are less than 0.05 with nan
        pfield[pfield < p_sig] = np.nan
    else:
        # replace values in pfield that are greater than 0.05 with nan
        pfield[pfield > p_sig] = np.nan

    # #print the pfield
    # #print("pfield mod", pfield)

    # Add stippling where rfield is significantly different from zero
    plt.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Correlation Coefficient')

    # extract the model name from the list
    # given as ['model']
    # we only want the model name
    # if the length of the list is 1
    # then the model name is the first element
    if len(models) == 1:
        model = models[0]
    elif len(models) > 1:
        models = "multi-model mean"
    else :
        #print("Error: model name not found")
        sys.exit()

    # Set up the significance threshold
    # if p_sig is 0.05, then sig_threshold is 95%
    sig_threshold = int((1 - p_sig) * 100)

    # Extract the number of ensemble members from the ensemble_members_count dictionary
    # if the ensemble_members_count is not None
    if ensemble_members_count is not None:
        total_no_members = sum(ensemble_members_count.values())

    # Add title
    plt.title(f"{models} {variable} {region} {season} {forecast_range} Correlation Coefficients, p < {p_sig} ({sig_threshold}%), N = {total_no_members}")

    # set up the path for saving the figure
    fig_name = f"{models}_{variable}_{region}_{season}_{forecast_range}_N_{total_no_members}_p_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

# Function for plotting the results for all of the models as 12 subplots
def plot_correlations_subplots(models, obs, variable_data, variable, region, season, forecast_range, plots_dir, azores_grid, iceland_grid, uk_n_box, uk_s_box, p_sig = 0.05):
    """Plot the spatial correlation coefficients and p-values for all models.

    This function plots the spatial correlation coefficients and p-values
    for all models in the dictionaries.models list for a given variable,
    region, season and forecast range.

    Parameters
    ----------
    models : List
        List of models.
    obs : str
        Observed dataset.
    variable_data : dict
        Variable data for each model.
    region : str
        Region.
    season : str
        Season.
    forecast_range : str
        Forecast range.
    plots_dir : str
        Path to the directory where the plots will be saved.
    azores_grid : array
        Array of longitudes and latitudes for the Azores region.
    iceland_grid : array
        Array of longitudes and latitudes for the Iceland region.
    uk_n_box : array
        Array of longitudes and latitudes for the northern UK index box.
    uk_s_box : array
        Array of longitudes and latitudes for the southern UK index box.
    p_sig : float, optional
        Significance threshold. The default is 0.05.
    """

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the projection
    proj = ccrs.PlateCarree()
    
    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']
    
    # Set up the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box['lon1'], uk_n_box['lon2']
    uk_n_lat1, uk_n_lat2 = uk_n_box['lat1'], uk_n_box['lat2']

    # Set up the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box['lon1'], uk_s_box['lon2']
    uk_s_lat1, uk_s_lat2 = uk_s_box['lat1'], uk_s_box['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # subtract 180 from all of the uk lons
    uk_n_lon1, uk_n_lon2 = uk_n_lon1 - 180, uk_n_lon2 - 180
    uk_s_lon1, uk_s_lon2 = uk_s_lon1 - 180, uk_s_lon2 - 180

    # Count the number of models available
    nmodels = len(models)

    # Set the figure size and subplot parameters
    if nmodels == 8:
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        # Remove the last subplot
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 11:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 12:
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1})
        # Set up where to plot the title
        title_index = 1
    else:
        raise ValueError(f"Invalid number of models: {nmodels}")
    
    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)
    
    # Flatten the axs array
    axs = axs.flatten()

    # Create a list to store the contourf objects
    cf_list = []
    
    # Loop over the models
    for i, model in enumerate(models):
        
        # #print the model name
        #print("Processing model:", model)
    
        # Convert the model to a single index list
        model = [model]
    
        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = calculate_spatial_correlations(obs,
                                                                                        variable_data, model, variable)

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        # if the region is global
        if region == 'global':
            lats = obs.lat
            lons = lons_converted
        # if the region is not global
        elif region == 'north-atlantic':
            lats = obs.lat
            lons = lons_converted
        else:
            #print("Error: region not found")
            sys.exit()

        # Set up the axes
        ax = axs[i]

        # Add coastlines
        ax.coastlines()
    
        # Add gridlines with labels for the latitude and longitude
        # gl = ax.gridlines(crs=proj, draw_labels=False, linewidth=2, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.xlabel_style = {'size': 12}
        # gl.ylabel_style = {'size': 12}
    
        # Add green lines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # Add green lines outlining the northern and southern UK index boxes
        ax.plot([uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1], [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=proj)
    
        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == 'tas':
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan
    
        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=proj)
    
        # Add title
        # ax.set_title(f"{model} {variable} {region} {season} {forecast_range} Correlation Coefficients")
    
        # extract the model name from the list
        if len(model) == 1:
            model = model[0]
        elif len(model) > 1:
            model = "all_models"
        else :
            #print("Error: model name not found")
            sys.exit()
    
        # Add textbox with model name
        ax.text(0.05, 0.95, model, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.5))
    
        # Add the contourf object to the list
        cf_list.append(cf)

        # If this is the centre subplot on the first row, set the title for the figure
        if i == title_index:
            # Add title
            ax.set_title(f"{variable} {region} {season} years {forecast_range} Correlation Coefficients, p < {p_sig} ({sig_threshold}%)", fontsize=12)
    
    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Specify a tight layout
    # plt.tight_layout()

    # set up the path for saving the figure
    fig_name = f"{variable}_{region}_{season}_{forecast_range}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # # Adjust the vertical spacing between the plots
    # plt.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Show the figure
    plt.show()


# Functions for choosing the observed data path
# and full variable name
def choose_obs_path(args):
    """
    Choose the obs path based on the variable
    """
    if args.variable == "psl":
        obs_path = dic.obs_psl
    elif args.variable == "tas":
        obs_path = dic.obs_tas
    elif args.variable == "sfcWind":
        obs_path = dic.obs_sfcWind
    elif args.variable == "rsds":
        obs_path = dic.obs_rsds
    else:
        #print("Error: variable not found")
        sys.exit()
    return obs_path

# Choose the observed variable name
def choose_obs_var_name(args):
    """
    Choose the obs var name based on the variable
    """
    if args.variable == "psl":
        obs_var_name = dic.psl_label
    elif args.variable == "tas":
        obs_var_name = dic.tas_label
    elif args.variable == "sfcWind":
        obs_var_name = dic.sfc_wind_label
    elif args.variable == "rsds":
        obs_var_name = dic.rsds_label
    else:
        #print("Error: variable not found")
        sys.exit()
    return obs_var_name

# Write a new function which will plot a series of subplots
# for the same variable, region and forecast range (e.g. psl global years 2-9)
# but with different seasons (e.g. DJFM, MAM, JJA, SON)
# TODO: this doesn't include bootstrapped p values
def plot_seasonal_correlations(models, observations_path, variable, region, region_grid, forecast_range, seasons_list_obs, seasons_list_mod, 
                                plots_dir, obs_var_name, azores_grid, iceland_grid, p_sig = 0.05, experiment = 'dcppA-hindcast'):
    """
    Plot the spatial correlation coefficients and p-values for the same variable,
    region and forecast range (e.g. psl global years 2-9) but with different seasons.
    
    Arguments
    ---------
    models : list
        List of models.
    obsservations_path : str
        Path to the observations.
    variable : str
        Variable.
    region : str
        Region.
    region_grid : dict
        Dictionary of region grid.
    forecast_range : str
        Forecast range.
    seasons_list_obs : list
        List of seasons for the obs.
    seasons_list_mod : list
        List of seasons for the models.
    plots_dir : str
        Path to the directory where the plots will be saved.
    obs_var_name : str
        Observed variable name.
    azores_grid : dict
        Dictionary of Azores grid.
    iceland_grid : dict
        Dictionary of Iceland grid.
    p_sig : float, optional
        Significance threshold. The default is 0.05.
    experiment : str, optional
        Experiment name. The default is 'dcppA-hindcast'.

    Returns
    -------
    None.

    """

    # Create an empty list to store the processed observations
    # for each season
    obs_list = []

    # Create empty lists to store the r and p fields
    # for each season
    rfield_list = []
    pfield_list = []

    # Create lists to store the obs_lons_converted and lons_converted
    # for each season
    obs_lons_converted_list = []
    lons_converted_list = []

    # Create an empty list to store the ensemble members count
    ensemble_members_count_list = []

    # Add labels A, B, C, D to the subplots
    ax_labels = ['A', 'B', 'C', 'D']

    # Loop over the seasons
    for i in range(len(seasons_list_obs)):
        
        # Print the season(s) being processed
        print("obs season", seasons_list_obs[i])
        print("mod season", seasons_list_mod[i])

        # Process the observations
        obs = process_observations(variable, region, region_grid, forecast_range, seasons_list_obs[i], observations_path, obs_var_name)

        # Load and process the model data
        model_datasets = load_data(dic.base_dir, models, variable, region, forecast_range, seasons_list_mod[i])
        # Process the model data
        model_data, model_time = process_data(model_datasets, variable)

        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = calculate_spatial_correlations(obs, model_data, models, variable)

        # Append the processed observations to the list
        obs_list.append(obs)

        # Append the r and p fields to the lists
        rfield_list.append(rfield)
        pfield_list.append(pfield)

        # Append the ensemble members count to the list
        ensemble_members_count_list.append(ensemble_members_count)

        # Append the obs_lons_converted and lons_converted to the lists
        obs_lons_converted_list.append(obs_lons_converted)
        lons_converted_list.append(lons_converted) 

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # Set up the fgure size and subplot parameters
    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), subplot_kw={'projection': proj})

    # Set up the title for the figure
    title = f"{variable} {region} {forecast_range} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=16)

    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # Flatten the axs array
    axs = axs.flatten()

    # Create a list to store the contourf objects
    cf_list = []

    # Loop over the seasons
    for i in range(len(seasons_list_obs)):

        # Print the season(s) being pplotted
        print("plotting season", seasons_list_obs[i])

        # Extract the season
        season = seasons_list_obs[i]

        # Extract the obs
        obs = obs_list[i]

        # Extract the r and p fields
        rfield, pfield = rfield_list[i], pfield_list[i]

        # Extract the obs_lons_converted and lons_converted
        obs_lons_converted, lons_converted = obs_lons_converted_list[i], lons_converted_list[i]

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        lats = obs.lat
        lons = lons_converted

        # Set up the axes
        ax = axs[i]

        # Add coastlines
        ax.coastlines()

        # Add greenlines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == 'tas':
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=proj)

        # Add a textbox with the season name
        ax.text(0.05, 0.95, season, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.5))

        # # Add a textbox with the number of ensemble members in the bottom right corner
        # ax.text(0.95, 0.05, f"N = {ensemble_members_count_list[i]}", transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbox in the bottom right with the figure letter
        # extract the figure letter from the ax_labels list
        fig_letter = ax_labels[i]
        ax.text(0.95, 0.05, fig_letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # # Set up the text for the subplot
        # ax.text(-0.1, 1.1, key, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

        # Add the contourf object to the list
        cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('correlation coefficients')



    print("ax_labels shape", np.shape(ax_labels))
    for i, ax in enumerate(axs):
        # Add the label to the bottom left corner of the subplot
        ax.text(0.05, 0.05, ax_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    # Set up the path for saving the figure
    fig_name = f"{variable}_{region}_{forecast_range}_{experiment}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

# Now we want to write another function for creating subplots
# This one will plot for the same season, region, forecast range
# but for different variables (e.g. psl, tas, sfcWind, rsds)
def plot_variable_correlations(models, observations_path, variables_list, region, region_grid, forecast_range, season,
                                plots_dir, obs_var_names, azores_grid, iceland_grid, p_sig = 0.05, experiment = 'dcppA-hindcast'):
    """
    Plot the spatial correlation coefficients and p-values for different variables,
    but for the same season, region, and forecast range.
    
    Arguments
    ---------
    models : list
        List of models.
    obsservations_path : str
        Path to the observations.
    variables_list : list
        List of variables.
    region : str
        Region.
    region_grid : dict
        Dictionary of region grid.
    forecast_range : str
        Forecast range.
    season : str
        Season.
    plots_dir : str
        Path to the directory where the plots will be saved.
    obs_var_names : list
        List of observed variable names.
    azores_grid : dict
        Dictionary of Azores grid.
    iceland_grid : dict
        Dictionary of Iceland grid.
    p_sig : float, optional
        Significance threshold. The default is 0.05.
    experiment : str, optional
        Experiment name. The default is 'dcppA-hindcast'.

    Returns
    -------
    None.

    """

    # Create an empty list to store the processed observations
    obs_list = []

    # Create empty lists to store the r and p fields
    rfield_list = []
    pfield_list = []

    # Create an empty list to store the ensemble members count
    ensemble_members_count_list = []

    # Loop over the variables
    for i in range(len(variables_list)):
        
        # Print the variable being processed
        print("processing variable", variables_list[i])

        # Process the observations
        obs = process_observations(variables_list[i], region, region_grid, forecast_range, season, observations_path, obs_var_names[i])

        # Load and process the model data
        model_datasets = load_data(dic.base_dir, models, variables_list[i], region, forecast_range, season)
        # Process the model data
        model_data, model_time = process_data(model_datasets, variables_list[i])

        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = calculate_spatial_correlations(obs, model_data, models, variables_list[i])

        # Append the processed observations to the list
        obs_list.append(obs)

        # Append the r and p fields to the lists
        rfield_list.append(rfield)
        pfield_list.append(pfield)

        # Append the ensemble members count to the list
        ensemble_members_count_list.append(ensemble_members_count)

    # Set the font size for the plots
    plt.rcParams.update({'font.size': 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid['lon1'], azores_grid['lon2']
    azores_lat1, azores_lat2 = azores_grid['lat1'], azores_grid['lat2']

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid['lon1'], iceland_grid['lon2']
    iceland_lat1, iceland_lat2 = iceland_grid['lat1'], iceland_grid['lat2']

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12), subplot_kw={'projection': proj})

    # Set up the title for the figure
    title = f"{region} {forecast_range} {season} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=16)

    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # Create a list to store the contourf objects
    cf_list = []

    # Loop over the variables
    for i in range(len(variables_list)):

        # Print the variable being plotted
        print("plotting variable", variables_list[i])

        # Extract the variable
        variable = variables_list[i]

        # Extract the obs
        obs = obs_list[i]

        # Extract the r and p fields
        rfield, pfield = rfield_list[i], pfield_list[i]

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        lats = obs.lat
        lons = lons_converted

        # Set up the axes
        ax = axs.flatten()[i]

        # Add coastlines
        ax.coastlines()

        # Add greenlines outlining the Azores and Iceland grids
        ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap='RdBu_r', transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == 'tas':
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=['....'], alpha=0, transform=proj)

        # Add a textbox with the variable name
        ax.text(0.05, 0.95, variable, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbox with the number of ensemble members in the bottom right corner
        ax.text(0.95, 0.05, f"N = {ensemble_members_count_list[i]}", transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # Add the contourf object to the list
        cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(cf_list[0], orientation='horizontal', pad=0.05, aspect=50, ax=fig.axes, shrink=0.8)
    cbar.set_label('correlation coefficients')

    # Add labels A, B, C, D to the subplots
    ax_labels = ['A', 'B', 'C', 'D']
    for i, ax in enumerate(fig.axes):
        # Add the label to the bottom left corner of the subplot
        ax.text(0.05, 0.05, ax_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    # Set up the path for saving the figure
    fig_name = f"{region}_{forecast_range}_{season}_{experiment}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

# define a main function
def main():
    """Main function for the program.
    
    This function parses the arguments from the command line
    and then calls the functions to load and process the data.
    """

    # Create a usage statement for the script.
    USAGE_STATEMENT = """python functions.py <variable> <model> <region> <forecast_range> <season>"""

    # Check if the number of arguments is correct.
    if len(sys.argv) != 6:
        #print(f"Expected 6 arguments, but got {len(sys.argv)}")
        #print(USAGE_STATEMENT)
        sys.exit()

    # Make the plots directory if it doesn't exist.
    if not os.path.exists(dic.plots_dir):
        os.makedirs(dic.plots_dir)

    # Parse the arguments from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", help="variable", type=str)
    parser.add_argument("model", help="model", type=str)
    parser.add_argument("region", help="region", type=str)
    parser.add_argument("forecast_range", help="forecast range", type=str)
    parser.add_argument("season", help="season", type=str)
    args = parser.parse_args()

    # #print the arguments to the screen.
    #print("variable = ", args.variable)
    #print("model = ", args.model)
    #print("region = ", args.region)
    #print("forecast range = ", args.forecast_range)
    #print("season = ", args.season)

    # If the model specified == "all", then run the script for all models.
    if args.model == "all":
        args.model = dic.models

    # If the type of the model argument is a string, then convert it to a list.
    if type(args.model) == str:
        args.model = [args.model]

    # Load the data.
    datasets = load_data(dic.base_dir, args.model, args.variable, args.region, args.forecast_range, args.season)

    # Process the model data.
    variable_data, model_time = process_data(datasets, args.variable)

    # Choose the obs path based on the variable
    obs_path = choose_obs_path(args)

    # choose the obs var name based on the variable
    obs_var_name = choose_obs_var_name(args)

    # Process the observations.
    obs = process_observations(args.variable, args.region, dic.north_atlantic_grid, args.forecast_range, args.season, obs_path, obs_var_name)

    # Call the function to calculate the ACC
    rfield, pfield, obs_lons_converted, lons_converted = calculate_spatial_correlations(obs, variable_data, args.model)

    # Call the function to plot the ACC
    plot_correlations(args.model, rfield, pfield, obs, args.variable, args.region, args.season, args.forecast_range, dic.plots_dir, obs_lons_converted, lons_converted, dic.azores_grid, dic.iceland_grid)

# Call the main function.
if __name__ == "__main__":
    main()
