# functions for the main program
# these should be tested one by one
# before being used in the main program
#
# Usage: python functions.py <variable> <region> <forecast_range> <season>
#
# Example: python functions.py "psl" "north-atlantic" "2-5" "DJF"
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
        files_path = base_directory + "/" + variable + "/" + model + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + "outputs" + "/" + "mergetime" + "/" + "*-anoms.nc"

        # print the path to the files
        print("Searching for files in ", files_path)

        # Create a list of the files for this model.
        files = glob.glob(files_path)

        # if the list of files is empty, print a warning and
        # exit the program
        if len(files) == 0:
            print("No files found for " + model)
            sys.exit()
        
        # Print the files to the screen.
        print("Files for " + model + ":", files)

        # Loop over the files.
        for file in files:

            # Print the file to the screen.
            print(file)
            
            # check that the file exists
            # if it doesn't exist, print a warning and
            # exit the program
            if not os.path.exists(file):
                print("File " + file + " does not exist")
                sys.exit()
            else:
                print("Loading " + file)

            # Load the dataset.
            dataset = xr.open_dataset(file, chunks = {"time": 50})
            
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
        A dictionary of processed datasets grouped by models.
    """
    print(f"Dataset type: {type(datasets_by_model)}")

    def process_model_dataset(dataset, variable):
        """Process a single dataset.
        
        This function takes a single dataset and processes the data.
        
        Args:
            dataset: A single dataset.
            variable: The variable to load, extracted from the command line.
            
        Returns:
            A processed dataset.
        """
        
        if variable == "psl":
            # Extract the variable.
            variable_data = dataset["psl"]

            # print the variable data
            print("Variable data: ", variable_data)
            # print the variable data type
            print("Variable data type: ", type(variable_data))

            # print the len of the variable data dimensions
            print("Variable data dimensions: ", len(variable_data.dims))
            
            # Convert from Pa to hPa.
            # Using try and except to catch any errors.
            try:
                variable_data = variable_data

                # print the values of the variable data
                print("Variable data values: ", variable_data.values)

            except:
                print("Error converting from Pa to hPa")
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
            print("Variable " + variable + " not recognised")
            sys.exit()

        # If variable_data is empty, print a warning and exit the program.
        if variable_data is None:
            print("Variable " + variable + " not found in dataset")
            sys.exit()

        # Extract the time dimension.
        model_time = dataset["time"].values
        # Set the type for the time dimension.
        model_time = model_time.astype("datetime64[Y]")

        # If model_time is empty, print a warning and exit the program.
        if model_time is None:
            print("Time not found in dataset")
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
            print(f"Error processing dataset for model {model}: {e}")
            print("Exiting the program")
            sys.exit()

    # Return the processed data.
    return variable_data_by_model, model_time_by_model

# Write a function to process the observations
# this takes the arguments
# variable, region, forecast_range, season
# and the path to the observations
# it returns the processed observations
def process_observations_long(variable, region, region_grid, forecast_range, season, observations_path, obs_var_name):
    """
    Process the observations. Regrids observations to the model grid (2.5x2.5). Selects the same region as the model data. Selects the same forecast range as the model data. Selects the same season as the model data.
    --------------------------------------------
    ERA5 observations are only available for DJFM!
    --------------------------------------------

    Args:
        variable: The variable to load, extracted from the command line.
        region: The region to load, extracted from the command line.
        region_grid: The region grid to load, extracted from the dictionary.
        forecast_range: The forecast range to load, extracted from the command line.
        season: The season to load, extracted from the command line.
        observations_path: The path to the observations, extracted from the command line.
        obs_var_name: The observations variable name, extracted from the dictionary.

    Returns:
        obs_variable_data: The processed observations variable data.
    """

    # First check if the observations file exists.
    # If it doesn't exist, print a warning and exit the program.
    if not os.path.exists(observations_path):
        print("Observations file " + observations_path + " does not exist")
        sys.exit()

    # Load the observations dataset.
    obs_dataset = xr.open_dataset(observations_path, chunks = {"time": 50})

    #  Check if the obs_dataset has been loaded correctly.
    # If it hasn't been loaded correctly, print a warning and exit the program.
    if obs_dataset is None:
        print("Observations dataset not loaded correctly")
        sys.exit()

    # Regrid the observations to the model grid.
    # Of 2.5 x 2.5 degrees.
    # Using try and except to catch any errors.
    try:
        regrid_example_dataset = xr.Dataset({"latitude": (["latitude"], np.arange(0, 360.1, 2.5)),"longitude": (["longitude"], np.arange(-90, 90.1, 2.5))})

        # Regrid the observations to the model grid.
        regridded_obs_dataset = obs_dataset.interp(latitude = regrid_example_dataset.latitude, longitude = regrid_example_dataset.longitude)

        # Now select the region.
        regridded_obs_dataset_region = regridded_obs_dataset.sel(lat=slice(region_grid[0], region_grid[1]), lon=slice(region_grid[2], region_grid[3]))

    except:
        print("Error regridding observations")
        sys.exit()

    # If regridded_obs_dataset is empty, print a warning and exit the program.
    if regridded_obs_dataset_region is None:
        print("Observations not regridded correctly")
        sys.exit()

    # Select the season
    # Using try and except to catch any errors.
    try:
        # Select the season.
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(time = regridded_obs_dataset_region["time.season"] == season)

    except:
        print("Error selecting season")
        sys.exit()

    # If regridded_obs_dataset_region_season is empty, print a warning and exit the program.
    if regridded_obs_dataset_region_season is None:
        print("Observations not selected for season correctly")
        sys.exit()

    # Calculate the anomalies for the observations.
    # Using try and except to catch any errors.
    try:
        # Calculate the mean climatology.
        obs_climatology = regridded_obs_dataset_region_season.mean("time")

        # Calculate the anomalies.
        obs_anomalies = regridded_obs_dataset_region_season - obs_climatology

    except:
        print("Error calculating anomalies for observations")
        sys.exit()

    # If obs_anomalies is empty, print a warning and exit the program.
    if obs_anomalies is None:
        print("Observations anomalies not calculated correctly")
        sys.exit()

    # Calculate the annual mean anomalies for the observations.
    # Using try and except to catch any errors.
    # By shifting the dataset back by a number of months
    # and then taking the yearly mean.
    try:
        if season == "DJFM" or season == "NDJFM":
            obs_anomalies_annual = obs_anomalies.shift(time = -3).resample(time = "Y").mean("time")
        elif season == "DJF" or season == "NDJF":
            obs_anomalies_annual = obs_anomalies.shift(time = -2).resample(time = "Y").mean("time")
        else :
            obs_anomalies_annual = obs_anomalies.resample(time = "Y").mean("time")

    except:
        print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

    # If obs_anomalies_annual is empty, print a warning and exit the program.
    if obs_anomalies_annual is None:
        print("Observations annual mean anomalies not calculated correctly")
        sys.exit()

    # Select the forecast range.
    # Using try and except to catch any errors.
    try:
        # extract the first year of the forecast range.
        forecast_range_start = int(forecast_range.split("-")[0])
        forecast_range_end = int(forecast_range.split("-")[1])

        # Echo the forecast range to the user.
        print("Forecast range: " + str(forecast_range_start) + "-" + str(forecast_range_end))

        # Calculate the rolling mean to take for the anomalies.
        rolling_mean_range = forecast_range_end - forecast_range_start + 1

        # Echo the rolling mean range to the user.
        print("Rolling mean range: " + str(rolling_mean_range))

        # Select the forecast range.
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time = rolling_mean_range, center = True).mean()

    except:
        print("Error selecting forecast range")
        sys.exit()

    # If obs_anomalies_annual_forecast_range is empty, print a warning and exit the program.
    if obs_anomalies_annual_forecast_range is None:
        print("Observations forecast range not selected correctly")
        sys.exit()

    # Return the observations.
    return obs_anomalies_annual_forecast_range


# Break this function up into smaller functions.
# ---------------------------------------------
# Function to load the observations.
def check_file_exists(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        sys.exit()

def regrid_observations(obs_dataset):
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
        print(f"Error regridding observations: {e}")
        sys.exit()


def select_region(regridded_obs_dataset, region_grid):
    try:

        # Echo the dimensions of the region grid
        print(f"Region grid dimensions: {region_grid}")

        # Define lon1, lon2, lat1, lat2
        lon1, lon2 = region_grid['lon1'], region_grid['lon2']
        lat1, lat2 = region_grid['lat1'], region_grid['lat2']

        # # Roll longitude to 0-360 if necessary
        # if (regridded_obs_dataset.coords['lon'] < 0).any():
        #     regridded_obs_dataset.coords['lon'] = np.mod(regridded_obs_dataset.coords['lon'], 360)
        #     regridded_obs_dataset = regridded_obs_dataset.sortby(regridded_obs_dataset.longitude)

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
        print(f"Error selecting region: {e}")
        sys.exit()

def select_season(regridded_obs_dataset_region, season):
    try:
        # Extract the months from the season string
        if season == "DJF":
            months = [12, 1, 2]
        elif season == "MAM":
            months = [3, 4, 5]
        elif season == "JJA":
            months = [6, 7, 8]
        elif season == "SON":
            months = [9, 10, 11]
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
        print("Error selecting season")
        sys.exit()

def calculate_anomalies(regridded_obs_dataset_region_season):
    try:
        obs_climatology = regridded_obs_dataset_region_season.mean("time")
        obs_anomalies = regridded_obs_dataset_region_season - obs_climatology
        return obs_anomalies
    except:
        print("Error calculating anomalies for observations")
        sys.exit()

def calculate_annual_mean_anomalies(obs_anomalies, season):
    try:

        # echo the season to the user
        print("Season:", season)
        # Echo the dataset being processed to the user
        print("Calculating annual mean anomalies for observations")
        print(obs_anomalies)

        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        else:
            obs_anomalies_shifted = obs_anomalies

        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

def select_forecast_range(obs_anomalies_annual, forecast_range):
    try:
        forecast_range_start, forecast_range_end = map(int, forecast_range.split("-"))
        print("Forecast range:", forecast_range_start, "-", forecast_range_end)
        rolling_mean_range = forecast_range_end - forecast_range_start + 1
        print("Rolling mean range:", rolling_mean_range)
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time=rolling_mean_range, center = True).mean()
        return obs_anomalies_annual_forecast_range
    except:
        print("Error selecting forecast range")
        sys.exit()


# WRITE A FUNCTION WHICH WILL CHECK FOR NAN VALUES IN THE OBSERVATIONS DATASET
# IF THERE ARE NAN VALUES, THEN echo an error
# and exit the program
def check_for_nan_values(obs):
    try:
        if obs['var151'].isnull().values.any():
            print("Error: NaN values in observations")
            sys.exit()
    except:
        print("Error checking for NaN values in observations")
        sys.exit()


def process_observations(
    variable, region, region_grid, forecast_range, season, observations_path, obs_var_name
):
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

    try:
        # Load the observations dataset
        # if variable is tas or sfcWind, we need to select the variable from the observations dataset
        if variable in ["tas", "sfcWind"]:
            obs_dataset = xr.open_dataset(observations_path, chunks={"time": 50})[variable]
        else:
            obs_dataset = xr.open_dataset(observations_path, chunks={"time": 50})
    except:
        print("Error loading observations dataset")
        sys.exit()

    # print the observations dataset
    print("Observations dataset:", obs_dataset)

    # Check for NaN values in the observations dataset
    # print
    print("Checking for NaN values in observations dataset")
    #check_for_nan_values(obs_dataset)

    # Regrid the observations to the model grid
    regridded_obs_dataset = regrid_observations(obs_dataset)

    # print the regridded observations dataset
    print("Regridded observations dataset:", regridded_obs_dataset)
    print("checking for NaN values in regridded observations dataset")
    #check_for_nan_values(regridded_obs_dataset)

    # Select the region
    regridded_obs_dataset_region = select_region(regridded_obs_dataset, region_grid)

    # Print the dataset being processed to the user
    print("Processing dataset before season:", regridded_obs_dataset_region)
    print("checking for NaN values in regridded observations dataset region")
    # print the values of var151 for the dataset
    print(regridded_obs_dataset_region['var151'])

    # Select the season
    regridded_obs_dataset_region_season = select_season(regridded_obs_dataset_region, season)

    # Print the dataset being processed to the user
    print("Processing dataset:", regridded_obs_dataset_region_season)
    print("checking for NaN values in regridded observations dataset region season")
    #check_for_nan_values(regridded_obs_dataset_region_season)

    # Calculate the anomalies
    obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)
    print("checking for NaN values in observations anomalies")
    #check_for_nan_values(obs_anomalies)

    # Calculate the annual mean anomalies
    obs_anomalies_annual = calculate_annual_mean_anomalies(obs_anomalies, season)
    print("checking for NaN values in observations annual anomalies")
    #check_for_nan_values(obs_anomalies_annual)

    print(obs_anomalies_annual['var151'].values)

    # Select the forecast range
    obs_anomalies_annual_forecast_range = select_forecast_range(obs_anomalies_annual, forecast_range)
    print("checking for NaN values in observations annual anomalies forecast range")
    # check_for_nan_values(obs_anomalies_annual_forecast_range)

    # print the var151 values
    print(obs_anomalies_annual_forecast_range['var151'].values)

    return obs_anomalies_annual_forecast_range

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

    # print the dimensions of the observations data
    print("Observations dimensions:", obs_data.dims)

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

    # print the dimensions of the observations data
    print("Observations dimensions:", obs_data.dims)
    print("Observations variables:", obs_data)

    # Print all of the latitude values
    print("Observations latitude values:", obs_data.lat.values)
    print("Observations longitude values:", obs_data.lon.values)

    # Select the first timestep of the observations
    obs_data_first = obs_data.isel(time=-1)

    # Select the variable to be plotted
    # and convert to hPa
    obs_var = obs_data_first["var151"]/100

    # print the value of the variable
    print("Observations variable:", obs_var.values)

    # print the dimensions of the observations data
    print("Observations dimensions:", obs_data_first)

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

def plot_model_data(model_data, models, gif_plots_path):
    """
    Plots the first timestep of the model data as a single subplot.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    gif_plots_path (str): The path to the directory where the plots will be saved.
    """

    # if the gif_plots_path directory does not exist
    if not os.path.exists(gif_plots_path):
        # Create the directory
        os.makedirs(gif_plots_path)

    # initialize an empty list to store the ensemble members
    ensemble_members = []

    # initialize a dictionary to store the count of ensemble members
    # for each model
    ensemble_members_count = {}

    # Initialize a dictionary to store the filepaths
    # of the plots for each model
    filepaths = []

    # For each model
    for model in models:
        model_data_combined = model_data[model]

        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        for member in model_data_combined:
            ensemble_members.append(member)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            years = member.time.dt.year.values

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Conver the ensemble members counts dictionary to a list of tuples
    ensemble_members_count_list = [(model, count) for model, count in ensemble_members_count.items()]

    # Conver the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # take the ensemble mean over the members
    ensemble_mean = ensemble_members.mean(axis=0)

    # # print the values of lat and lon
    # print("lat values", ensemble_mean[0, :, 0])
    # print("lon values", ensemble_mean[0, 0, :])

    # lat_test = ensemble_mean[0, :, 0]
    # lon_test = ensemble_mean[0, 0, :]

    # print the dimensions of the model data
    print("ensemble mean shape", np.shape(ensemble_mean))

    # Extract the years from the model data
    # print the values of the years
    print("years values", years)
    print("years shape", np.shape(years))
    print("years type", type(years))


    # set the vmin and vmax values
    vmin = -500
    vmax = 500

    # Loop over the years array
    for year in years:
        # print the year
        print("year", year)

        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot the ensemble mean on the subplot
        # for the specified year
        # Check that the year index is within the range of the years array
        if year < years[0] or year > years[-1]:
            continue

        # Find the index of the year in the years array
        year_index = np.where(years == year)[0][0]

        # Plot the ensemble mean on the subplot
        # for the specified year
        c = ax.contourf(lon, lat, ensemble_mean[year_index, :, :], transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax, norm=plt.Normalize(vmin=vmin, vmax=vmax))

        # Add coastlines and gridlines to the plot
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Annotate the plot with the year
        ax.annotate(f"{year}", xy=(0.01, 0.92), xycoords='axes fraction', fontsize=16)

        # Set up the filepath for saving
        filepath = os.path.join(gif_plots_path, f"{year}.png")
        # Save the figure
        fig.savefig(filepath)

        # Add the filepath to the list of filepaths
        filepaths.append(filepath)

        # Add coastlines and gridlines to the plot
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Annotate the plot with the year
        ax.annotate(f"{year}", xy=(0.01, 0.92), xycoords='axes fraction', fontsize=16)

        # Set up the filepath for saving
        filepath = os.path.join(gif_plots_path, f"{year}_anomalies.png")
        # Save the figure
        fig.savefig(filepath)

        # Add the filepath to the list of filepaths
        filepaths.append(filepath)

    # Create the gif
    # Using the function defined above
    make_gif(gif_plots_path)

    # Show the plot
    # plt.show()


# We want to define a function
# Which takes as input the observed and model data
# Ensures that these are the same shape, format and dimensions
# And then calculates the spatial correlations between the two datasets
def calculate_spatial_correlations(observed_data, model_data, models, region, forecast_range, season, variable):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. Before calculating the spatial correlations between the two datasets.
    
    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    region (str): The region to be plotted.
    forecast_range (str): The forecast range to be plotted.
    season (str): The season to be plotted.
    variable (str): The variable to be plotted.

    Returns:
    spatial_correlations (dict): A dictionary containing the spatial correlations between the observed and model data.
    
    """

    # First process the model data
    # Taking the equally weighted ensemble mean
    # For the ensemble members in all of the models specified
    # Initialize a list for the ensemble members
    ensemble_members = []

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Set the ensemble members count to zero
        # if the model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0
        
        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            years = member.time.dt.year.values

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # Convert the ensemble members counts dictionary to a list of tuples
    ensemble_members_count_list = [(model, count) for model, count in ensemble_members_count.items()]

    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(ensemble_mean, coords=member.coords, dims=member.dims)


    # Print the shape of the ensemble mean dataset
    print("ensemble mean shape", np.shape(ensemble_mean))
    print("ensemble mean type", type(ensemble_mean))

    # Print the shape and type of the observed data
    print("observed data shape", (observed_data.dims))
    print("observed data type", type(observed_data))
    print("observed data values", observed_data['var151'].values)

    # For the observed data
    # Identify the years where there are Nan values
    # and print these years
    # Loop over the years
    print("Checking for Nan values in the observed data")
    for year in observed_data.time.dt.year.values:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # If there are any Nan values in the data
        if np.isnan(data['var151'].values).any():
            # Print the year
            print(year)
        # if there are no Nan values in the data for a year
        # then print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no nan for this year")

    #         # exit the script
    #         # sys.exit()


    # Print the shape and type of the model data lats and lons
    print("model lat shape", np.shape(lat))
    print("model lat type", type(lat))
    print("model lat values", lat)

    print("model lon shape", np.shape(lon))
    print("model lon type", type(lon))
    print("model lon values", lon)

    # Print the shape and type of the model data years
    print("model years shape", np.shape(years))
    print("model years type", type(years))
    print("model years values", years)

    # Do the same for the observed data
    # Extract the lat and lon values
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values

    # Extract the years
    obs_years = observed_data.time.dt.year.values

    # Print the shape and type of the observed data lats and lons
    print("observed lat shape", np.shape(obs_lat))
    print("observed lat type", type(obs_lat))
    print("observed lat values", obs_lat)

    print("observed lon shape", np.shape(obs_lon))
    print("observed lon type", type(obs_lon))
    print("observed lon values", obs_lon)

    # Print the shape and type of the observed data years
    print("observed years shape", np.shape(obs_years))
    print("observed years type", type(obs_years))
    print("observed years values", obs_years)

    # Now they have the same shape, we want to make sure that they are on the same grid system
    # obs uses 0 to 360
    # model uses -180 to 180
    # we want to convert the obs from 0 to 360 to -180 to 180
    # so that they are on the same grid system
    # Convert the observed data lons from 0 to 360 to -180 to 180
    obs_lon = np.where(obs_lon > 180, obs_lon - 360, obs_lon)

    # print the shape and type of the transformed observed data lons
    print("observed lon shape", np.shape(obs_lon))
    print("observed lon type", type(obs_lon))
    print("observed lon values", obs_lon)
    
    # now print the model data lons
    print("model lon shape", np.shape(lon))
    print("model lon type", type(lon))
    print("model lon values", lon)

    # Now we want to make sure that the years in the time dimension are the same for the observed and model data
    # Select only where the years are the same
    # First find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, years)

    # Print the years in both
    print("years in both", years_in_both)

    # Now select only the years that are in both the observed and model data
    # For the observed data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))

    # For the model data
    ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))


    print("Checking for Nan values in the observed data after selecting only the years that are in both the observed and model data")
    for year in observed_data.time.dt.year.values:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # If there are any Nan values in the data
        if np.isnan(data['var151'].values).any():
            # Print the year
            print(year)

    # extract the years from the observed data
    observed_data_years = observed_data.time.dt.year.values
    # print these years
    print("observed data years", observed_data_years)

    # extract the years from the model data
    ensemble_mean_years = ensemble_mean.time.dt.year.values
    # print these years
    print("model data years", ensemble_mean_years)

    # Print the shape and type of the observed data
    print("observed data shape", (observed_data.dims))
    print("observed data type", type(observed_data))

    # Print the shape and type of the model data
    print("ensemble mean shape", np.shape(ensemble_mean))
    print("ensemble mean type", type(ensemble_mean))

    # Convert both the observed and model data to numpy arrays
    observed_data_array = observed_data['var151'].values
    ensemble_mean_array = ensemble_mean.values


    # Print the dimensions of these arrays
    print("observed data", observed_data_array)
    print("observed data shape", np.shape(observed_data_array))
    print("ensemble mean shape", np.shape(ensemble_mean_array))



    # Now calculate the correlations between the observed and model data
    # Given that for the model data
    # the first dimension is the time
    # the second dimension is the lat
    # the third dimension is the lon
    # and for the observed data
    # the first dimension is the time
    # the second dimension is the lon
    # the third dimension is the lat




# define a main function
def main():
    """Main function for the program.
    
    This function parses the arguments from the command line
    and then calls the functions to load and process the data.
    """

    # Create a usage statement for the script.
    USAGE_STATEMENT = """python functions.py <variable> <region> <forecast_range> <season>"""

    # Check if the number of arguments is correct.
    if len(sys.argv) != 5:
        print(USAGE_STATEMENT)
        sys.exit()

    # Make the plots directory if it doesn't exist.
    if not os.path.exists(dic.plots_dir):
        os.makedirs(dic.plots_dir)

    # Parse the arguments from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", help="variable", type=str)
    parser.add_argument("region", help="region", type=str)
    parser.add_argument("forecast_range", help="forecast range", type=str)
    parser.add_argument("season", help="season", type=str)
    args = parser.parse_args()

    # Print the arguments to the screen.
    print("variable = ", args.variable)
    print("region = ", args.region)
    print("forecast range = ", args.forecast_range)
    print("season = ", args.season)

    # Load the data.
    datasets = load_data(dic.base_dir, dic.test_model, args.variable, args.region, args.forecast_range, args.season)

    # # Print the datasets.
    # print(datasets)
    # # Dimensions of the datasets
    # print(datasets["BCC-CSM2-MR"])
    # # Print the shape
    # print(datasets["BCC-CSM2-MR"])

    # Process the model data.
    variable_data, model_time = process_data(datasets, args.variable)

    # Print the processed data.
    print(variable_data)
    print(model_time)
    # print(variable_data["BCC-CSM2-MR"])
    # print(model_time["BCC-CSM2-MR"])

    # Choose the obs path based on the variable
    if args.variable == "psl":
        obs_path = dic.obs_psl
    elif args.variable == "tas":
        obs_path = dic.obs_tas
    elif args.variable == "sfcWind":
        obs_path = dic.obs_sfcWind
    elif args.variable == "rsds":
        obs_path = dic.obs_rsds
    else:
        print("Error: variable not found")
        sys.exit()

    # choose the obs var name based on the variable
    if args.variable == "psl":
        obs_var_name = dic.psl_label
    elif args.variable == "tas":
        obs_var_name = dic.tas_label
    elif args.variable == "sfcWind":
        obs_var_name = dic.sfc_wind_label
    elif args.variable == "rsds":
        obs_var_name = dic.rsds_label
    else:
        print("Error: variable not found")
        sys.exit()

    # Process the observations.
    obs = process_observations(args.variable, args.region, dic.north_atlantic_grid, args.forecast_range, args.season, obs_path, obs_var_name)

    # Print the processed observations.
    print("obs = ", obs)
    print("dimensions = ", obs.dims)

    # Print the dimensions of the variable data.
    print("variable_data dimensions = ", variable_data)

    # Call the function to calculate the ACC
    spatial_correlations = calculate_spatial_correlations(obs, variable_data, dic.test_model, args.region, args.forecast_range, args.season, args.variable)



# Call the main function.
if __name__ == "__main__":
    main()
