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
                variable_data = variable_data[:, 0, 0] / 100

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
            "latitude": (["latitude"], np.arange(0, 360.1, 2.5)),
            "longitude": (["longitude"], np.arange(-90, 90.1, 2.5))
        })
        regridded_obs_dataset = obs_dataset.interp(
            latitude=regrid_example_dataset.latitude,
            longitude=regrid_example_dataset.longitude
        )
        return regridded_obs_dataset
    except:
        print("Error regridding observations")
        sys.exit()

def select_region(regridded_obs_dataset, region_grid):
    try:
        regridded_obs_dataset_region = regridded_obs_dataset.sel(
            lat=slice(region_grid[0], region_grid[1]),
            lon=slice(region_grid[2], region_grid[3])
        )
        return regridded_obs_dataset_region
    except:
        print("Error selecting region")
        sys.exit()

def select_season(regridded_obs_dataset_region, season):
    try:
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(
            time=regridded_obs_dataset_region["time.season"] == season
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
        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_annual = obs_anomalies.shift(time=-3).resample(time="Y").mean("time")
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_annual = obs_anomalies.shift(time=-2).resample(time="Y").mean("time")
        else:
            obs_anomalies_annual = obs_anomalies.resample(time="Y").mean("time")
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
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(
            time=rolling_mean_range, center=True
        ).mean()
        return obs_anomalies_annual_forecast_range
    except:
        print("Error selecting forecast range")
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
    # function code here
def process_observations(
    variable, region, region_grid, forecast_range, season, observations_path, obs_var_name
):
    # Check if the observations file exists
    check_file_exists(observations_path)

    # Load the observations dataset
    obs_dataset = xr.open_dataset(observations_path, chunks={"time": 50})
    check_file_exists(obs_dataset)

    # Regrid the observations to the model grid
    regridded_obs_dataset = regrid_observations(obs_dataset)

    # Select the region
    regridded_obs_dataset_region = select_region(regridded_obs_dataset, region_grid)

    # Select the season
    regridded_obs_dataset_region_season = select_season(regridded_obs_dataset_region, season)

    # Calculate the anomalies
    obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)

    # Calculate the annual mean anomalies
    obs_anomalies_annual = calculate_annual_mean_anomalies(obs_anomalies, season)

    # Select the forecast range
    obs_anomalies_annual_forecast_range = select_forecast_range(obs_anomalies_annual, forecast_range)

    return obs_anomalies_annual_forecast_range

# def Call_scripts_to_process_obs(
#     variable, region, region_grid, forecast_range, season, observations_path, obs_var_name
# ):
#     processed_observations = process_observations(
#         variable, region, region_grid, forecast_range, season, observations_path, obs_var_name
#     )
#     # Perform additional processing or save the processed observations as needed
#     pass


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

    # Process the data.
    variable_data, model_time = process_data(datasets, args.variable)

    # Print the processed data.
    print(variable_data)
    print(model_time)
    print(variable_data["BCC-CSM2-MR"])
    print(model_time["BCC-CSM2-MR"])

# Call the main function.
if __name__ == "__main__":
    main()
