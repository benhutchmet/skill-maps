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
from _datetime import datetime
import scipy.stats as stats
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image
from sklearn.utils import resample
from scipy.stats import pearsonr
import random

import matplotlib.cm as mpl_cm
import matplotlib
import cartopy.crs as ccrs
import iris
import iris.coord_categorisation as coord_cat
import iris.plot as iplt
import scipy
import pdb
import iris.quickplot as qplt

from tqdm import tqdm

# # Import CDO
# from cdo import *

# cdo = Cdo()

# # Import specific functions
# from nao_skill_functions import calculate_spna_index

# # Install imageio
# # ! pip install imageio
# import imageio.v3 as iio

# Set the path to imagemagick
rcParams["animation.convert_path"] = r"/usr/bin/convert"

# Local imports
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dic

# We want to write a function that takes a data directory and list of models
# which loads all of the individual ensemble members into a dictionary of datasets /
# grouped by models
# the arguments are:
# base_directory: the base directory where the data is stored
# models: a list of models to load
# variable: the variable to load, extracted from the command line
# forecast_range: the forecast range to load, extracted from the command line
# season: the season to load, extracted from the command line


def load_data(
    base_directory, models, variable, region, forecast_range, season, level=None
):
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
        Level: The level to load, extracted from the command line. Default is None.

    Returns:
        A dictionary of datasets grouped by models.
    """

    # Create an empty dictionary to store the datasets.
    datasets_by_model = {}

    # initialise a counter
    counter = 0

    # Loop over the models.
    for model in models:

        # Create an empty list to store the datasets for this model.
        datasets_by_model[model] = []

        # If the level is not None, then we want to load the data for the specified level
        if level is not None and level != 85000:
            files_path = (
                base_directory
                + "/"
                + variable
                + "/"
                + model
                + "/"
                + region
                + "/"
                + f"years_{forecast_range}"
                + "/"
                + season
                + "/"
                + f"plev_{level}"
                + "/"
                + "outputs"
                + "/"
                + "mergetime"
                + "/"
                + "*.nc"
            )
            print("Searching for files in ", files_path)
        else:
            # create the path to the files for this model
            files_path = (
                base_directory
                + "/"
                + variable
                + "/"
                + model
                + "/"
                + region
                + "/"
                + f"{forecast_range}"
                + "/"
                + season
                + "/"
                + "outputs"
                + "/"
                + "*anoms*.nc"
            )

        # #print the path to the files
        # print("Searching for files in ", files_path)

        # Create a list of the files for this model.
        files = glob.glob(files_path)

        # print the files path
        # print(f"files path {files_path}")
        # print(f"files {files}")

        # print the len of files for the model
        print(f"Number of files for {model}: {len(files)}")

        # if the list of files is empty, #print a warning and
        # exit the program
        if len(files) == 0:
            print("No files found for " + model)
            raise AttributeError("No files found for " + model)

        # #print the files to the screen.
        # print("Files for " + model + ":", files)

        # Loop over the files.
        for file in tqdm(files):

            # #print the file to the screen.
            # print(file)

            # Conditional statement to ensure that models are common to all variables
            # if model == "CMCC-CM2-SR5":
            #     # Don't use the files containing r11 and above or r2?i?
            #     if re.search(r"r1[1-9]", file) or re.search(r"r2.i.", file):
            #         print("Skipping file", file)
            #         continue
            # elif model == "EC-Earth3":
            #     # Don't use the files containing r?i2 or r??i2
            #     if re.search(r"r.i2", file) or re.search(r"r..i2", file):
            #         print("Skipping file", file)
            #         continue
            # elif model == "FGOALS-f3-L":
            #     # Don't use files containing r1-6i? or r??i?
            #     if any(re.search(fr"r{i}i.", file) for i in range(1, 7)) or re.search(r"r..i.", file):
            #         print("Skipping file", file)
            #         continue

            # check that the file exists
            # if it doesn't exist, #print a warning and
            # exit the program
            if not os.path.exists(file):
                # print("File " + file + " does not exist")
                sys.exit()

            # Load the dataset.
            dataset = xr.open_dataset(file, chunks={"time": 50, "lat": 100, "lon": 100})

            # Hard code psl to be extracted from the dataset
            dataset_var = dataset[variable]

            # Extract the years from the dataset.
            years = dataset_var["time.year"].values

            # If time range is 2-9
            # then remove 2020 and years following 2020 from the dataset
            if forecast_range == "2-9":
                # print("Removing 2020 and years following 2020 from the years array")
                # print("Years before removing 2020 and years following 2020: ", years)
                years = years[years < 2020]

            # Assert that there are at least 10 years in the dataset.
            assert len(years) >= 10, f"Less than 10 years found in {file}"

            # Assert that years does not have any duplicates.
            assert len(years) == len(set(years)), f"Duplicate years found in {file}"

            # Check if there are any gaps of more than one year between the years
            if not np.all(np.diff(years) <= 1):
                print(f"Non-consecutive years found in {file}: {years}")
                # Find where the years are not consecutive
                index = np.where(np.diff(years) != 1)[0]

                # Print the years where the years are not consecutive
                print(
                    f"Non-consecutive years found in {file}: {years[index - 1]}-{years[index]}-{years[index + 1]}"
                )

                # print a warning and exit the program
                print(f"Member: {file} has non-consecutive years")
                print("Will not be included in the analysis")

                # increment the counter
                counter += 1

                print(f"Counter: {counter}")
                continue

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

    # print(f"Dataset type: {type(datasets_by_model)}")

    def process_model_dataset(dataset, variable, attributes):
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
                # print("Error converting from Pa to hPa")
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
        elif variable == "tos":
            # Extract the variable
            variable_data = dataset["tos"]
        elif variable == "ua":
            # Extract the variable
            variable_data = dataset["ua"]
        elif variable == "va":
            # Extract the variable
            variable_data = dataset["va"]
        else:
            # print("Variable " + variable + " not recognised")
            sys.exit()

        # If variable_data is empty, #print a warning and exit the program.
        if variable_data is None:
            # print("Variable " + variable + " not found in dataset")
            sys.exit()

        # Extract the time dimension.
        model_time = dataset["time"].values
        # Set the type for the time dimension.
        model_time = model_time.astype("datetime64[Y]")

        # If model_time is empty, #print a warning and exit the program.
        if model_time is None:
            # print("Time not found in dataset")
            sys.exit()

        # Set up the attributes for the variable.
        variable_data.attrs = attributes

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
                # Extract the attributes from the dataset.
                attributes = dataset.attrs
                # Process the dataset
                variable_data, model_time = process_model_dataset(
                    dataset, variable, attributes
                )
                # Append the processed data to the lists.
                variable_data_by_model[model].append(variable_data)
                model_time_by_model[model].append(model_time)
        except Exception as e:
            # print(f"Error processing dataset for model {model}: {e}")
            # print("Exiting the program")
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
        # print(f"File {file_path} does not exist")
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

        regrid_example_dataset = xr.Dataset(
            {
                "lon": (["lon"], np.arange(0.0, 359.9, 2.5)),
                "lat": (["lat"], np.arange(90.0, -90.1, -2.5)),
            }
        )
        regridded_obs_dataset = obs_dataset.interp(
            lon=regrid_example_dataset.lon, lat=regrid_example_dataset.lat
        )
        return regridded_obs_dataset

    except Exception as e:
        # print(f"Error regridding observations: {e}")
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
        # print(f"Region grid dimensions: {region_grid}")

        # Define lon1, lon2, lat1, lat2
        lon1, lon2 = region_grid["lon1"], region_grid["lon2"]
        lat1, lat2 = region_grid["lat1"], region_grid["lat2"]

        # dependent on whether this wraps around the prime meridian
        if lon1 < lon2:
            regridded_obs_dataset_region = regridded_obs_dataset.sel(
                lon=slice(lon1, lon2), lat=slice(lat1, lat2)
            )
        else:
            # If the region crosses the prime meridian, we need to do this in two steps
            # Select two slices and concatenate them together
            regridded_obs_dataset_region = xr.concat(
                [
                    regridded_obs_dataset.sel(
                        lon=slice(0, lon2), lat=slice(lat1, lat2)
                    ),
                    regridded_obs_dataset.sel(
                        lon=slice(lon1, 360), lat=slice(lat1, lat2)
                    ),
                ],
                dim="lon",
            )

        return regridded_obs_dataset_region
    except Exception as e:
        # print(f"Error selecting region: {e}")
        sys.exit()


# Define a function which regrids and selects the region according to the specified gridspec
# The same is done for the model data


def regrid_and_select_region_nao(variable, region, observations_path, level=85000):
    """
    Regrids and selects the region according to the specified gridspec.

    Parameters
    ----------
    variable : str
        Variable name.
    region : str
        Region name.
    observations_path : str
        Path to the observations.
    level : str, optional
        Level name. The default is None.

    Returns
    -------
    regrid_obs_path : str
        Path to the regridded observations.
    """

    # Check whether the gridspec path exists for the specified region
    gridspec_path = f"/home/users/benhutch/gridspec/gridspec-{region}.txt"

    if not os.path.exists(gridspec_path):
        print("The gridspec path does not exist for the specified region: ", region)
        sys.exit()

    # Form the wind speed variables list
    wind_speed_variables = ["ua", "va", "var131", "var132"]

    if variable in wind_speed_variables and level is None:
        # print the variable
        print(wind_speed_variables)
        print(level)
        
        print("The level must be specified for the wind speed variables")
        sys.exit()

    # Form the regrid sel region path accordingly
    if variable in wind_speed_variables:
        regrid_obs_path = f"/gws/nopw/j04/canari/users/benhutch/ERA5/{region}_regrid_sel_region_{variable}_{level}.nc"
    else:
        regrid_obs_path = f"/gws/nopw/j04/canari/users/benhutch/ERA5/{region}_regrid_sel_region.nc"

    # Check whether the regrid sel region path exists
    if not os.path.exists(regrid_obs_path):
        print("The regrid sel region path does not exist")
        print("Regridding and selecting the region")

        # Regrid and select the region using CDO
        cdo.remapbil(gridspec_path, input=observations_path, output=regrid_obs_path)

    return regrid_obs_path


# Using cdo to do the regridding and selecting the region


def regrid_and_select_region(observations_path, region, obs_var_name, level=None):
    """
    Uses CDO remapbil and a gridspec file to regrid and select the correct region for the obs dataset. Loads for the specified variable.

    Parameters:
    observations_path (str): The path to the observations dataset.
    region (str): The region to select.
    obs_var_name (str): The name of the variable in the observations dataset.
    level (str): The level to load, extracted from the command line. Default is None.

    Returns:
    xarray.Dataset: The regridded and selected observations dataset.
    """

    # FIXME:
    # By default set level to None
    if level is not None:
        level = None

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
    elif region == "north-sea":
        gridspec = gridspec_path + "/" + "gridspec-north-sea.txt"
    elif region == "central-europe":
        gridspec = gridspec_path + "/" + "gridspec-central-europe.txt"
    elif region == "snao-south":
        gridspec = gridspec_path + "/" + "gridspec-snao-south.txt"
    elif region == "snao-north":
        gridspec = gridspec_path + "/" + "gridspec-snao-north.txt"
    else:
        print("Invalid region")
        sys.exit()

    # echo the gridspec file
    print("Gridspec file:", gridspec)

    # Check that the gridspec file exists
    if not os.path.exists(gridspec):
        print("Gridspec file does not exist")
        sys.exit()

    # if obs_var_name is ua or va
    if obs_var_name in ["ua"]:
        obs_var_name = "var131"
    elif obs_var_name in ["va"]:
        obs_var_name = "var132"
    elif obs_var_name in ["pr"]:
        obs_var_name = "var228"

    # If the variable is ua or va, then we want to select the plev=85000
    if obs_var_name in ["ua", "va", "var131", "var132", "var228", "pr"]:
        print("Variable is ua or va, creating new file name")
        if level is not None:
            regrid_sel_region_file = (
                "/gws/nopw/j04/canari/users/benhutch/ERA5/"
                + region
                + "_"
                + "regrid_sel_region_"
                + obs_var_name
                + "_"
                + level
                + ".nc"
            )
        else:
            regrid_sel_region_file = (
                "/gws/nopw/j04/canari/users/benhutch/ERA5/"
                + region
                + "_"
                + "regrid_sel_region_"
                + obs_var_name
                + ".nc"
            )
    else:
        print("Variable is not ua or va, creating new file name")
        regrid_sel_region_file = (
            "/gws/nopw/j04/canari/users/benhutch/ERA5/" + region + "_" + "regrid_sel_region" + ".nc"
        )

    # print the regrid_sel_region_file
    print("Regrid and select region file:", regrid_sel_region_file)

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
    if obs_var_name not in [
        "psl",
        "tas",
        "t2m",
        "sfcWind",
        "rsds",
        "tos",
        "ua",
        "va",
        "var131",
        "var132",
        "var228",
        "pr",
    ]:
        print("Invalid variable name:", obs_var_name)
        sys.exit()

    # Translate the variable name to the name used in the obs dataset
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "t2m":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    elif obs_var_name == "ua":
        obs_var_name = "var131"
    elif obs_var_name == "va":
        obs_var_name = "var132"
    elif obs_var_name == "var131":
        obs_var_name = "var131"
    elif obs_var_name == "var132":
        obs_var_name = "var132"
    elif obs_var_name == "pr":
        obs_var_name = "var228"
    elif obs_var_name == "var228":
        obs_var_name = "var228"
    else:
        print("Invalid variable name")
        sys.exit()

    # Load the regridded and selected region dataset
    # for the provided variable
    try:

        # If variable is ua or va, then we want to load the dataset differently
        if obs_var_name in ["var131", "var132"]:
            # if regrid_sel_region_file is a grib file, then we need to use the grib option
            if regrid_sel_region_file.endswith(".grib"):
                # Load the dataset for the selected variable
                regrid_sel_region_dataset_combine = xr.open_dataset(
                    regrid_sel_region_file,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )[obs_var_name]
            else:
                regrid_sel_region_dataset_combine = xr.open_dataset(
                    regrid_sel_region_file, chunks={"time": 100, "lat": 100, "lon": 100}
                )[obs_var_name]
        # Elif expver is a dimension in the dataset
        elif "expver" in xr.open_dataset(regrid_sel_region_file).dims:

            # Load the dataset for the selected variable
            regrid_sel_region_dataset = xr.open_mfdataset(
                regrid_sel_region_file,
                combine="by_coords",
                parallel=True,
                chunks={"time": 100, "lat": 100, "lon": 100},
            )[obs_var_name]

            print("Dataset loaded: ", regrid_sel_region_dataset.values)

            # Combine the two expver variables
            regrid_sel_region_dataset_combine = regrid_sel_region_dataset.sel(
                expver=1
            ).combine_first(regrid_sel_region_dataset.sel(expver=5))

        else:
            # Load the dataset for the selected variable
            regrid_sel_region_dataset_combine = xr.open_dataset(
                regrid_sel_region_file, chunks={"time": 100, "lat": 100, "lon": 100}
            )[obs_var_name]

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
        elif season == "ONDJFM":
            months = [10, 11, 12, 1, 2, 3]
        elif season == "AMJJAS":
            months = [4, 5, 6, 7, 8, 9]
        else:
            raise ValueError("Invalid season")

        # Select the months from the dataset
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(
            time=regridded_obs_dataset_region["time.month"].isin(months)
        )

        return regridded_obs_dataset_region_season
    except:
        # print("Error selecting season")
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
        # print("Error calculating anomalies for observations")
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

    # if the type of obs_anomalies is an iris cube, then convert to an xarray dataset
    if type(obs_anomalies) == iris.cube.Cube:
        obs_anomalies = xr.DataArray.from_iris(obs_anomalies)

    try:
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM", "ONDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF", "ONDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-1)
        else:
            obs_anomalies_shifted = obs_anomalies

        # Calculate the annual mean anomalies
        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        print("Error shifting and calculating annual mean anomalies for observations")
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
        if "-" in forecast_range:
            forecast_range_start, forecast_range_end = map(
                int, forecast_range.split("-")
            )
            # print("Forecast range:", forecast_range_start, "-", forecast_range_end)

            rolling_mean_range = forecast_range_end - forecast_range_start + 1
            # print("Rolling mean range:", rolling_mean_range)
        else:
            rolling_mean_range = int(forecast_range)

        # P[RINT THE TIME DIMENSION OF THE OBSERVATIONS
        print("Time dimension of obs:", obs_anomalies_annual.time.values)

        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(
            time=rolling_mean_range, center=True
        ).mean()

        return obs_anomalies_annual_forecast_range
    except Exception as e:
        # print("Error selecting forecast range:", e)
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
        if obs["msl"].isnull().values.any():
            # print("Error: NaN values in observations")
            sys.exit()
    except Exception as e:
        # print("Error checking for NaN values in observations:", e)
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


# Define a function to load the obs data into Iris cubes


def load_obs(variable, regrid_obs_path):
    """
    Loads the obs data into Iris cubes.

    Parameters
    ----------
    variable : str
        Variable name.
    regrid_obs_path : str
        Path to the regridded observations.

    Returns
    -------
    obs : iris.cube.Cube
        Observations.
    """

    # Verify that the regrid obs path exists
    if not os.path.exists(regrid_obs_path):
        print("The regrid obs path does not exist")
        sys.exit()

    if variable not in dic.var_name_map:
        print("The variable is not in the dictionary")
        sys.exit()

    # Extract the variable name from the dictionary
    obs_variable = dic.var_name_map[variable]

    # If the obs variable is 'var228'
    if obs_variable == "var228":
        # Set the regrid obs path
        regrid_obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_var228.nc"

    if obs_variable in dic.obs_ws_var_names:
        print("The obs variable is a wind speed variable")
        print("Loading regrid obs file using xarray: ", regrid_obs_path)

        # Load the regrid obs file into an Iris cube
        obs = iris.load_cube(regrid_obs_path, obs_variable)
    else:
        # Load using xarray
        obs = xr.open_mfdataset(
            regrid_obs_path,
            combine="by_coords",
            parallel=True,
            chunks={"time": "auto", "lat": "auto", "lon": "auto"},
        )[obs_variable]

        # Combine the two expver variables
        obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

        # Convert to an Iris cube
        obs = obs.to_iris()

        # if the type of obs is not a cube, then exit
        if type(obs) != iris.cube.Cube:
            print("The type of obs is not a cube")
            sys.exit()

    return obs


# We want to write a function which reads and processes the observations
# then returns the obs anomaly field
def read_obs(
    variable,
    region,
    forecast_range,
    season,
    observations_path,
    start_year,
    end_year,
    level=None,
):
    """
    Processes the observations to have the same grid as the model data
    using CDO. Then selects the region and season. Then calculates the
    anomaly field using the climatology. Then calculates the annual
    mean of the anomaly field. Then selects the forecast range (e.g.
    years 2-5). Then selects the season. Then returns the anomaly field.


    Parameters
    ----------
    variable : str
        Variable name.
    region : str
        Region name.
    forecast_range : str
        Forecast range.
    season : str
        Season name.
    observations_path : str
        Path to the observations.
    start_year : str
        Start year.
    end_year : str
        End year.
    level : str, optional
        Level name. The default is None.

    Returns
    -------
    obs_anomaly : iris.cube.Cube
        Anomaly field.

    """

    # First check that the obs_path exists
    if not os.path.exists(observations_path):
        print("The observations path does not exist")
        sys.exit()

    # print the variable
    print(f"variable {variable}")

    #  if the variable is ua, replace with var131
    if variable in ["ua", "var131"]:
        variable = "var131"
    elif variable in ["va", "var132"]:
        variable = "var132"
    else:
        variable = variable

    # Get the path to the regridded and selected region observations
    regrid_obs_path = regrid_and_select_region_nao(
        variable, region, observations_path, level=level
    )

    # Load the obs data into Iris cubes
    obs = load_obs(variable, regrid_obs_path)

    # print the obs pre level extract
    print("Obs pre level extract:", obs)

    # If the level is not None, then extract the level
    if level is not None:
        obs = obs.extract(iris.Constraint(air_pressure=int(level)))

    # print the obs post level extract
    print("Obs post level extract:", obs)

    # Select the season
    if season not in dic.season_month_map:
        raise ValueError("The season is not in the dictionary")
        sys.exit()

    # Extract the months corresponding to the season
    months = dic.season_month_map[season]

    # Set up the iris constraint for the start and end years
    # Create the date time objects
    start_date = datetime(int(start_year), 12, 1)
    end_date = datetime(int(end_year), 3, 31)
    iris_constraint = iris.Constraint(
        time=lambda cell: start_date <= cell.point <= end_date
    )

    # print the start date
    print("Start date:", start_date)
    print("End date:", end_date)

    # print the obs
    print("Obs:", obs)

    # Apply the iris constraint to the cube
    obs = obs.extract(iris_constraint)

    # Set up the iris constraint
    iris_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)
    # Apply the iris constraint to the cube
    obs = obs.extract(iris_constraint)

    # # Add a month coordinate to the cube
    # coord_cat.add_month(obs, 'time')

    # Calculate the seasonal climatology
    # First collapse the time dimension by taking the mean
    climatology = obs.collapsed("time", iris.analysis.MEAN)

    # Calculate the anomaly field
    obs_anomaly = obs - climatology

    # Calculate the annual mean anomalies
    obs_anomaly_annual = calculate_annual_mean_anomalies(obs_anomaly, season)

    # Select the forecast range
    obs_anomaly_annual_forecast_range = select_forecast_range(
        obs_anomaly_annual, forecast_range
    )

    # If the type of obs_anomaly_annual_forecast_range is not a cube, then convert to a cube
    # if type(obs_anomaly_annual_forecast_range) != iris.cube.Cube:
    #     obs_anomaly_annual_forecast_range = xr.DataArray.to_iris(obs_anomaly_annual_forecast_range)

    # Return the anomaly field
    return obs_anomaly_annual_forecast_range


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
        # print("Invalid variable name")
        sys.exit()

    try:
        # Load the observations dataset
        obs_dataset = xr.open_dataset(observations_path, chunks={"time": 50})[
            obs_var_name
        ]

        ERA5 = xr.open_mfdataset(
            observations_path, combine="by_coords", chunks={"time": 50}
        )[obs_var_name]
        ERA5_combine = ERA5.sel(expver=1).combine_first(ERA5.sel(expver=5))
        ERA5_combine.load()
        ERA5_combine.to_netcdf(observations_path + "_copy.nc")

        # #print the dimensions of the observations dataset
        # #print("Observations dataset:", obs_dataset.dims)

        # Check for NaN values in the observations dataset
        # check_for_nan_values(obs_dataset)

        return obs_dataset, ERA5_combine

    except Exception as e:
        # print(f"Error loading observations dataset: {e}")
        sys.exit()


# Call the functions to process the observations


def process_observations(
    variable,
    region,
    region_grid,
    forecast_range,
    season,
    observations_path,
    obs_var_name,
    plev=None,
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
        plev (int): The pressure level to select for the observations dataset. Defaults to None.

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

    # If the variable is ua or va, then we want to select the plev=85000
    # level for the observations dataset
    # Create the output file path

    # Process the observations using try and except to catch any errors
    try:
        # Regrid using CDO, select region and load observation dataset
        # for given variable
        obs_dataset = regrid_and_select_region(
            observations_path, region, obs_var_name, plev
        )

        print("Observations dataset:", obs_dataset.values)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_dataset")
        # check_for_nan_values(obs_dataset)
        if variable in ["ua", "va"]:
            # Use xarray to select the plev=85000 level
            print("Selecting plev=,", plev, "level for observations dataset")
            obs_dataset = obs_dataset.sel(plev=plev)

            # If the dataset contains more than one vertical level
            # then give an error and exit the program
            # if len(obs_dataset.plev) > 1:
            #     print("Error: More than one vertical level in observations dataset")
            #     sys.exit()

        # Select the season
        # --- Although will already be in DJFM format, so don't need to do this ---
        regridded_obs_dataset_region_season = select_season(obs_dataset, season)

        print(
            "Regridded and selected region dataset:",
            regridded_obs_dataset_region_season.values,
        )

        # # #print the dimensions of the regridded and selected region dataset
        # print("Regridded and selected region dataset:", regridded_obs_dataset_region_season.time)

        # # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in regridded_obs_dataset_region_season")
        # check_for_nan_values(regridded_obs_dataset_region_season)

        # Calculate anomalies
        obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)

        print("Observations anomalies:", obs_anomalies.values)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_anomalies")
        # check_for_nan_values(obs_anomalies)

        # Calculate annual mean anomalies
        obs_annual_mean_anomalies = calculate_annual_mean_anomalies(
            obs_anomalies, season
        )

        print("Observations annual mean anomalies:", obs_annual_mean_anomalies.values)

        # Check for NaN values in the observations dataset
        # #print("Checking for NaN values in obs_annual_mean_anomalies")
        # check_for_nan_values(obs_annual_mean_anomalies)

        # If the forecast range contains a hyphen
        if "-" in forecast_range:
            # Select the forecast range
            obs_anomalies_annual_forecast_range = select_forecast_range(
                obs_annual_mean_anomalies, forecast_range
            )
            # Check for NaN values in the observations dataset
            # #print("Checking for NaN values in obs_anomalies_annual_forecast_range")
            # check_for_nan_values(obs_anomalies_annual_forecast_range)

            print(
                "Observations anomalies annual forecast range:",
                obs_anomalies_annual_forecast_range.values,
            )

            # if the forecast range is "2-2" i.e. a year ahead forecast
            # then we need to shift the dataset by 1 year
            # where the model would show the DJFM average as Jan 1963 (s1961)
            # the observations would show the DJFM average as Dec 1962
            # so we need to shift the observations to the following year
            # if the forecast range is "2-2" and the season is "DJFM"
            # then shift the dataset by 1 year
            if forecast_range == "2-2" and season == "DJFM":
                obs_anomalies_annual_forecast_range = (
                    obs_anomalies_annual_forecast_range.shift(time=1)
                )

            # Save the processed observations dataset as a netCDF file
            # print that the file is being saved
            # Save the processed observations dataset as a netCDF file
            # Convert the variable to a DataArray object before saving
            # print("Saving processed observations dataset")
            # obs_anomalies_annual_forecast_range.to_netcdf(processed_obs_file)

            return obs_anomalies_annual_forecast_range
        else:
            print("just need single year forecast range - seasonal averages work")

            return obs_annual_mean_anomalies
    except Exception as e:
        # print(f"Error processing observations dataset: {e}")
        sys.exit()


def process_observations_timeseries(
    variable, region, forecast_range, season, observations_path
):
    """
    Processes the observations for a specific variable, region, forecast range, and season.

    Args:
        variable (str): The variable to process.
        region (str): The region to process.
        forecast_range (list): The forecast range to process.
        season (str): The season to process.
        observations_path (str): The path to the observations file.

    Returns:
        xarray.Dataset: The processed observations dataset.
    """

    # First check if the observations file exists
    check_file_exists(observations_path)

    # First use try and except to process the observations for a specific variable
    # and region
    try:
        # Regrid using CDO, select region and load observation dataset
        # for given variable
        obs_dataset = regrid_and_select_region(observations_path, region, variable)
    except Exception as e:
        print(f"Error processing observations dataset using CDO to regrid: {e}")
        sys.exit()

    # Then use try and except to process the observations for a specific season
    try:
        # Select the season
        obs_dataset_season = select_season(obs_dataset, season)
    except Exception as e:
        print(f"Error processing observations dataset selecting season: {e}")
        sys.exit()

    # Then use try and except to process the observations and calculate anomalies
    try:
        # Calculate anomalies
        obs_anomalies = calculate_anomalies(obs_dataset_season)
    except Exception as e:
        print(f"Error processing observations dataset calculating anomalies: {e}")
        sys.exit()

    # Then use try and except to process the observations and calculate annual mean anomalies
    try:
        # Calculate annual mean anomalies
        obs_annual_mean_anomalies = calculate_annual_mean_anomalies(
            obs_anomalies, season
        )
    except Exception as e:
        print(
            f"Error processing observations dataset calculating annual mean anomalies: {e}"
        )
        sys.exit()

    # Then use try and except to process the observations and select the forecast range
    try:
        # Select the forecast range
        obs_anomalies_annual_forecast_range = select_forecast_range(
            obs_annual_mean_anomalies, forecast_range
        )
    except Exception as e:
        print(f"Error processing observations dataset selecting forecast range: {e}")
        sys.exit()

    # Then use try and except to process the observations and shift the forecast range
    try:
        # if the forecast range is "2-2" i.e. a year ahead forecast
        # then we need to shift the dataset by 1 year
        # where the model would show the DJFM average as Jan 1963 (s1961)
        # the observations would show the DJFM average as Dec 1962
        # so we need to shift the observations to the following year
        # if the forecast range is "2-2" and the season is "DJFM"
        # then shift the dataset by 1 year
        if forecast_range == "2-2" and season == "DJFM":
            obs_anomalies_annual_forecast_range = (
                obs_anomalies_annual_forecast_range.shift(time=1)
            )
    except Exception as e:
        print(f"Error processing observations dataset shifting forecast range: {e}")
        sys.exit()

    # Then use try and except to process the gridbox mean of the observations
    try:
        # Calculate the gridbox mean of the observations
        obs_gridbox_mean = obs_anomalies_annual_forecast_range.mean(dim=["lat", "lon"])
    except Exception as e:
        print(f"Error processing observations dataset calculating gridbox mean: {e}")
        sys.exit()

    # Return the processed observations dataset
    return obs_gridbox_mean


# Define a new function which calculates the observed NAO index
# for a given season
# as azores minus iceland


def process_obs_nao_index(
    forecast_range, season, observations_path, variable="psl", nao_type="default"
):
    """
    Calculate the observed NAO index for a given season, using the pointwise definition of the summertime NAO index
    from Wang and Ting (2022).

    Parameters
    ----------
    forecast_range : str
        Forecast range to calculate the NAO index for, in the format 'YYYY-MM'.
    season : str
        Season to calculate the NAO index for, one of 'DJFM', 'MAM', 'JJA', 'SON'.
    observations_path : str
        Path to the observations file.
    azores_grid : tuple of float
        Latitude and longitude coordinates of the Azores grid point.
    iceland_grid : tuple of float
        Latitude and longitude coordinates of the Iceland grid point.
    snao_south_grid : tuple of float
        Latitude and longitude coordinates of the southern SNAO grid point.
    snao_north_grid : tuple of float
        Latitude and longitude coordinates of the northern SNAO grid point.
    variable : str, optional
        Name of the variable to use for the NAO index calculation, by default 'psl'.
    nao_type : str, optional
        Type of NAO index to calculate, by default 'default'. Also supports 'snao'.

    Returns
    -------
    float
        The observed NAO index for the given season and forecast range.
    """
    # If the NAO type is 'default'
    if nao_type == "default":
        print("Calculating observed NAO index using default definition")

        # Process the gridbox mean of the observations
        # for both the Azores and Iceland
        # Set up the region grid for the Azores
        region = "azores"
        obs_azores = process_observations_timeseries(
            variable, region, forecast_range, season, observations_path
        )
        # Set up the region grid for Iceland
        region = "iceland"
        obs_iceland = process_observations_timeseries(
            variable, region, forecast_range, season, observations_path
        )

        # Calculate the observed NAO index
        obs_nao_index = obs_azores - obs_iceland
    elif nao_type == "snao":
        print("Calculating observed NAO index using SNAO definition")

        # Process the gridbox mean of the observations
        # for both the southern and northern SNAO
        # Set up the region grid for the southern SNAO
        region = "snao-south"
        obs_snao_south = process_observations_timeseries(
            variable, region, forecast_range, season, observations_path
        )
        # Set up the region grid for the northern SNAO
        region = "snao-north"
        obs_snao_north = process_observations_timeseries(
            variable, region, forecast_range, season, observations_path
        )

        # Calculate the observed NAO index
        obs_nao_index = obs_snao_south - obs_snao_north
    else:
        print("Invalid NAO type")
        sys.exit()

    # Return the observed NAO index
    return obs_nao_index


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
    obs_data_mean = obs_data.mean(dim="time")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Plot the observations on the left subplot
    obs_data_mean.plot(
        ax=ax1, transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=-2, vmax=2
    )
    ax1.set_title("Observations")

    # Plot the model data on the right subplot
    variable_data.mean(dim=model_time).plot(
        ax=ax2, transform=ccrs.PlateCarree(), cmap="coolwarm", vmin=-2, vmax=2
    )
    ax2.set_title("Model Data")

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
    obs_var = obs_data_first["var151"] / 100

    # #print the value of the variable
    # #print("Observations variable:", obs_var.values)

    # #print the dimensions of the observations data
    # #print("Observations dimensions:", obs_data_first)

    # Create a figure with one subplot
    fig, ax = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot the observations on the subplot
    c = ax.contourf(
        obs_data_first.lon,
        obs_data_first.lat,
        obs_var,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
    )

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
    frames = [
        Image.open(os.path.join(frame_folder, f))
        for f in os.listdir(frame_folder)
        if f.endswith("_anomalies.png")
    ]
    frame_one = frames[0]
    # Save the frames as a gif
    frame_one.save(
        os.path.join(frame_folder, "animation.gif"),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=300,
        loop=0,
    )


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
    # print("ensemble mean shape", np.shape(ensemble_mean))

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
    observed_data = observed_data.sel(
        time=observed_data.time.dt.year.isin(years_in_both)
    )
    ensemble_mean = ensemble_mean.sel(
        time=ensemble_mean.time.dt.year.isin(years_in_both)
    )

    # remove the years with NaN values from the model data
    observed_data, ensemble_mean = remove_years_with_nans(observed_data, ensemble_mean)

    # convert to numpy arrays
    # and convert from pa to hpa
    obs_array = observed_data["var151"].values / 100
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
        fig, axs = plt.subplots(
            ncols=3,
            nrows=1,
            figsize=(18, 6),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

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
        c1 = axs[0].contourf(
            lon,
            lat,
            model[year_index, :, :],
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )

        # Add coastlines and gridlines to the plot
        axs[0].coastlines()
        # axs[0].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[0].annotate(
            f"{year}", xy=(0.01, 0.92), xycoords="axes fraction", fontsize=16
        )
        # annotate the plot with model
        # in the top right corner
        axs[0].annotate(
            f"{models[0]}", xy=(0.8, 0.92), xycoords="axes fraction", fontsize=16
        )

        # Plot the observations on the subplot
        c2 = axs[1].contourf(
            lon,
            lat,
            obs[year_index, :, :],
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )

        # Add coastlines and gridlines to the plot
        axs[1].coastlines()
        # axs[1].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[1].annotate(
            f"{year}", xy=(0.01, 0.92), xycoords="axes fraction", fontsize=16
        )
        # annotate the plot with obs
        # in the top right corner
        axs[1].annotate(f"obs", xy=(0.8, 0.92), xycoords="axes fraction", fontsize=16)

        # Plot the anomalies on the subplot
        c3 = axs[2].contourf(
            lon,
            lat,
            model[year_index, :, :] - obs[year_index, :, :],
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )

        # Add coastlines and gridlines to the plot
        axs[2].coastlines()
        # axs[2].gridlines(draw_labels=True)

        # Annotate the plot with the year
        axs[2].annotate(
            f"{year}", xy=(0.01, 0.92), xycoords="axes fraction", fontsize=16
        )
        axs[2].annotate(f"anoms", xy=(0.8, 0.92), xycoords="axes fraction", fontsize=16)

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

            # If time is zero
            # then continue with the loop
            if len(member.time) == 0:
                continue

            # If there are any NaN values in the model data
            if np.isnan(member).any():
                print("There are NaN values in the model data for model ", model)
                # if there are only NaN values in the model data
                if np.all(np.isnan(member)):
                    print(
                        "All values in the model data are NaN values for model ", model
                    )
                    # continue with the loop
                    continue

            # Extract the years
            years = member.time.dt.year.values

            # # print the model name
            # # #print("model name:", model)
            # print("years len:", len(years), "for model:", model)

            # If the years has duplicate values
            if len(years[years < 2020]) != len(set(years[years < 2020])):
                # Raise a value error
                print("The models years has duplicate values for model ", model)
                # continue with the loop
                continue

            # If there is a gap of more than 1 year in the years
            # then raise a value error
            # Check whats going on with Canesm5
            if np.any(np.diff(years[years < 2020]) > 1):
                print(
                    "There is a gap of more than 1 year in the years for model ", model
                )
                # continue with the loop
                continue

            # if len years is less than 10
            # print the model name, member name, and len years
            if len(years) < 10:
                print("model name:", model)
                print("member name:", member)
                print("years len:", len(years))

            # Append the years to the list of years
            years_list.append(years)

    # # #print the years list for debugging
    # print("years list:", years_list)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))

    # # #print the common years for debugging
    # print("Common years:", common_years)
    # print("Common years type:", type(common_years))
    # print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:

            # if time is zero
            # then continue with the loop
            if len(member.time) == 0:
                continue

            # Extract the years
            years = member.time.dt.year.values

            # #print the years extracted from the model
            # #print('model years', years)
            # #print('model years shape', np.shape(years))

            # If the years has duplicate values
            if len(years[years < 2020]) != len(set(years[years < 2020])):
                # Raise a value error
                print("The models years has duplicate values for model ", model)
                # continue with the loop
                continue

            # If there is a gap of more than 1 year in the years
            # then raise a value error
            if np.any(np.diff(years[years < 2020]) > 1):
                print(
                    "There is a gap of more than 1 year in the years for model ", model
                )
                # print the values of the years where the gap is greater than 1
                # Convert the tuple returned by np.where() to a NumPy array
                indices = np.array(np.where(np.diff(years[years < 2020]) > 1))

                # Subtract 2 from the indices to adjust for the gap
                adjusted_indices = indices - 2

                print("adjusted indices:", adjusted_indices[0][0])

                # Print the years where the gap is greater than 1, using the adjusted indices
                print(
                    "years where gap is greater than 1:",
                    years[adjusted_indices[0][0] :],
                )
                # continue with the loop
                continue

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


# checking for Nans in observed data
def remove_years_with_nans_nao(
    observed_data,
    model_data,
    models,
    NAO_matched=False,
    matched_var_ensemble_members=None,
):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        model_data (dict): The model data.
        models (list): The list of models to be plotted.
        variable (str): the variable name.
        NAO_matched (bool): Whether or not the data has been matched to the NAO index.
        matched_var_ensemble_members (list): The list of ensemble members that have been matched to the NAO index. Defaults to None.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
    """

    # If NAO_matched is False
    if NAO_matched == False:
        # Check that there are no NaN values in the model data
        # Loop over the models
        for model in models:
            # Extract the model data
            model_data_by_model = model_data[model]

            # Loop over the ensemble members in the model data
            for member in model_data_by_model:

                # # Modify the time dimension
                # if type is not already datetime64
                # then convert the time type to datetime64
                if type(member.time.values[0]) != np.datetime64:
                    member_time = member.time.astype("datetime64[ns]")

                    # # Modify the time coordinate using the assign_coords() method
                    member = member.assign_coords(time=member_time)

                # Extract the years
                model_years = member.time.dt.year.values

                # If the years has duplicate values
                if len(model_years) != len(set(model_years)):
                    # Raise a value error
                    print(
                        "The models years has duplicate values for model "
                        + model
                        + "member "
                        + member.attrs["variant_label"]
                    )
                    # continue with the loop
                    continue

                # Only if there are no NaN values in the model data
                # Will we loop over the years
                if not np.isnan(member.values).any():
                    print("No NaN values in the model data")
                    # continue with the loop
                    continue

                print("NaN values in the model data")
                print("Model:", model)
                print("Member:", member)
                print("Looping over the years")
                # Loop over the years
                for year in model_years:
                    # Extract the data for the year
                    data = member.sel(time=f"{year}")

                    if np.isnan(data.values).any():
                        print("NaN values in the model data for this year")
                        print("Model:", model)
                        print("Year:", year)
                        if np.isnan(data.values).all():
                            print("All NaN values in the model data for this year")
                            print("Model:", model)
                            print("Year:", year)
                            # De-Select the year from the observed data
                            member = member.sel(time=member.time.dt.year != year)

                            print(year, "all NaN values for this year")
                    else:
                        print(year, "no NaN values for this year")

        # Extract the model years
        model_years = member.time.dt.year.values
    else:
        print("NAO_matched is True")
        print("Checking for NaN values in the xarray dataset")

        # if matched_var_ensemble_members is not None:
        # if there are any NaN values in the xarray dataset
        # Extract the years from the xarray dataset
        model_years = model_data.time.dt.year.values

        if matched_var_ensemble_members is not None:
            print("Aligining the years for the matched var ensemble members")

            # Extract the years
            members_model_years = matched_var_ensemble_members.time.values

            # If the years has duplicate values
            if len(members_model_years) != len(set(members_model_years)):
                # Raise a value error
                raise ValueError(
                    "The models years has duplicate values for model "
                    + model
                    + "member "
                    + member.attrs["variant_label"]
                )

            # Find the years that are in all of the models
            if np.array_equal(model_years, members_model_years) == False:
                print(
                    "The model years and the matched var ensemble members years are not the same"
                )
                print("Model years:", model_years)
                print("Matched var ensemble members years:", members_model_years)
                raise ValueError(
                    "The model years and the matched var ensemble members years are not the same"
                )

        # Loop over the years
        for year in model_years:
            # Extract the data for the year
            data = model_data.sel(time=f"{year}")

            # If there are any NaN values in the data
            if np.isnan(data["__xarray_dataarray_variable__"].values).any():
                # If there are only NaN values in the data
                if np.isnan(data["__xarray_dataarray_variable__"].values).all():
                    # Select the year from the observed data
                    model_data = model_data.sel(time=model_data.time.dt.year != year)

                    print(year, "all NaN values for this year")
            # if there are no NaN values in the data for a year
            # then #print the year
            # and "no nan for this year"
            # and continue the script
            else:
                print(year, "no NaN values for this year")

    # Now check that there are no NaN values in the observed data
    for year in observed_data.time.dt.year.values:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # print("data type", (type(data)))
        # print("data vaues", data)
        # print("data shape", np.shape(data))

        # If there are any NaN values in the data
        if np.isnan(data.values).any():
            # If there are only NaN values in the data
            if np.isnan(data.values).all():
                # Select the year from the observed data
                observed_data = observed_data.sel(
                    time=observed_data.time.dt.year != year
                )

                print(year, "all NaN values for this year")
        # if there are no NaN values in the data for a year
        # then #print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no NaN values for this year")

    # Set up the years to be returned
    obs_years = observed_data.time.dt.year.values

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # print the shape of the model data
    # print("model years shape", np.shape(model_data))
    # print("obs years shape", np.shape(obs_years))

    # if obs years and model years are not the same
    if np.array_equal(obs_years, model_years) == False:
        print("obs years and model years are not the same")
        print("Aligning the years")

        # Find the years that are in both the model data and the common years
        years_in_both = np.intersect1d(obs_years, model_years)

        # Select only those years from the model data
        observed_data = observed_data.sel(
            time=observed_data.time.dt.year.isin(years_in_both)
        )

        # if NAO_matched is False
        if NAO_matched == False:
            # for the model data
            for model in models:
                # Extract the model data
                model_data_by_model = model_data[model]

                # Loop over the ensemble members in the model data
                for member in model_data_by_model:
                    # Extract the years
                    model_years = member.time.dt.year.values

                    # Select only those years from the model data
                    member = member.sel(time=member.time.dt.year.isin(years_in_both))

                    # Add the member to the constrained data dictionary
                    if model not in constrained_data:
                        constrained_data[model] = []

                    # Append the member to the constrained data dictionary
                    constrained_data[model].append(member)
        else:
            # Select only those years from the model data
            constrained_data = model_data.sel(
                time=model_data.time.dt.year.isin(years_in_both)
            )

    return observed_data, constrained_data, matched_var_ensemble_members


# Define a new function to align forecast1, forecast2, and obs
# before converting each to numpy arrays
def align_forecast1_forecast2_obs(
    forecast1, forecast1_models, forecast2, forecast2_models, obs
):
    """
    After removing years with NaNs, aligns the forecast1, forecast2, and obs datasets by their time axis.

    Converts the forecast1, forecast2, and obs datasets to numpy arrays.

    Inputs:

        forecast1 (dict) = dictionary of forecast1 data, indexed by model

        forecast1_models (list) = list of models used as indices for forecast1

        forecast2 (dict) = dictionary of forecast2 data, indexed by model

        forecast2_models (list) = list of models used as indices for forecast2

        obs[time, lat, lon] (xarray.Dataset) = observations data

    Outputs:

        forecast1[members, time, lat, lon] (array) = forecast1 data, aligned by time axis

        forecast2[members, time, lat, lon] (array) = forecast2 data, aligned by time axis

        obs[time, lat, lon] (array) = observations data, aligned by time axis

        common_time (array) = an array of years common to all datasets

    """
    # Extract the obs years
    obs_years = obs.time.dt.year.values
    print("shape of obs years pre Nan removal", np.shape(obs_years))

    # Loop over the obs years and get rid of years containing only NaN values
    for year in obs_years:
        odata = obs.sel(time=f"{year}")

        # If there are any NaN values in the data
        if np.isnan(odata.values).any():
            print("NaN values in the obs data for this year: ", year)
            # If there are only NaN values in the data
            if np.isnan(odata.values).all():
                # Select the year from the observed data
                obs = obs.sel(time=obs.time.dt.year != year)

                print(year, "all NaN values for this year in the obs")
            else:
                print(year, "Not all NaN values for this year in the obs")
        else:
            print(year, "no NaN values for this year in the obs")

    # Extract the obs years
    obs_years = obs.time.dt.year.values
    print("shape of obs years post Nan removal", np.shape(obs_years))

    # Both forecast1 and forecast2 should have consistent years between models and members
    f1_years = forecast1[forecast1_models[0]][0].time.dt.year.values

    # if forecast 2 and forecast 2 models are not none
    if forecast2 is not None and forecast2_models is not None:
        f2_years = forecast2[forecast2_models[0]][0].time.dt.year.values

        # If the years are not the same
        if np.array_equal(f1_years, f2_years) == False:
            print("forecast1 and forecast2 years are not the same")
            print("forecast1 years:", f1_years)
            print("forecast2 years:", f2_years)
            # Find the common years between forecast1 and forecast2
            common_years = np.intersect1d(f1_years, f2_years)

            # Set up dictionaries for the forecast1 and forecast2 data
            forecast1_data_common = {}
            forecast2_data_common = {}

            # Select only those years from forecast1 and forecast2
            for model in forecast1_models:
                # Extract the forecast1 data
                forecast1_data = forecast1[model]

                if model not in forecast1_data_common:
                    forecast1_data_common[model] = []

                # Loop over the ensemble members in the forecast1 data
                for member in forecast1_data:
                    # Extract the years
                    years = member.time.dt.year.values

                    # Select only those years from the forecast1 data
                    member = member.sel(time=member.time.dt.year.isin(common_years))

                    # Append the member to the forecast1_data_common dictionary
                    forecast1_data_common[model].append(member)

            for model in forecast2_models:
                # Extract the forecast2 data
                forecast2_data = forecast2[model]

                if model not in forecast2_data_common:
                    forecast2_data_common[model] = []

                # Loop over the ensemble members in the forecast2 data
                for member in forecast2_data:
                    # Extract the years
                    years = member.time.dt.year.values

                    # Select only those years from the forecast2 data
                    member = member.sel(time=member.time.dt.year.isin(common_years))

                    # Append the member to the forecast2_data_common dictionary
                    forecast2_data_common[model].append(member)

            # Extract the forecast1 years
            f1_years = forecast1_data_common[forecast1_models[0]][0].time.dt.year.values

            # Extract the forecast2 years
            f2_years = forecast2_data_common[forecast2_models[0]][0].time.dt.year.values

            # If these are not the same, raise a value error
            if np.array_equal(f1_years, f2_years) == False:
                raise ValueError(
                    "forecast1 and forecast2 years are not the same after processing"
                )

    # If the forecast1 and forecast2 years are the same
    common_f_years = f1_years

    # If the obs and forecast1 years are not the same
    if np.array_equal(obs_years, common_f_years) == False:
        print("obs and forecast1 years are not the same")
        print("obs years:", obs_years)
        print("obs years shape:", np.shape(obs_years))
        print("commonf years:", common_f_years)
        print("commonf years shape:", np.shape(common_f_years))

        # Find the common years between obs and forecast1
        common_years = np.intersect1d(obs_years, common_f_years)

        # Select only those years from obs and forecast1
        obs = obs.sel(time=obs.time.dt.year.isin(common_years))

        # Initialize dictionaries for the forecast1 and forecast2 data
        forecast1_data_common = {}
        forecast2_data_common = {}

        for model in forecast1_models:
            # Extract the forecast1 data
            forecast1_data = forecast1[model]

            if model not in forecast1_data_common:
                forecast1_data_common[model] = []

            # Loop over the ensemble members in the forecast1 data
            for member in forecast1_data:
                # Extract the years
                years = member.time.dt.year.values

                # Select only those years from the forecast1 data
                member = member.sel(time=member.time.dt.year.isin(common_years))

                # Append the member to the forecast1_data_common dictionary
                forecast1_data_common[model].append(member)
        
        if forecast2 is not None and forecast2_models is not None:

            # For the forecast2 data
            for model in forecast2_models:
                # Extract the forecast2 data
                forecast2_data = forecast2[model]

                if model not in forecast2_data_common:
                    forecast2_data_common[model] = []

                # Loop over the ensemble members in the forecast2 data
                for member in forecast2_data:
                    # Extract the years
                    years = member.time.dt.year.values

                    # Select only those years from the forecast2 data
                    member = member.sel(time=member.time.dt.year.isin(common_years))

                    # Append the member to the forecast2_data_common dictionary
                    forecast2_data_common[model].append(member)

            f2_years = forecast2_data_common[forecast2_models[0]][0].time.dt.year.values

            if np.array_equal(obs_years, f2_years) == False:
                raise ValueError(
                "obs and forecast2 years are not the same after processing"
            )

            if np.array_equal(f1_years, f2_years) == False:
                raise ValueError(
                    "forecast1 and forecast2 years are not the same after processing"
                )

            forecast2 = forecast2_data_common

        # Extract the years for obs, forecast1, and forecast2
        obs_years = obs.time.dt.year.values
        f1_years = forecast1_data[forecast1_models[0]][0].time.dt.year.values


        # If these are not the same, raise a value error
        if np.array_equal(obs_years, f1_years) == False:
            raise ValueError(
                "obs and forecast1 years are not the same after processing"
            )

        # Set the forecast1 and forecast2 data to the common data
        forecast1 = forecast1_data_common

    # Now that the years are the same, we can convert to numpy arrays
    # First convert the obs to a numpy array
    obs = obs.values

    print("shape of obs array", np.shape(obs))

    # Extract the lats from the obs
    lats = obs.shape[1]
    lons = obs.shape[2]

    # Extract the years
    years = obs.shape[0]

    # Count the number of ensemble members for forecast1
    f1_ensemble_members = np.sum([len(forecast1[model]) for model in forecast1_models])

    # Create an empty array to store the forecast1 data
    f1 = np.empty((f1_ensemble_members, years, lats, lons))

    if forecast2 is not None and forecast2_models is not None:
        # Count the number of ensemble members for forecast2
        f2_ensemble_members = np.sum([len(forecast2[model]) for model in forecast2_models])

        # Create an empty array to store the forecast2 data
        f2 = np.empty((f2_ensemble_members, years, lats, lons))

        # initialize a variable to keep track of the ensemble member index
        member_index = 0

        # Loop over the forecast2 models
        for model in forecast2_models:
            # Extract the forecast2 data
            forecast2_data = forecast2[model]

            # Loop over the ensemble members in the forecast2 data
            for member in forecast2_data:
                # Extract the ensemble member index
                member_index += 1

                # Extract the data
                data = member.values

                # If the data has four dimensions
                if len(data.shape) == 4:
                    # Squeeze the data
                    data = np.squeeze(data)

                # Assign the data to the forecast2 array
                f2[member_index - 1, :, :, :] = data

    # initialize a variable to keep track of the ensemble member index
    member_index = 0

    # Loop over the forecast1 models
    for model in forecast1_models:
        # Extract the forecast1 data
        forecast1_data = forecast1[model]

        # Loop over the ensemble members in the forecast1 data
        for member in forecast1_data:
            # Extract the ensemble member index
            member_index += 1

            # Extract the data
            data = member.values

            # If the data has four dimensions
            if len(data.shape) == 4:
                # Squeeze the data
                data = np.squeeze(data)

            # Assign the data to the forecast1 array
            f1[member_index - 1, :, :, :] = data

    # Print the shapes of the forecast1 and forecast2 arrays
    print("shape of forecast1 array", np.shape(f1))
    # print("shape of forecast2 array", np.shape(f2))

    if forecast2 is not None and forecast2_models is not None:
        f2 = None

    # Return the forecast1, forecast2, and obs arrays
    return f1, f2, obs, common_years


# Define a new function to rescalse the NAO index for each year


def rescale_nao_by_year(
    year,
    obs_nao,
    ensemble_mean_nao,
    ensemble_members_nao,
    season,
    forecast_range,
    output_dir,
    lagged=False,
    omit_no_either_side=1,
):
    """
    Rescales the observed and model NAO indices for a given year and season, and saves the results to disk.

    Parameters
    ----------
    year : int
        The year for which to rescale the NAO indices.
    obs_nao : pandas.DataFrame
        A DataFrame containing the observed NAO index values, with a DatetimeIndex.
    ensemble_mean_nao : pandas.DataFrame
        A DataFrame containing the ensemble mean NAO index values, with a DatetimeIndex.
    ensemble_members_nao : dict
        A dictionary containing the NAO index values for each ensemble member, with a DatetimeIndex.
    season : str
        The season for which to rescale the NAO indices. Must be one of 'DJF', 'MAM', 'JJA', or 'SON'.
    forecast_range : int
        The number of months to forecast ahead.
    output_dir : str
        The directory where to save the rescaled NAO indices.
    lagged : bool, optional
        Whether to use lagged NAO indices in the rescaling. Default is False.

    Returns
    -------
    None
    """

    # Print the year for which the NAO indices are being rescaled
    print(f"Rescaling NAO indices for {year}")

    # Extract the model years
    model_years = ensemble_mean_nao.time.dt.year.values

    # Ensure that the type of ensemble_mean_nao and ensemble_members_nao is a an array
    if (
        type(ensemble_mean_nao)
        and type(ensemble_members_nao) != np.ndarray
        and type(obs_nao) != np.ndarray
    ):
        AssertionError(
            "The type of ensemble_mean_nao and ensemble_members_nao and obs_nao is not a numpy array"
        )
        sys.exit()

    # If the year is not in the ensemble members years
    if year not in model_years:
        # Print a warning and exit the program
        print(f"Year {year} is not in the ensemble members years")
        sys.exit()

    # Extract the index for the year
    year_index = np.where(model_years == year)[0]

    # Extract the ensemble members for the year
    ensemble_members_nao_year = ensemble_members_nao[:, year_index]

    # Compute the ensemble mean NAO for this year
    ensemble_mean_nao_year = ensemble_members_nao_year.mean(axis=0)

    # Set up the indicies for the cross-validation
    # In the case of the first year
    if year == model_years[0]:
        print("Cross-validation case for the first year")
        print("Removing the first year and:", omit_no_either_side, "years forward")
        # Set up the indices to use for the cross-validation
        # Remove the first year and omit_no_either_side years forward
        cross_validation_indices = np.arange(0, omit_no_either_side + 1)
    # In the case of the last year
    elif year == model_years[-1]:
        print("Cross-validation case for the last year")
        print("Removing the last year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Remove the last year and omit_no_either_side years backward
        cross_validation_indices = np.arange(-1, -omit_no_either_side - 2, -1)
    # In the case of any other year
    else:
        # Omit the year and omit_no_either_side years forward and backward
        print("Cross-validation case for any other year")
        print("Removing the year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Use the year index and omit_no_either_side years forward and backward
        cross_validation_indices = np.arange(
            year_index - omit_no_either_side, year_index + omit_no_either_side + 1
        )

    # Log which years are being used for the cross-validation
    print("Cross-validation indices:", cross_validation_indices)
    print("shape of ensemble members pre delete: ", ensemble_members_nao)
    # Extract the ensemble members for the cross-validation
    # i.e. don't use the years given by the cross_validation_indices
    ensemble_members_nao_array_cross_val = np.delete(
        ensemble_members_nao, cross_validation_indices, axis=1
    )
    # Take the mean over the ensemble members
    # to get the ensemble mean nao for the cross-validation
    print("shape of cross val remaining: ", ensemble_members_nao_array_cross_val.shape)
    ensemble_mean_nao_cross_val = ensemble_members_nao_array_cross_val.mean(axis=0)

    print(
        "shape of ensemble members post delete: ", ensemble_members_nao_array_cross_val
    )

    # Remove the indicies from the obs_nao
    obs_nao_cross_val = np.delete(obs_nao, cross_validation_indices, axis=0)

    # Calculate the pearson correlation coefficient between the observed and model NAO indices
    acc_score, p_value = stats.pearsonr(obs_nao_cross_val, ensemble_mean_nao_cross_val)

    print("acc score :", acc_score)
    print

    # Calculate the RPS score
    rps_score = calculate_rps(
        acc_score, ensemble_members_nao_array_cross_val, obs_nao_cross_val
    )

    print("rps score :", rps_score)
    print("ensemble mean nao year pre *rps :", ensemble_mean_nao_year)

    # Compute the rescaled NAO index for the year
    signal_adjusted_nao_index = ensemble_mean_nao_year * rps_score

    print("signal adjusted nao index post *rpc :", signal_adjusted_nao_index)

    return signal_adjusted_nao_index, ensemble_mean_nao_year


# Write a function which performs the NAO matching
# TODO: Modify variable names


def nao_matching_other_var(
    rescaled_model_nao,
    model_nao,
    psl_models,
    match_variable_model,
    match_variable_obs,
    match_var_base_dir,
    match_var_models,
    match_var_obs_path,
    region,
    season,
    forecast_range,
    start_year,
    end_year,
    output_dir,
    save_dir,
    lagged_years=None,
    lagged_nao=False,
    no_subset_members=20,
    level=None,
    ensemble_mean_nao=None,
    load_files=True,
):
    """
    Performs the NAO matching for the given variable. E.g. T2M. By default will select from the lagged ensemble members.

    Parameters
    ----------
    rescaled_model_nao : xarray.DataArray
        Rescaled NAO index.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    psl_models : list
        List of models to be plotted. Different models for each variable.
    match_variable_model : str
        Variable name for the variable which will undergo matching for the model.
    match_variable_obs : str
        Variable name for the variable which will undergo matching for the obs.
    match_var_base_dir : str
        Path to the base directory containing the variable data.
    match_var_models : list
        List of models to be plotted for the matched variable. Different models for each variable.
    region : str
        Region name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    start_year : int
        Start year.
    end_year : int
        End year.
    output_dir : str
        Path to the output directory.
    save_dir : str`
        Path to the save directory.
    years : list, optional
        List of years to loop over. The default is None.
    lagged : bool, optional
        Flag to indicate whether the ensemble is lagged or not. The default is False.
    no_subset_members : int, optional
        Number of ensemble members to subset. The default is 20.
    level : int, optional
        Pressure level. The default is None. For the matched variable.

    Returns
    -------
    None
    """

    # Set up the folder to save the data
    save_dir = f"{save_dir}/{match_variable_model}/{region}/{season}/{forecast_range}/{start_year}-{end_year}"
    # If the folder does not exist
    if not os.path.exists(save_dir):
        # Create the folder
        os.makedirs(save_dir)

    # Set up the filename
    filename_mean = f"{match_variable_model}_{region}_{season}_{forecast_range}_{start_year}-{end_year}_matched_var_ensemble_mean.nc"
    filename_members = f"{match_variable_model}_{region}_{season}_{forecast_range}_{start_year}-{end_year}_matched_var_ensemble_members.nc"

    # if laggged_nao is True
    if lagged_nao == True:
        # Set up the filename
        filename_mean = f"{match_variable_model}_{region}_{season}_{forecast_range}_{start_year}-{end_year}_matched_var_ensemble_mean_lagged.nc"
        filename_members = f"{match_variable_model}_{region}_{season}_{forecast_range}_{start_year}-{end_year}_matched_var_ensemble_members_lagged.nc"

    # Set up the path to save the data
    save_path_mean = f"{save_dir}/{filename_mean}"
    save_path_members = f"{save_dir}/{filename_members}"

    # If the file already exists
    if (
        os.path.exists(save_path_mean)
        and os.path.exists(save_path_members)
        and load_files == True
    ):
        # Print a notification
        print(f"The files {filename_mean} and {filename_members} already exist")
        print("Loading the files")
        # Load the file
        matched_var_ensemble_mean = xr.open_dataset(save_path_mean)
        matched_var_ensemble_members = xr.open_dataset(save_path_members)
    else:
        # Print the variable which is being matched
        print(f"Performing NAO matching for {match_variable_model}")

        # Extract the obs data for the matched variable
        match_var_obs_anomalies = read_obs(
            match_variable_obs,
            region,
            forecast_range,
            season,
            match_var_obs_path,
            start_year,
            end_year,
            level=level,
        )

        # Extract the model data for the matched variable
        match_var_datasets = load_data(
            match_var_base_dir,
            match_var_models,
            match_variable_model,
            region,
            forecast_range,
            season,
            level=level,
        )

        # process the model data
        match_var_model_anomalies, _ = process_data(
            match_var_datasets, match_variable_model
        )

        # Make sure that each of the models have the same time period
        match_var_model_anomalies = constrain_years(
            match_var_model_anomalies, match_var_models
        )

        # Remove years containing NaN values from the obs and model data
        # and align the time periods
        match_var_obs_anomalies, match_var_model_anomalies, _ = (
            remove_years_with_nans_nao(
                match_var_obs_anomalies, match_var_model_anomalies, match_var_models
            )
        )

        # Now we want to make sure that the match_var_model_anomalies and the model_nao
        # have the same models
        model_nao_constrained, match_var_model_anomalies_constrained, models_in_both = (
            constrain_models_members(
                model_nao, psl_models, match_var_model_anomalies, match_var_models
            )
        )

        # Now we want to ensure that the match var model data is lagged
        _, _, match_var_model_anomalies_constrained = form_ensemble_members_list(
            match_var_model_anomalies_constrained, match_var_models, lagged=True, lag=4
        )

        # Make sure that the years for rescaled_model_nao and model_nao
        # and match_var_model_anomalies_constrained are the same
        rescaled_model_years = rescaled_model_nao.time.dt.year.values
        model_nao_years = model_nao_constrained[psl_models[0]][0].time.dt.year.values
        match_var_model_years = match_var_model_anomalies_constrained[
            0
        ].time.dt.year.values

        # # If the years are not equal
        # if not np.array_equal(rescaled_model_years, model_nao_years) or not np.array_equal(rescaled_model_years, match_var_model_years):
        #     # Print a warning and exit the program
        #     print("The years for the rescaled model NAO, the model NAO and the matched variable model anomalies are not equal")

        #     # Extract the years which are in the rescaled model nao and the model nao
        #     # Constrain the rescaled NAO and the model NAO constrained to the same years as match var model years
        #     model_nao_constrained, match_var_model_anomalies_constrained, years_in_both \
        #                         = constrain_years_psl_match_var(model_nao_constrained, model_nao_years, models_in_both,
        #                                                             match_var_model_anomalies_constrained, match_var_model_years, models_in_both)
        #     # Set rescalled_model_nao to the years_in_both
        #     rescaled_model_years = years_in_both

        # Set up the years to loop over
        years = rescaled_model_years

        # if lagged_nao is True
        if lagged_nao == True:
            # lagged years is just the years skipping the first 0, 1 and 2 values
            lagged_years = years[3:]

        # Set up the lats and lons for the array
        lats = match_var_model_anomalies_constrained[0].lat.values
        lons = match_var_model_anomalies_constrained[0].lon.values

        # Set up the empty arrays to be filled
        matched_var_ensemble_mean_array = np.empty((len(years), len(lats), len(lons)))

        # Set up an array to fill the matched variable ensemble members
        matched_var_ensemble_members_array = np.empty(
            (len(years), no_subset_members, len(lats), len(lons))
        )

        # Extract the coords for the first years=years of the match_var_model_anomalies_constrained
        # Select the years from the match_var_model_anomalies_constrained
        # Select only the data for the years in the 'years' array
        match_var_model_anomalies_constrained_years = (
            match_var_model_anomalies_constrained[0].sel(
                time=match_var_model_anomalies_constrained[0].time.dt.year.isin(years)
            )
        )

        # if the match variable model is ua or va
        if match_variable_model in ["ua", "va"]:
            print("match variable model is ua or va")
            print("squeezing single dimesion for plev")

            # Squeeze the single dimension for plev
            match_var_model_anomalies_constrained_years = (
                match_var_model_anomalies_constrained_years.squeeze()
            )

        # Extract the coords for the first years=years of the model_nao_constrained
        coords = match_var_model_anomalies_constrained_years.coords
        dims = match_var_model_anomalies_constrained_years.dims

        print("looping over the years:", years)

        # Prior to the loop, form the lagged ensemble members list
        # for NAO containing the individual members
        _, ensemble_members_count_nao_lagged, nao_lagged_ensemble_members = (
            form_ensemble_members_list(
                model_nao_constrained, models_in_both, lagged=True, lag=4
            )
        )

        # Loop over the years and perform the NAO matching
        for i, year in enumerate(years):
            print("Selecting members for year: ", year)

            # Extract the members with the closest NAO index to the rescaled NAO index
            # for the given year
            smallest_diff = calculate_closest_members(
                year,
                rescaled_model_nao,
                model_nao_constrained,
                models_in_both,
                season,
                forecast_range,
                output_dir,
                lagged=True,
                no_subset_members=no_subset_members,
                nao_lagged_ensemble_members=nao_lagged_ensemble_members,
            )

            # Using the closest NAO index members, extract the same members
            # for the matched variable
            matched_var_members = extract_matched_var_members(
                year, match_var_model_anomalies_constrained, smallest_diff, lagged=True
            )

            matched_var_members_array = np.empty((len(matched_var_members)))

            # Now we want to calculate the ensemble mean for the matched variable for this year
            matched_var_ensemble_mean, matched_var_ensemble_members = (
                calculate_matched_var_ensemble_mean(matched_var_members, year)
            )

            # # Extract the member_coords from matched_var_ensemble_members
            # member_coords = matched_var_ensemble_members.coords
            # member_dims = matched_var_ensemble_members.dims

            # Squeeze the matched_var_ensemble_members array to remove the single dimension of year
            matched_var_ensemble_members = np.squeeze(matched_var_ensemble_members)

            # Append the matched_var_ensemble_mean to the array
            matched_var_ensemble_mean_array[i] = matched_var_ensemble_mean
            matched_var_ensemble_members_array[i] = matched_var_ensemble_members

        # Convert the matched_var_ensemble_mean_array to an xarray DataArray
        matched_var_ensemble_mean = xr.DataArray(
            matched_var_ensemble_mean_array, coords=coords, dims=dims
        )

        # # Ensure that member_coords has dimension len(years) for the year dimension
        # member_coords = matched_var_ensemble_members.coords
        # # add a new coordinate for the year dimension
        # member_coords['year'] = years
        # # Change the order of member dims, so that member_dims=['year', 'member', 'lat', 'lon']
        # member_dims = ['year', 'member', 'lat', 'lon']

        # Set up the member coords
        member_coords = {
            "time": years,
            "member": matched_var_ensemble_members.member.values,
            "lat": matched_var_ensemble_members.lat.values,
            "lon": matched_var_ensemble_members.lon.values,
        }

        # Set up the member dims
        member_dims = ("time", "member", "lat", "lon")

        # Convert the matched_var_ensemble_members_array to an xarray DataArray
        matched_var_ensemble_members = xr.DataArray(
            matched_var_ensemble_members_array, coords=member_coords, dims=member_dims
        )

        # Save the data
        matched_var_ensemble_mean.to_netcdf(save_path_mean)

        # Save the data
        matched_var_ensemble_members.to_netcdf(save_path_members)

    # Open the dataset
    matched_var_ensemble_mean = xr.open_dataset(save_path_mean)

    # Open the dataset
    matched_var_ensemble_members = xr.open_dataset(save_path_members)

    # Return the matched_var_ensemble_mean
    return matched_var_ensemble_mean, matched_var_ensemble_members


# Define a function which will make sure that the model_nao and the match_var_model_anomalies
# have the same models and members
def constrain_models_members(
    model_nao, psl_models, match_var_model_anomalies, match_var_models
):
    """
    Makes sure that the model_nao and the match_var_model_anomalies have the same models and members.
    """

    # Set up dictionaries to store the models and members
    psl_models_dict = {}
    match_var_models_dict = {}

    # If the two models lists are not equal
    if not np.array_equal(psl_models, match_var_models):
        # Print a warning and exit the program
        print("The two models lists are not equal")
        print("Constraining the models")

        # Find the models that are in both the psl_models and the match_var_models
        models_in_both = np.intersect1d(psl_models, match_var_models)
    else:
        # Set the models_in_both to the psl_models
        print("The two models lists are equal")
        models_in_both = psl_models

    # Loop over the models in the model_nao
    for model in models_in_both:
        print("Model:", model)

        # Append the model to the psl_models_dict
        psl_models_dict[model] = []

        # Append the model to the match_var_models_dict
        match_var_models_dict[model] = []

        # Extract the NAO data for the model
        model_nao_by_model = model_nao[model]

        # Extract the match_var_model_anomalies for the model
        match_var_model_anomalies_by_model = match_var_model_anomalies[model]

        # Extract a list of the variant labels for the model
        variant_labels_psl = [
            member.attrs["variant_label"] for member in model_nao_by_model
        ]
        print("Variant labels for the model psl:", variant_labels_psl)
        # Extract a list of the variant labels for the match_var_model_anomalies
        variant_labels_match_var = [
            member.attrs["variant_label"]
            for member in match_var_model_anomalies_by_model
        ]
        print("Variant labels for the model match_var:", variant_labels_match_var)

        # If the two variant labels lists are not equal
        if not set(variant_labels_psl) == set(variant_labels_match_var):
            # Print a warning and exit the program
            print("The two variant labels lists are not equal")
            print("Constraining the variant labels")

            # Find the variant labels that are in both the variant_labels_psl and the variant_labels_match_var
            variant_labels_in_both = np.intersect1d(
                variant_labels_psl, variant_labels_match_var
            )

            # Now filter the model_nao data
            psl_models_dict[model] = filter_model_data_by_variant_labels(
                model_nao_by_model, variant_labels_in_both, psl_models_dict[model]
            )

            # Now filter the match_var_model_anomalies data
            match_var_models_dict[model] = filter_model_data_by_variant_labels(
                match_var_model_anomalies_by_model,
                variant_labels_in_both,
                match_var_models_dict[model],
            )

        else:
            print("The two variant labels lists are equal")
            # Loop over the members in the model_nao_by_model
            for member in model_nao_by_model:
                # Append the member to the psl_models_dict
                psl_models_dict[model].append(member)

            # Loop over the members in the match_var_model_anomalies_by_model
            for member in match_var_model_anomalies_by_model:
                # if the type of the member is not datetime64
                if type(member.time.values[0]) != np.datetime64:
                    # Extract the time values as a datetime64
                    member_time = member.time.astype("datetime64[ns]")

                    # Add the time values back to the member
                    member = member.assign_coords(time=member_time)

                # Append the member to the match_var_models_dict
                match_var_models_dict[model].append(member)

    return psl_models_dict, match_var_models_dict, models_in_both


def filter_model_data_by_variant_labels(model_data, variant_labels_in_both, model_dict):
    """
    Filters the model data to only include ensemble members with variant labels that are in both the model NAO data
    and the observed data.

    Parameters
    ----------
    model_data : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    variant_labels_in_both : list
        List of variant labels that are in both the model NAO data and the observed NAO data.
    model_dict : dict
        Dictionary containing the model names as keys and the variant labels as values.

    Returns
    -------
    psl_models_dict : dict
        Dictionary of filtered model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Loop over the members in the model_data
    for member in model_data:
        # Extract the variant label for the member
        variant_label = member.attrs["variant_label"]

        # Only if the variant label is in the variant_labels_in_both
        if variant_label in variant_labels_in_both:
            # Append the member to the model_dict
            model_dict.append(member)
        else:
            print("Variant label:", variant_label, "not in the variant_labels_in_both")

    return model_dict


# Function to constrain the years between the rescaled model nao and the matched variable
# For NAO, the variable will always be psl
def constrain_years_psl_match_var(
    model_nao_constrained,
    model_nao_years,
    psl_models,
    match_var_model_anomalies_constrained,
    match_var_model_years,
    match_var_models,
):
    """
    Ensures that the years are the same for both the matched variable and the NAO index (psl).

    Parameters
    ----------
    model_nao_constrained : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
        This is the constrained model NAO index.
    model_nao_years : numpy.ndarray
        Array of years for the model NAO index.
    psl_models : list
        List of models to be plotted for the NAO index. Different models for each variable.
    match_var_model_anomalies_constrained : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the matched variable.
        This is the constrained matched variable.
    match_var_model_years : numpy.ndarray
        Array of years for the matched variable.
    match_var_models : list
        List of models to be plotted for the matched variable. Different models for each variable.

        Returns
        -------
    model_nao_constrained : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
        This is the constrained model NAO index.
    match_var_model_anomalies_constrained : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the matched variable.
    """

    # First identify which years are in both the model_nao_constrained and the match_var_model_anomalies_constrained
    # find where model_nao_years and match_var_model_years are equal
    years_in_both = np.intersect1d(model_nao_years, match_var_model_years)
    print("Years in both:", years_in_both)

    # Initialize dictionaries to store the constrained model_nao and the constrained match_var_model_anomalies
    model_nao_constrained_dict = {}
    match_var_model_anomalies_constrained_dict = {}

    # Loop over the models in the model_nao_constrained
    for model in psl_models:
        # Extract the model data for the model
        model_nao_constrained_model = model_nao_constrained[model]

        # Loop over the members in the model_nao_constrained_model
        for member in model_nao_constrained_model:
            # Extract the years
            model_nao_constrained_years = member.time.dt.year.values

            # if the years are not equal
            if not np.array_equal(model_nao_constrained_years, years_in_both):
                # Print a warning and exit the program
                print(
                    "The years for the model_nao_constrained and the years_in_both are not equal"
                )
                print("Constraining the years")
                # Constrain the years
                member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the model_nao_constrained_dict
            if model not in model_nao_constrained_dict:
                model_nao_constrained_dict[model] = []
            # Append the member to the model_nao_constrained_dict
            model_nao_constrained_dict[model].append(member)

    # Loop over the models in the match_var_model_anomalies_constrained
    for model in match_var_models:
        # Extract the model data for the model
        match_var_model_anomalies_constrained_model = (
            match_var_model_anomalies_constrained[model]
        )

        # Loop over the members in the match_var_model_anomalies_constrained_model
        for member in match_var_model_anomalies_constrained_model:
            # Extract the years
            match_var_model_anomalies_constrained_years = member.time.dt.year.values

            # if the years are not equal
            if not np.array_equal(
                match_var_model_anomalies_constrained_years, years_in_both
            ):
                # Print a warning and exit the program
                print(
                    "The years for the match_var_model_anomalies_constrained and the years_in_both are not equal"
                )
                print("Constraining the years")
                # Constrain the years
                member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the match_var_model_anomalies_constrained_dict
            if model not in match_var_model_anomalies_constrained_dict:
                match_var_model_anomalies_constrained_dict[model] = []
            # Append the member to the match_var_model_anomalies_constrained_dict
            match_var_model_anomalies_constrained_dict[model].append(member)

    # Return the model_nao_constrained_dict and the match_var_model_anomalies_constrained_dict
    return (
        model_nao_constrained_dict,
        match_var_model_anomalies_constrained_dict,
        years_in_both,
    )


# Function to calculate the ensemble mean for the matched variable
def calculate_matched_var_ensemble_mean(matched_var_members, year):
    """
    Calculates the ensemble mean for the matched variable for a given year.

    Parameters
    ----------
    matched_var_members : list
        List of ensemble members for the matched variable.
        Each ensemble member is an xarray dataset containing the matched variable.
    year : int
        The year for which to calculate the ensemble mean.

    Returns
    -------
    matched_var_ensemble_mean : xarray.DataArray
        Ensemble mean for the matched variable for the specified year.
    """

    # Create an empty list to store the matched variable members
    matched_var_members_list = []

    # Loop over the ensemble members for the matched variable
    for i, member in enumerate(matched_var_members):

        # Chceck that the data is for the correct year
        if member.time.dt.year.values != year:
            print("member time", member.time.dt.year.values)
            print("year", year)
            # Print a warning and exit the program
            print("The data is not for the correct year")
            sys.exit()

        # Append the member to the list
        matched_var_members_list.append(member)

    # Concatenate the matched_var_members_list
    matched_var_members = xr.concat(
        matched_var_members_list, dim="member", coords="minimal", compat="override"
    )

    # for each of the members in the matched_var_members
    # group by the year and take the mean
    matched_var_members = matched_var_members.groupby("time.year").mean()

    # Calculate the ensemble mean for the matched variable
    matched_var_ensemble_mean = matched_var_members.mean(dim="member")

    # Convert the matched_var_ensemble_mean to an xarray DataArray
    coords = matched_var_members[0].coords
    dims = matched_var_members[0].dims
    matched_var_ensemble_mean = xr.DataArray(
        matched_var_ensemble_mean, coords=coords, dims=dims
    )

    return matched_var_ensemble_mean, matched_var_members


# Define a function which will extract the right model members for the matched variable


def extract_matched_var_members(
    year, match_var_model_anomalies_constrained, smallest_diff, lagged=True
):
    """
    Extracts the right model members for the matched variable.
    These members have the correct magnitude of the NAO index.
    """

    # Create an empty list to store the matched variable members
    matched_var_members = []

    # Extract the models from the smallest_diff
    smallest_diff_models = [member.attrs["source_id"] for member in smallest_diff]

    # Extract only the unique models
    smallest_diff_models = np.unique(smallest_diff_models)

    # print the smallest_diff_models
    print("smallest_diff_models", smallest_diff_models)

    # Create a dictionary to store the models and their members contained within the smallest_diff
    smallest_diff_models_dict = {}

    # Set up the model variant pairs as a set of tuples
    model_variant_pairs = set()

    # Loop over the members in the smallest_diff
    for member in smallest_diff:
        # Extract the model name
        model_name = member.attrs["source_id"]

        # Extract the associated variant label
        variant_label = member.attrs["variant_label"]

        # If laggd is True
        if lagged == True:
            # Extract the lag
            lag = member.attrs["lag"]

            # Set up the model variant list
            model_variant_pairs.add((model_name, variant_label, lag))
        else:
            # Append this pair to the dictionary
            model_variant_pairs.add((model_name, variant_label))

    print("model_variant_pairs", model_variant_pairs)

    # Loop over the members in the model_data
    for member in match_var_model_anomalies_constrained:
        # Check if the model and variant label pair is in the model_variant_pairs
        if lagged == True:
            if (
                member.attrs["source_id"],
                member.attrs["variant_label"],
                member.attrs["lag"],
            ) in model_variant_pairs:
                print(
                    "Appending member:",
                    member.attrs["variant_label"],
                    "from model:",
                    member.attrs["source_id"],
                    "with lag:",
                    member.attrs["lag"],
                )

                # Select the data for the year
                member = member.sel(time=f"{year}")

                # Append the member to the matched_var_members
                matched_var_members.append(member)
        else:
            if (
                member.attrs["source_id"],
                member.attrs["variant_label"],
            ) in model_variant_pairs:
                print(
                    "Appending member:",
                    member.attrs["variant_label"],
                    "from model:",
                    member.attrs["source_id"],
                )

                # Select the data for the year
                member = member.sel(time=f"{year}")

                # Append the member to the matched_var_members
                matched_var_members.append(member)
        # continue

    # return the matched_var_members
    return matched_var_members


# Calculate the members which have the closest NAO index to the rescaled NAO index


def calculate_closest_members(
    year,
    rescaled_model_nao,
    model_nao,
    models,
    season,
    forecast_range,
    output_dir,
    lagged=True,
    no_subset_members=20,
    nao_lagged_ensemble_members=None,
):
    """
    Calculates the ensemble members (within model_nao) which have the closest NAO index to the rescaled NAO index.

    Parameters
    ----------
    year : int
        The year for which to rescale the NAO indices.
    rescaled_model_nao : xarray.DataArray
        Rescaled NAO index.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    models : list
        List of models to be plotted. Different models for each variable.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    lagged : bool, optional
        Flag to indicate whether the ensemble is lagged or not. The default is False.
    no_subset_members : int, optional
        Number of ensemble members to subset. The default is 20.
    nao_lagged_ensemble_members : list, optional
        List of ensemble members for NAO. The default is None.

    Returns
    -------
    closest_nao_members : dict
        Dictionary containing the closest ensemble members for each model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Print the year which is being processed
    print(f"Calculating nearest members for year: {year}")

    # Extract the years for the rescaled NAO index and the model NAO index
    rescaled_model_nao_years = rescaled_model_nao.time.dt.year.values
    model_nao_years = model_nao[models[0]][0].time.dt.year.values

    # # If the two years arrays are not equal
    # if not np.array_equal(rescaled_model_nao_years, model_nao_years):
    #     # Print a warning and exit the program
    #     print("The years for the rescaled NAO index and the model NAO index are not equal")
    #     sys.exit()

    # Initialize a list to store the smallest difference between the rescaled NAO index and the model NAO index
    smallest_diff = []

    # Extract the data for the year for the rescaled NAO index
    rescaled_model_nao_year = rescaled_model_nao.sel(time=f"{year}")

    # If nao_lagged_ensemble_members is None
    if nao_lagged_ensemble_members == None:
        if lagged == True:
            # Form the list of ensemble members
            _, ensemble_members_count, ensemble_members_list = (
                form_ensemble_members_list(model_nao, models, lagged=True, lag=4)
            )
        else:
            # Form the list of ensemble members
            ensemble_members_list, ensemble_members_count, _ = (
                form_ensemble_members_list(model_nao, models, lagged=False)
            )
    else:
        print("Using nao_lagged_ensemble_members")
        # Set the ensemble_members_list to the nao_lagged_ensemble_members
        ensemble_members_list = nao_lagged_ensemble_members

    # Loop over the ensemble members
    for member in ensemble_members_list:
        # Extract the data for the year
        model_nao_year = member.sel(time=f"{year}")

        # Print the model and member name
        # print("Model:", member.attrs["source_id"])
        # print("Member:", member.attrs["variant_label"])

        # # print the values of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index", rescaled_model_nao_year.values)
        # print("model NAO index", model_nao_year.values)

        # # print the types of the values of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index type", type(rescaled_model_nao_year.values))
        # print("model NAO index type", type(model_nao_year.values))

        # # Print the dimensions of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index dimensions", rescaled_model_nao_year.dims)
        # print("model NAO index dimensions", model_nao_year.dims)

        # # Print the coordinates of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index coordinates", rescaled_model_nao_year.coords)
        # print("model NAO index coordinates", model_nao_year.coords)

        # Make sure that rescaled model nao and model nao have the same dimensions
        if rescaled_model_nao_year.dims != model_nao_year.dims:
            AssertionError(
                "The dimensions of rescaled model nao and model nao are not the same"
            )
            sys.exit()

        # # If the coordinates of the rescaled NAO index and the model NAO index are not the same
        # if rescaled_model_nao_year.coords != model_nao_year.coords:
        #     # Print a warning and exit the program
        #     print("The coordinates of the rescaled NAO index and the model NAO index are not the same")
        #     print("rescaled NAO index coordinates", rescaled_model_nao_year.coords)
        #     print("model NAO index coordinates", model_nao_year.coords)
        #     print("reshaping coordinates")
        #     # Extract the time coordinate from the rescaled NAO index
        #     rescaled_model_nao_year_time = rescaled_model_nao_year.time.values

        #     # Extract the time coordinate from the model NAO index
        #     model_nao_year_time = model_nao_year.time.values

        #     # Find the difference between the two time coordinates
        #     time_diff = (rescaled_model_nao_year_time - model_nao_year_time)

        #     # print the time difference
        #     print("time difference", time_diff)
        #     # and the type of the time difference
        #     print("time difference type", type(time_diff))

        #     # Now we want to extract the values of model_nao_year for the current time
        #     model_nao_index_value = model_nao_year.sel(time=model_nao_year_time)

        #     # And we want to assign this value to the same time as the rescaled model nao
        #     model_nao_index = model_nao_index.assign(time=model_nao_year_time + pd.Timedelta(days=time_diff), model_nao_index = model_nao_index_value)

        # # print the coordinates of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index coordinates", rescaled_model_nao_year.coords)
        # print("model NAO index coordinates", model_nao_year.coords)

        # Calculate the annual mean for the rescaled NAO index and the model NAO index
        rescaled_model_nao_year_ann_mean = rescaled_model_nao_year.groupby(
            "time.year"
        ).mean()
        model_nao_year_ann_mean = model_nao_year.groupby("time.year").mean()

        # # print the coordinates of the rescaled NAO index and the model NAO index
        # print("rescaled NAO index coordinates", rescaled_model_nao_year_ann_mean.coords)
        # print("model NAO index coordinates", model_nao_year_ann_mean.coords)

        # Calculate the difference between the rescaled NAO index and the model NAO index
        nao_diff = np.abs(rescaled_model_nao_year_ann_mean - model_nao_year_ann_mean)

        # # Print the difference
        # print("Difference:", nao_diff.values)

        # Assign the coordinates of the rescaled NAO index to the difference
        nao_diff = nao_diff.assign_coords(
            coords=rescaled_model_nao_year_ann_mean.coords
        )

        # Extract the attributes from the member
        member_attributes = member.attrs

        # Add the attributes to the diff
        nao_diff.attrs = member_attributes

        # Append the difference to the list
        smallest_diff.append(nao_diff)

    # Sort the list of differences
    smallest_diff.sort()

    # # Logging the smallest difference
    # for i, member in enumerate(smallest_diff):
    #     print("Smallest difference member full ensemble:", i+1)
    #     # print the model name and the member name
    #     print("Model:", member.attrs["source_id"])
    #     print("Member:", member.attrs["variant_label"])
    #     # Print the value of the difference
    #     print("Difference:", member.values)

    # Select only the first no_subset_members members
    smallest_diff = smallest_diff[:no_subset_members]

    # Print the values of the smallest_diff
    print("Smallest difference values:", [member.values for member in smallest_diff])

    # Loop over the members with the smallest differences
    # for i, member in enumerate(smallest_diff):
    #     print("Smallest difference member:", i+1)
    #     # print the model name and the member name
    #     print("Model:", member.attrs["source_id"])
    #     print("Member:", member.attrs["variant_label"])
    #     # Print the value of the difference
    #     print("Difference:", member.values)

    return smallest_diff


# Define a new function to form the list of ensemble members


def form_ensemble_members_list(model_nao, models, lagged=False, lag=None):
    """
    Forms a list of ensemble members, not a dictionary with model keys.
    Each xarray object should have the associated metadata stored in attributes.

    Parameters
    ----------
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    models : list
        List of models to be plotted. Different models for each variable.
    lagging : bool, optional
        Flag to indicate whether to lag the ensemble members. The default is False.
    lag : int, optional
        The value of the lag. The default is None.

    Returns
    -------
    ensemble_members_list : list
        List of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Initialize a list to store the ensemble members
    ensemble_members_list = []

    # Form the list to store the lagged ensemble members
    lagged_ensemble_members_list = []

    # Initialize a dictionary to store the number of ensemble members for each model
    ensemble_members_count = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_nao_by_model = model_nao[model]

        # If the model is not in the ensemble_members_count dictionary
        if model not in ensemble_members_count:
            # Add the model to the ensemble_members_count dictionary
            ensemble_members_count[model] = 0

        # Loop over the ensemble members
        for member in model_nao_by_model:

            # if the type of time is not a datetime64
            if type(member.time.values[0]) != np.datetime64:
                # Extract the time values as a datetime64
                member_time = member.time.astype("datetime64[ns]")

                # Add the time values back to the member
                member = member.assign_coords(time=member_time)

            # If the years are not unique
            years = member.time.dt.year.values

            # Check that the years are unique
            if len(years) != len(set(years)):
                raise ValueError("Duplicate years in the member")

            # Check that the difference between the years is 1
            if not np.all(np.diff(years) == 1):
                print(
                    "The years are not consecutive for model:",
                    model,
                    "member:",
                    member.attrs["variant_label"],
                )
                continue

            # if the lagging flag is set to True
            if lagged == True:
                print("Lagging the ensemble members")
                # if lag is None, raise an error
                if lag is None:
                    raise ValueError("Trying to perform lagging, but the lag is None")

                # Loop over the lag indices
                for i in range(lag):
                    print(
                        "Applying lagging for ensemble member:",
                        member.attrs["variant_label"],
                        "for model:",
                        model,
                    )
                    print("Lag:", i)
                    # Shift the time series forward by the lag
                    shifted_member = member.shift(time=i)

                    # Assign a new attribute to the member
                    shifted_member.attrs["lag"] = i

                    # Append the member to the ensemble_members_list
                    lagged_ensemble_members_list.append(shifted_member)

            # Add the member to the ensemble_members_list
            ensemble_members_list.append(member)

            # Add one to the ensemble_members_count dictionary
            ensemble_members_count[model] += 1

    return ensemble_members_list, ensemble_members_count, lagged_ensemble_members_list


# Write a function to calculate the NAO index
# For both the obs and model data


def calculate_nao_index_and_plot(
    obs_anomaly,
    model_anomaly,
    models,
    variable,
    season,
    forecast_range,
    output_dir,
    plot_graphics=False,
    azores_grid=dic.azores_grid,
    iceland_grid=dic.iceland_grid,
    snao_south_grid=dic.snao_south_grid,
    snao_north_grid=dic.snao_north_grid,
    lag=None,
):
    """
    Calculates the NAO index for both the obs and model data.
    Then plots the NAO index for both the obs and model data if the plot_graphics flag is set to True.

    Parameters
    ----------
    obs_anomaly : xarray.Dataset
        Observations.
    model_anomaly : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets.
    models : list
        List of models to be plotted. Different models for each variable.
    variable : str
        Variable name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    plot_graphics : bool, optional
        Flag to plot the NAO index. The default is False.
    azores_grid : str, optional
        Azores grid. The default is dic.azores_grid.
    iceland_grid : str, optional
        Iceland grid. The default is dic.iceland_grid.
    snao_south_grid : str, optional
        SNAO south grid. The default is dic.snao_south_grid.
    snao_north_grid : str, optional
        SNAO north grid. The default is dic.snao_north_grid.
    lag : int, optional
        Lag. The default is None.

    Returns
    -------
    obs_nao: xarray.Dataset
        Observations. NAO index.
    model_nao: dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # If the variable is not psl, then exit
    if variable != "psl":
        AssertionError("The variable is not psl")
        sys.exit()

    # if the season is JJA, use the summer definition of the NAO
    if season == "JJA":
        print("Calculating NAO index using summer definition")
        # Set up the dict for the southern box and northern box
        south_grid, north_grid = snao_south_grid, snao_north_grid
        # Set up the NAO type for the summer definition
        nao_type = "snao"
    else:
        print("Calculating NAO index using standard definition")
        # Set up the dict for the southern box and northern box
        south_grid, north_grid = azores_grid, iceland_grid
        # Set up the NAO type for the standard definition
        nao_type = "default"

    # Calculate the NAO index for the observations
    obs_nao = calculate_obs_nao(obs_anomaly, south_grid, north_grid)

    # Calculate the NAO index for the model data
    model_nao, years, ensemble_members_count_nao = calculate_model_nao_anoms_matching(
        model_anomaly,
        models,
        azores_grid,
        iceland_grid,
        snao_south_grid,
        snao_north_grid,
        nao_type=nao_type,
    )

    # If the plot_graphics flag is set to True
    if plot_graphics:
        # First calculate the ensemble mean NAO index
        ensemble_mean_nao, _ = calculate_ensemble_mean(model_nao, models)

        # Calculate the correlation coefficients between the observed and model data
        r, p, _, _, _, _ = calculate_nao_correlations(
            obs_nao, ensemble_mean_nao, variable
        )

        # Plot the NAO index
        plot_nao_index(
            obs_nao,
            ensemble_mean_nao,
            variable,
            season,
            forecast_range,
            r,
            p,
            output_dir,
            ensemble_members_count_nao,
            nao_type=nao_type,
        )

    return obs_nao, model_nao


# Define a function to calculate and plot the spna index
def calculate_spna_index_and_plot(
    obs_anom,
    model_anom,
    models,
    variable,
    season,
    forecast_range,
    output_dir,
    plot_graphics=False,
    spna_grid=dic.spna_grid_strommen,
):
    """
    Calculates and (optionally) plots the SPNA index as in Strommen et al.
    2023.

    Inputs:
    -------
    obs_anom : xarray.Dataset
        Observations of temperature anomalies.
    model_anom: dict[xarray.Dataset]
        Dictionary of xarray datasets for each model.
    models:
        List of the models to be plotted. Different models for eahc variable.
    variable : str
        Variable name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    plot_graphics : bool, optional
        Flag to plot the NAO index. The default is False.
    spna_grid : dict, optional
        Dictionary containing the longitude and latitude values of the SPNA gridbox.
        The default is dic.spna_grid_strommen.

    Returns
    -------
    obs_spna : xarray.Dataset
        Observations of the SPNA index.
    model_spna : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members,
        which are xarray datasets containing the SPNA index.
    """

    # Assert that the variable is tas
    assert (
        variable == "tas"
    ), "The variable is not tas. SPNA index is only defined for tas."

    # calculate the SPNA index for the observations
    obs_spna = calculate_spna_index(t_anom=obs_anom, gridbox=spna_grid)

    # Initialize a dictionary to store the model SPNA index
    model_spna = {}

    # calculate the SPNA index for the model data
    for model in models:
        print("Calculating SPNA index for model:", model)

        # Extract the model data
        model_anom_by_model = model_anom[model]

        # TODO: Finish off the SPNA calculation function here
        # Loop over the members for this model
        for member in model_anom_by_model:

            # Extract the attributes for this member
            member_attributes = member.attrs

            # Calculate the SPNA index for this member
            model_spna_member = calculate_spna_index(t_anom=member, gridbox=spna_grid)

            # Add the member to the model_spna dictionary
            if model not in model_spna:
                model_spna[model] = []

            # Associate the attributes with the model_spna_member
            model_spna_member.attrs = member_attributes

            # Append the member to the model_spna dictionary
            model_spna[model].append(model_spna_member)

    # If the plot_graphics flag is set to True
    if plot_graphics:
        print("Plotting SPNA index")

        # First calculate the ensemble mean SPNA index
        ensemble_mean_spna, ensemble_members_spna = calculate_ensemble_mean(
            model_var=model_spna, models=models, lag=None
        )

        # Assert that ensemble_mean_spna has one dimension
        assert (
            len(ensemble_mean_spna.dims) == 1
        ), "ensemble_mean_spna does not have one dimension."

        # Assert that ensemble members spna array has two dimensions
        assert (
            len(ensemble_members_spna.shape) == 2
        ), "ensemble_members_spna does not have two dimensions."

        # Set up the figure
        fig = plt.figure(figsize=(10, 6))

        # Extract the years
        obs_years = obs_spna.time.dt.year.values
        model_years = ensemble_mean_spna.time.dt.year.values

        # Assert that the obs_years and model_years are the same
        # using np.array_equal
        assert np.array_equal(
            obs_years, model_years
        ), "The obs_years and model_years are not the same."

        # FIXME: Hardcoded for years 2-9 forecast range
        # Plot the obs and the model data
        plt.plot(obs_years - 5, -obs_spna, label="ERA5", color="black")

        # Plot the ensemble mean
        plt.plot(model_years - 5, -ensemble_mean_spna, label="dcppA", color="red")

        # Add a horizontal line at y=0
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        # Set the ylabel
        plt.ylabel("SPNA index (K)")
        plt.xlabel("initialisation year")

        # Set up a textbox with the season name in the top left corner
        plt.text(
            0.05,
            0.95,
            season,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Calculate the correlation coefficients
        corr1, p1 = pearsonr(obs_spna, ensemble_mean_spna)

        # Convert ensemble_members_spna to a numpy array
        ensemble_mean_spna = ensemble_mean_spna.values

        # Calculate the RPC for the SPNA index
        rpc1 = corr1 / (np.std(ensemble_mean_spna) / np.std(ensemble_members_spna))

        # Extract the start and finish initialisation years
        # FIXME: hardcoded for years 2-9 forecast range
        start_year = obs_years[0] - 5
        finish_year = obs_years[-1] - 5

        # Extract the number of ensemble members
        no_ensemble_members = ensemble_members_spna.shape[0]

        # Set up the title for the plot
        plt.title(
            f"ACC = {corr1:.2f}, (p = {p1:.2f})"
            f"\nRPC = {rpc1:.2f}, n = {no_ensemble_members},"
            f"\nyears_{start_year}_{finish_year}, {season},"
            f" {forecast_range}, dcppA-hindcast",
            fontsize=10,
        )

        # Set up the figure name
        fig_name = (
            f"{variable}_{forecast_range}_{season}_"
            f"dcppA-hindcast_SPNA_index_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        # Save the figure
        plt.savefig(output_dir + "/" + fig_name, dpi=300, bbox_inches="tight")

        # Show the figure
        plt.show()

    return obs_spna, model_spna


# Define a function for plotting the NAO index
def plot_nao_index(
    obs_nao,
    ensemble_mean_nao,
    variable,
    season,
    forecast_range,
    r,
    p,
    output_dir,
    ensemble_members_count,
    experiment="dcppA-hindcast",
    nao_type="default",
):
    """
    Plots the NAO index for both the observations and model data.

    Parameters
    ----------
    obs_nao : xarray.Dataset
        Observations.
    ensemble_mean_nao : xarray.Dataset
        Ensemble mean of the model data.
    variable : str
        Variable name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    r : float
        Correlation coefficients between the observed and model data.
    p : float
        p-values for the correlation coefficients between the observed and model data.
    output_dir : str
        Path to the output directory.
    ensemble_members_count : dict
        Number of ensemble members for each model.
    experiment : str, optional
        Experiment name. The default is "dcppA-hindcast".
    nao_type : str, optional
        NAO type. The default is "default".


    Returns
    -------
    None.

    """

    # Set the font size
    plt.rcParams.update({"font.size": 12})

    # Set up the figure
    fig = plt.figure(figsize=(10, 6))

    # Set up the title
    plot_name = (
        f"{variable} {forecast_range} {season} {experiment} {nao_type} NAO index"
    )

    # Process the obs and the model data
    # from Pa to hPa
    obs_nao = obs_nao / 100
    ensemble_mean_nao = ensemble_mean_nao / 100

    # Extract the years
    obs_years = obs_nao.time.dt.year.values
    model_years = ensemble_mean_nao.time.dt.year.values

    # If the obs years and model years are not the same
    if len(obs_years) != len(model_years):
        raise ValueError("Observed years and model years must be the same.")

    # Plot the obs and the model data
    plt.plot(obs_years, obs_nao, label="ERA5", color="black")

    # Plot the ensemble mean
    plt.plot(model_years, ensemble_mean_nao, label="dcppA", color="red")

    # Add a horizontal line at y=0
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    # Set the ylim
    plt.ylim(-10, 10)
    plt.ylabel("NAO index (hPa)")
    plt.xlabel("year")

    # Set up a textbox with the season name in the top left corner
    plt.text(
        0.05,
        0.95,
        season,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # If the nao_type is not default
    # then add a textbox with the nao_type in the top right corner
    if nao_type != "default":
        # nao type = summer nao
        # add a textbox with the nao_type in the top right corner
        plt.text(
            0.95,
            0.95,
            nao_type,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Set up the p value text box
    if p < 0.01:
        p_text = "< 0.01"
    elif p < 0.05:
        p_text = "< 0.05"
    else:
        p_text = f"= {p:.2f}"

    # Extract the ensemble members count
    if ensemble_members_count is not None:
        no_ensemble_members = sum(ensemble_members_count.values())
    else:
        no_ensemble_members = None

    # Set up the title for the plot
    plt.title(
        f"ACC = {r:.2f}, p {p_text}, n = {no_ensemble_members}, years_{forecast_range}, {season}, {experiment}",
        fontsize=10,
    )

    # Set up the figure name
    fig_name = f"{variable}_{forecast_range}_{season}_{experiment}_{nao_type}_NAO_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Save the figure
    plt.savefig(output_dir + "/" + fig_name, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Calculate obs nao

# Define a function to calculate the ensemble mean NAO index


def calculate_ensemble_mean(model_var, models, lag=None):
    """
    Calculates the ensemble mean NAO index for the given model data.

    Parameters
    ----------
    model_nao (dict): The model data containing the NAO index for each ensemble member.
    models (list): The list of models to be plotted.

    Returns
    -------
    ensemble_mean_nao (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    """

    # Initialize a list for the ensemble members
    ensemble_members_var = []

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_var[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members_var.append(member)

    coords = member.coords
    dims = member.dims

    # Convert the list of ensemble members to a numpy array
    ensemble_members_var = np.array(ensemble_members_var)

    # if lag is not None
    if lag is not None:
        # Lage the ensemble members
        ensemble_members_var, years_to_keep = lag_ensemble(
            ensemble_members_var, lag, NAO_index=True
        )

        # Remove the first lag - 1 years from the member
        years = member.time.dt.year.values

        # remove the first lag - 1 years from the years
        years_constrained = years[lag - 1 :]

        # Extract the constrained years from the ensemble members
        member = member.sel(time=member.time.dt.year.isin(years_constrained))

        # extract the coords from the member
        coords = member.coords
        dims = member.dims

    # Calculate the ensemble mean NAO index
    ensemble_mean_var = np.mean(ensemble_members_var, axis=0)

    # print the dimensions
    print("dimensions of the lagged:", lag, "ensemble: ", dims)
    print("coordinates of the lagged: ", lag, "ensemble: ", coords)

    # Convert the ensemble mean NAO index to an xarray DataArray
    ensemble_mean_var = xr.DataArray(
        ensemble_mean_var, coords=member.coords, dims=member.dims
    )

    return ensemble_mean_var, ensemble_members_var


# Write a function to rescale the NAO index
# We will only consider the non-lagged ensemble index for now


def rescale_nao(
    obs_nao, model_nao, models, season, forecast_range, output_dir, lag=None
):
    """
    Rescales the NAO index according to Doug Smith's (2020) method.

    Parameters
    ----------
    obs_nao : xarray.Dataset
        Observations.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets.
    models : list
        List of models to be plotted. Different models for each variable.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    lag : int, optional
        Lag. The default is None.

    Returns
    -------
    rescaled_model_nao : numpy.ndarray
        Array contains the rescaled NAO index.
    ensemble_mean_nao : numpy.ndarray
        Ensemble mean NAO index. Not rescaled
    ensemble_members_nao : numpy.ndarray
        Ensemble members NAO index. Not rescaled
    """

    if lag is None:
        # First calculate the ensemble mean NAO index
        ensemble_mean_nao, ensemble_members_nao = calculate_ensemble_mean(
            model_nao, models
        )
    else:
        # Calculate the lagged ensemble mean NAO index
        ensemble_mean_nao, ensemble_members_nao = calculate_ensemble_mean(
            model_nao, models, lag
        )

        # Remove the first lag - 1 years from the obs_nao
        years = obs_nao.time.dt.year.values
        years_constrained = years[lag - 1 :]

        # Extract the constrained years from the obs_nao
        obs_nao = obs_nao.sel(time=obs_nao.time.dt.year.isin(years_constrained))

        obs_years = years_constrained

    # Extract the years from the ensemble members
    model_years = ensemble_mean_nao.time.dt.year.values
    # Extract the years from the obs
    obs_years = obs_nao.time.dt.year.values

    # If the two years arrays are not equal
    if not np.array_equal(model_years, obs_years):
        # Print a warning and exit the program
        print("The years for the ensemble members and the observations are not equal")
        sys.exit()

    # if the type of obs_nao is not a numpy array
    # Then convert to a numpy array
    if type(obs_nao) != np.ndarray:
        print("Converting obs_nao to a numpy array")
        obs_nao = obs_nao.values

    # Create an empty numpy array to store the rescaled NAO index
    rescaled_model_nao = np.empty((len(model_years)))

    # dimensions of memmbers
    print("dimensions of ensemble members nao: ", ensemble_members_nao.shape)
    print("coords of ensemble members nao: ", ensemble_members_nao.shape)
    print("shape of ensemble mean: ", ensemble_mean_nao.shape)

    # Loop over the years and perform the rescaling (including cross-validation)
    for i, year in enumerate(model_years):

        # Compute the rescaled NAO index for this year
        signal_adjusted_nao_index_year, _ = rescale_nao_by_year(
            year,
            obs_nao,
            ensemble_mean_nao,
            ensemble_members_nao,
            season,
            forecast_range,
            output_dir,
            lagged=False,
            omit_no_either_side=1,
        )

        # Append the rescaled NAO index to the list, along with the year
        rescaled_model_nao[i] = signal_adjusted_nao_index_year

    # Print the rescaled model nao
    print("rescaled model nao before xarray", rescaled_model_nao)

    # Convert the list to an xarray DataArray
    # With the same coordinates as the ensemble mean NAO index
    rescaled_model_nao = xr.DataArray(
        rescaled_model_nao, coords=ensemble_mean_nao.coords, dims=ensemble_mean_nao.dims
    )

    print("rescaled model nao after xarray", rescaled_model_nao.values)

    # If the time type is not datetime64 for the rescaled model nao
    # Then convert the time type to datetime64
    if type(rescaled_model_nao.time.values[0]) != np.datetime64:
        rescaled_model_nao_time = rescaled_model_nao.time.astype("datetime64[ns]")

        # Modify the time coordinate using the assign_coords() method
        rescaled_model_nao = rescaled_model_nao.assign_coords(
            time=rescaled_model_nao_time
        )

    # Return the rescaled model NAO index
    return rescaled_model_nao, ensemble_mean_nao, ensemble_members_nao, obs_years


# Define a new function to rescalse the NAO index for each year
def rescale_nao_by_year_mod(
    year,
    obs_nao,
    ensemble_mean_nao,
    ensemble_members_nao,
    season,
    forecast_range,
    output_dir,
    lagged=False,
    omit_no_either_side=1,
    lag=None,
):
    """
    Rescales the observed and model NAO indices for a given year and season, and saves the results to disk.

    Parameters
    ----------
    year : int
        The year for which to rescale the NAO indices.
    obs_nao : pandas.DataFrame
        A DataFrame containing the observed NAO index values, with a DatetimeIndex.
    ensemble_mean_nao : pandas.DataFrame
        A DataFrame containing the ensemble mean NAO index values, with a DatetimeIndex.
    ensemble_members_nao : dict
        A dictionary containing the NAO index values for each ensemble member, with a DatetimeIndex.
    season : str
        The season for which to rescale the NAO indices. Must be one of 'DJF', 'MAM', 'JJA', or 'SON'.
    forecast_range : int
        The number of months to forecast ahead.
    output_dir : str
        The directory where to save the rescaled NAO indices.
    lagged : bool, optional
        Whether to use lagged NAO indices in the rescaling. Default is False.

    Returns
    -------
    None
    """

    # Print the year for which the NAO indices are being rescaled
    print(f"Rescaling NAO indices for {year}")

    # Extract the model years
    model_years = ensemble_mean_nao.time.dt.year.values

    # Ensure that the type of ensemble_mean_nao and ensemble_members_nao is a an array
    if (
        type(ensemble_mean_nao)
        and type(ensemble_members_nao) != np.ndarray
        and type(obs_nao) != np.ndarray
    ):
        AssertionError(
            "The type of ensemble_mean_nao and ensemble_members_nao and obs_nao is not a numpy array"
        )
        sys.exit()

    # If the year is not in the ensemble members years
    if year not in model_years:
        # Print a warning and exit the program
        print(f"Year {year} is not in the ensemble members years")
        sys.exit()

    # If lag is not none
    if lag is not None:
        # print that we are removing the years containing nans from the arrays
        print("Removing the years containing nans from the arrays")
        # extract the size of the years axis
        years_size = ensemble_members_nao.shape[1]

        # Loop over the years
        for i in range(years_size):
            # extract the values for the year
            current_year = ensemble_members_nao[:, i]

            # If all of the values are nans
            if np.isnan(current_year).all():
                # Remove the year from the ensemble members
                ensemble_members_nao = np.delete(ensemble_members_nao, i, axis=1)
                # Remove the year from the model years
                model_years = np.delete(model_years, i)

    # Extract the index for the year
    year_index = np.where(model_years == year)[0]

    # Extract the ensemble members for the year
    ensemble_members_nao_year = ensemble_members_nao[:, year_index]

    # Compute the ensemble mean NAO for this year
    ensemble_mean_nao_year = ensemble_members_nao_year.mean(axis=0)

    # Set up the indicies for the cross-validation
    # In the case of the first year
    if year == model_years[0]:
        print("Cross-validation case for the first year")
        print("Removing the first year and:", omit_no_either_side, "years forward")
        # Set up the indices to use for the cross-validation
        # Remove the first year and omit_no_either_side years forward
        cross_validation_indices = np.arange(0, omit_no_either_side + 1)
    # In the case of the last year
    elif year == model_years[-1]:
        print("Cross-validation case for the last year")
        print("Removing the last year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Remove the last year and omit_no_either_side years backward
        cross_validation_indices = np.arange(-1, -omit_no_either_side - 2, -1)
    # In the case of any other year
    else:
        # Omit the year and omit_no_either_side years forward and backward
        print("Cross-validation case for any other year")
        print("Removing the year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Use the year index and omit_no_either_side years forward and backward
        cross_validation_indices = np.arange(
            year_index - omit_no_either_side, year_index + omit_no_either_side + 1
        )

    # Log which years are being used for the cross-validation
    print("Cross-validation indices:", cross_validation_indices)

    # Extract the ensemble members for the cross-validation
    # i.e. don't use the years given by the cross_validation_indices
    ensemble_members_nao_array_cross_val = np.delete(
        ensemble_members_nao, cross_validation_indices, axis=1
    )
    # Take the mean over the ensemble members
    # to get the ensemble mean nao for the cross-validation
    ensemble_mean_nao_cross_val = ensemble_members_nao_array_cross_val.mean(axis=0)

    # Remove the indicies from the obs_nao
    obs_nao_cross_val = np.delete(obs_nao, cross_validation_indices, axis=0)

    # Calculate the pearson correlation coefficient between the observed and model NAO indices
    acc_score, p_value = stats.pearsonr(obs_nao_cross_val, ensemble_mean_nao_cross_val)

    # Calculate the RPS score
    rps_score = calculate_rps(
        acc_score, ensemble_members_nao_array_cross_val, obs_nao_cross_val
    )

    # Compute the rescaled NAO index for the year
    signal_adjusted_nao_index = ensemble_mean_nao_year * rps_score

    return signal_adjusted_nao_index, ensemble_mean_nao_year


def calculate_rpc(acc_score, ensemble_members_array):
    """
    Calculates the RPC score. Ratio of predictable components.

    Parameters
    ----------
    acc_score : float
        The ACC score.
    ensemble_members_array : numpy.ndarray
        The ensemble members array.

    Returns
    -------
    rpc_score : float
        The RPC score.
    """

    # Calculate the ensemble mean over all members
    ensemble_mean = np.mean(ensemble_members_array, axis=0)

    # Calculate the standard deviation of the predictable signal for the forecasts (σfsig)
    sigma_fsig = np.std(ensemble_mean)

    # Calculate the total standard deviation of the forecasts (σftot)
    sigma_ftot = np.std(ensemble_members_array)

    # Calculate the RPC score
    rpc_score = acc_score / (sigma_fsig / sigma_ftot)

    return rpc_score


# Calculate the RPS score - ratio of predictable signals


def calculate_rps(acc_score, ensemble_members_array, obs_nao):
    """
    Calculates the RPS score. Ratio of predictable signals.

    Parameters
    ----------
    acc_score : float
        The ACC score.
    ensemble_members_array : numpy.ndarray
        The ensemble members array.
    obs_nao : numpy.ndarray
        The observed NAO index.

    Returns
    -------
    rps_score : float
        The RPS score.
    """

    # Calculate the ratio of predictable components (for the model)
    rpc = calculate_rpc(acc_score, ensemble_members_array)

    # Calculate the total standard deviation of the observations (σotot)
    obs_std = np.std(obs_nao)

    # Calculate the total standard deviation of the forecasts (σftot)
    model_std = np.std(ensemble_members_array)

    # Calculate the RPS score
    rps_score = rpc * (obs_std / model_std)

    return rps_score


# Function to ensure that the years contrained are consistent across the models
#


# Define a function which processes the model data for spatial correlations
def process_model_data_for_plot(model_data, models, lag=None):
    """
    Processes the model data and calculates the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    lag (int): The lag to be plotted. Default is None.

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
        # print("extracting data for model:", model)

        # Set the ensemble members count to zero
        # if the model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:

            # # Modify the time dimension
            # if type is not already datetime64
            # then convert the time type to datetime64
            if type(member.time.values[0]) != np.datetime64:
                member_time = member.time.astype("datetime64[ns]")

                # # Modify the time coordinate using the assign_coords() method
                member = member.assign_coords(time=member_time)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            years = member.time.dt.year.values

            # If the years index has duplicate values
            # Then we will skip over this ensemble member
            # and not append it to the list of ensemble members
            if len(years) != len(set(years)):
                print("Duplicate years in ensemble member")
                continue

            # If the difference between the years is not 1
            # Then we will skip over this ensemble member
            # and not append it to the list of ensemble members
            if np.diff(years).all() != 1:
                print(
                    "Non-consecutive years in ensemble member, model:",
                    model,
                    "member:",
                    member,
                )
                continue

            # Print the type of the calendar
            # print(model, "calendar type:", member.time)
            # print("calendar type:", type(member.time))

            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # member_id = member.attrs['variant_label']

            # Try to #print values for each member
            # #print("trying to #print values for each member for debugging")
            # #print("values for model:", model)
            # #print("values for members:", member)
            # #print("member values:", member.values)

            # #print statements for debugging
            # #print('shape of years', np.shape(years))
            # # #print('years', years)
            # print("len years for model", model, "and member", member, ":", len(years))

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    coords = member.coords
    dims = member.dims

    # If the lag is not None
    # then lag the ensemble members
    if lag is not None:
        # Set up a list for the lagged ensemble members
        lagged_ensemble_members = []
        # If lag is not an int, raise a value error
        if type(lag) != int:
            raise ValueError("The lag must be an integer")
        # Loop over the ensemble members
        for member in ensemble_members:
            # Loop over the lag
            for lag_index in range(lag):
                print(
                    "Shifting model:",
                    member.attrs["source_id"],
                    "member:",
                    member.attrs["variant_label"],
                    "and applying lag:",
                    lag_index,
                )
                # Shift the time series for each member forward by the lag index
                shifted_member = member.shift(time=lag_index)

                # Assign a new attribute to the shifted member
                shifted_member.attrs["lag"] = lag_index

                # Append the shifted member to the list of lagged ensemble members
                lagged_ensemble_members.append(shifted_member)
        # Set up the constrained years
        years_constrained = years[lag - 1 :]

        lagged_ensemble_members_constrained = []

        # Remove the first lag - 1 years from each member
        for member in lagged_ensemble_members:
            # Extract the years
            years = member.time.dt.year.values

            # Extract the constrained years from the member
            member = member.sel(time=member.time.dt.year.isin(years_constrained))

            # Extract the coords from the member
            coords = member.coords
            dims = member.dims

            # Append the member to the list of lagged ensemble members
            lagged_ensemble_members_constrained.append(member)

        # Set the ensemble members to the lagged ensemble members
        ensemble_members = lagged_ensemble_members_constrained
    else:
        years_constrained = years

    # #print the dimensions of the ensemble members
    # #print("ensemble members shape", np.shape(ensemble_members))

    # Convert the list of ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # #print the ensemble members count
    print("ensemble members count", ensemble_members_count)

    # Print the type of ensemble members
    print("type of ensemble members", type(ensemble_members))
    print("shape of ensemble members", np.shape(ensemble_members))

    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # #print the dimensions of the ensemble mean
    print(np.shape(ensemble_mean))
    # #print(type(ensemble_mean))
    print(ensemble_mean)

    # print the dims
    print("dims", dims)
    print("coords", coords)

    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(ensemble_mean, coords=coords, dims=dims)

    return (
        ensemble_mean,
        lat,
        lon,
        years,
        ensemble_members_count,
        years_constrained,
        ensemble_members,
    )


# Create a function for lagging the ensemble
def lag_ensemble(ensemble_members, lag, NAO_index=False):
    """
    Lag the ensemble members array by combining each year with the previous lag-1 years.

    Parameters:
    ensemble_members (numpy.ndarray): The ensemble members to be lagged.
    lag (int): The lag to be applied.
    NAO_index (bool): Flag to indicate whether the NAO index is being lagged. Default is False.

    Returns:
    lagged_ensemble (numpy.ndarray): The lagged ensemble members.
    """

    # if the type of the ensemble members is not a numpy array
    # print an error message and exit
    if type(ensemble_members) != np.ndarray:
        raise ValueError("The ensemble members must be a numpy array")

    # Extract the number of ensemble members
    m_ensemble_members = ensemble_members.shape[0]

    # Extract the number of years
    n_years = ensemble_members.shape[1]

    # if the nao_index is not true
    if not NAO_index:
        # Extract the shape of the lat and lon dimensions
        lat_shape = ensemble_members.shape[2]
        lon_shape = ensemble_members.shape[3]

    # Print the number of ensemble members and years
    print("Number of ensemble members:", m_ensemble_members)
    print("Number of years:", n_years)

    # Set up the no_lagged_members
    m_lagged_ensemble_members = m_ensemble_members * lag

    if not NAO_index:
        # Set up an empty array for the lagged ensemble members
        lagged_ensemble = np.empty(
            (m_lagged_ensemble_members, n_years, lat_shape, lon_shape)
        )
    else:
        # Set up an empty array for the lagged ensemble members
        lagged_ensemble = np.empty((m_lagged_ensemble_members, n_years))

    # Loop over the ensemble members
    for member in range(m_ensemble_members):
        # Loop over each year
        for year in range(n_years):
            # If the year index is less than lag - 1
            # Then set the lagged ensemble member to NaN
            if year < lag - 1:
                if not NAO_index:
                    lagged_ensemble[member, year, :, :] = np.nan
                else:
                    lagged_ensemble[member, year] = np.nan
                # Loop over the lag
                for lag_index in range(lag):
                    # Set the lag_index members with the year <= lag - 1 to NaN
                    if not NAO_index:
                        lagged_ensemble[member * lag + lag_index, year, :, :] = np.nan
                    else:
                        lagged_ensemble[member * lag + lag_index, year] = np.nan
            # if the year index is greater than or equal to lag - 1
            else:
                # Loop over the lag
                for lag_index in range(lag):
                    # Set the lagged ensemble member
                    # to the ensemble member
                    # for the current year minus the lag index
                    if not NAO_index:
                        lagged_ensemble[member * lag + lag_index, year, :, :] = (
                            ensemble_members[member, year - lag_index, :, :]
                        )
                    else:
                        lagged_ensemble[member * lag + lag_index, year] = (
                            ensemble_members[member, year - lag_index]
                        )

    # Remove the years which only contain NaN values
    years_to_keep = []
    # Loop over the years
    for year in range(n_years):
        # If the year only contains NaN values
        if not NAO_index:
            if not np.isnan(lagged_ensemble[:, year, :, :]).all():
                years_to_keep.append(year)
        else:
            if not np.isnan(lagged_ensemble[:, year]).all():
                years_to_keep.append(year)
    # Create a new array that only contains the non-NaN years
    lagged_ensemble_constrained = (
        lagged_ensemble[:, years_to_keep, :, :]
        if not NAO_index
        else lagged_ensemble[:, years_to_keep]
    )

    # Set the index of the years to keep
    years_to_keep = np.array(years_to_keep)

    # Return the lagged ensemble
    return lagged_ensemble_constrained, years_to_keep


# Define a new function
# process_model_data_for_plot_timeseries
# which processes the model data for timeseries
def process_model_data_for_plot_timeseries(model_data, models, region):
    """
    Processes the model data and calculates the ensemble mean as a timeseries.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    region (str): The region to be plotted.

    Returns:
    ensemble_mean (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    years (numpy.ndarray): The years.
    ensemble_members_count (dict): The number of ensemble members for each model.
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

        # Set the ensemble members count to zero
        # if model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:

            # Modify the time dimension
            # if type is not already datetime64
            # then convert the time type to datetime64
            if type(member.time.values[0]) != np.datetime64:
                # Extract the time values as datetime64
                member_time = member.time.astype("datetime64[ns]")

                # Modify the time coordinate using the assign_coords() method
                member = member.assign_coords(time=member_time)

            # Set up the region
            if region == "north-sea":
                print("North Sea region gridbox mean")
                gridbox_dict = dic.north_sea_grid
            elif region == "central-europe":
                print("Central Europe region gridbox mean")
                gridbox_dict = dic.central_europe_grid
            else:
                print("Invalid region")
                sys.exit()

            # Extract the lat and lon values
            # from the gridbox dictionary
            lon1, lon2 = gridbox_dict["lon1"], gridbox_dict["lon2"]
            lat1, lat2 = gridbox_dict["lat1"], gridbox_dict["lat2"]

            # Take the mean over the lat and lon values
            # to get the mean over the region
            # for the ensemble member
            try:
                member_gridbox_mean = member.sel(
                    lat=slice(lat1, lat2), lon=slice(lon1, lon2)
                ).mean(dim=["lat", "lon"])
            except Exception as e:
                print(f"Error taking gridbox mean: {e}")
                sys.exit()

            # Extract the years
            years = member_gridbox_mean.time.dt.year.values

            # If the years index has duplicate values
            # Then we will skip over this ensemble member
            # and not append it to the list of ensemble members
            if len(years) != len(set(years)):
                print("Duplicate years in ensemble member")
                continue

            # Print the years for debugging
            print("len years for model", model, "and member", member, ":", len(years))

            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member_gridbox_mean)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # #print the dimensions of the ensemble members
    print("ensemble members shape", np.shape(ensemble_members))

    # #print the ensemble members count
    print("ensemble members count", ensemble_members_count)

    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(
        ensemble_mean, coords=member_gridbox_mean.coords, dims=member_gridbox_mean.dims
    )

    return ensemble_mean, years, ensemble_members_count


# Define a new function to calculate the model NAO index
# like process_model_data_for_plot_timeseries
# but for the NAO index


def calculate_model_nao_anoms(
    model_data,
    models,
    azores_grid,
    iceland_grid,
    snao_south_grid,
    snao_north_grid,
    nao_type="default",
    lag=None,
):
    """
    Calculates the model NAO index for each ensemble member and the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    azores_grid (dict): Latitude and longitude coordinates of the Azores grid point.
    iceland_grid (dict): Latitude and longitude coordinates of the Iceland grid point.
    snao_south_grid (dict): Latitude and longitude coordinates of the southern SNAO grid point.
    snao_north_grid (dict): Latitude and longitude coordinates of the northern SNAO grid point.
    nao_type (str, optional): Type of NAO index to calculate, by default 'default'. Also supports 'snao'.
    lag (int, optional): The lag to be applied. Default is None.

    Returns:
    ensemble_mean_nao_anoms (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    ensemble_members_nao_anoms (list): The NAO index anomalies for each ensemble member.
    years (numpy.ndarray): The years.
    ensemble_members_count (dict): The number of ensemble members for each model.
    """

    # Initialize a list for the ensemble members
    ensemble_members_nao_anoms = []

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Set the ensemble members count to zero
        # if model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # depending on the NAO type
            # set up the region grid
            if nao_type == "default":
                print("Calculating model NAO index using default definition")

                # Set up the dict for the southern box
                south_gridbox_dict = azores_grid
                # Set up the dict for the northern box
                north_gridbox_dict = iceland_grid
            elif nao_type == "snao":
                print("Calculating model NAO index using SNAO definition")

                # Set up the dict for the southern box
                south_gridbox_dict = snao_south_grid
                # Set up the dict for the northern box
                north_gridbox_dict = snao_north_grid
            else:
                print("Invalid NAO type")
                sys.exit()

            # Extract the lat and lon values
            # from the gridbox dictionary
            # first for the southern box
            s_lon1, s_lon2 = south_gridbox_dict["lon1"], south_gridbox_dict["lon2"]
            s_lat1, s_lat2 = south_gridbox_dict["lat1"], south_gridbox_dict["lat2"]

            # second for the northern box
            n_lon1, n_lon2 = north_gridbox_dict["lon1"], north_gridbox_dict["lon2"]
            n_lat1, n_lat2 = north_gridbox_dict["lat1"], north_gridbox_dict["lat2"]

            # Take the mean over the lat and lon values
            # for the southern box for the ensemble member
            try:
                south_gridbox_mean = member.sel(
                    lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)
                ).mean(dim=["lat", "lon"])
                north_gridbox_mean = member.sel(
                    lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)
                ).mean(dim=["lat", "lon"])
            except Exception as e:
                print(f"Error taking gridbox mean: {e}")
                sys.exit()

            # Calculate the NAO index for the ensemble member
            try:
                nao_index = south_gridbox_mean - north_gridbox_mean
            except Exception as e:
                print(f"Error calculating NAO index: {e}")
                sys.exit()

            # Extract the years
            years = nao_index.time.dt.year.values

            # Append the ensemble member to the list of ensemble members
            ensemble_members_nao_anoms.append(nao_index)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members_nao_anoms = np.array(ensemble_members_nao_anoms)

    # If the lag is not None
    # then lag the ensemble members
    if lag is not None:
        # Lag the ensemble members
        ensemble_members_nao_anoms = lag_ensemble(
            ensemble_members_nao_anoms, lag, NAO_index=True
        )

        # Multiply the ensemble members count by the lag
        ensemble_members_count = {k: v * lag for k, v in ensemble_members_count.items()}

    # #print the dimensions of the ensemble members
    print("ensemble members shape", np.shape(ensemble_members_nao_anoms))

    # #print the ensemble members count
    print("ensemble members count", ensemble_members_count)

    # Take the equally weighted ensemble mean
    ensemble_mean_nao_anoms = ensemble_members_nao_anoms.mean(axis=0)

    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean_nao_anoms = xr.DataArray(
        ensemble_mean_nao_anoms, coords=nao_index.coords, dims=nao_index.dims
    )

    return (
        ensemble_mean_nao_anoms,
        ensemble_members_nao_anoms,
        years,
        ensemble_members_count,
    )


# Define a new function to calculate the model NAO index
# like process_model_data_for_plot_timeseries
# but for the NAO index
def calculate_model_nao_anoms_matching(
    model_data,
    models,
    azores_grid,
    iceland_grid,
    snao_south_grid,
    snao_north_grid,
    nao_type="default",
):
    """
    Calculates the model NAO index for each ensemble member and the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    azores_grid (dict): Latitude and longitude coordinates of the Azores grid point.
    iceland_grid (dict): Latitude and longitude coordinates of the Iceland grid point.
    snao_south_grid (dict): Latitude and longitude coordinates of the southern SNAO grid point.
    snao_north_grid (dict): Latitude and longitude coordinates of the northern SNAO grid point.
    nao_type (str, optional): Type of NAO index to calculate, by default 'default'. Also supports 'snao'.

    Returns:
    ensemble_mean_nao_anoms (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    ensemble_members_nao_anoms (list): The NAO index anomalies for each ensemble member.
    years (numpy.ndarray): The years.
    ensemble_members_count (dict): The number of ensemble members for each model.
    """

    # Initialize a list for the ensemble members
    ensemble_members_nao_anoms = {}

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Set the ensemble members count to zero
        # if model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # depending on the NAO type
            # set up the region grid
            if nao_type == "default":
                print("Calculating model NAO index using default definition")

                # Set up the dict for the southern box
                south_gridbox_dict = azores_grid
                # Set up the dict for the northern box
                north_gridbox_dict = iceland_grid
            elif nao_type == "snao":
                print("Calculating model NAO index using SNAO definition")

                # Set up the dict for the southern box
                south_gridbox_dict = snao_south_grid
                # Set up the dict for the northern box
                north_gridbox_dict = snao_north_grid
            else:
                print("Invalid NAO type")
                sys.exit()

            # Extract the attributes
            member_id = member.attrs["variant_label"]

            # Print the model and member id
            print("calculated NAO for model", model, "member", member_id)

            # Extract the attributes from the member
            attributes = member.attrs

            # Extract the lat and lon values
            # from the gridbox dictionary
            # first for the southern box
            s_lon1, s_lon2 = south_gridbox_dict["lon1"], south_gridbox_dict["lon2"]
            s_lat1, s_lat2 = south_gridbox_dict["lat1"], south_gridbox_dict["lat2"]

            # second for the northern box
            n_lon1, n_lon2 = north_gridbox_dict["lon1"], north_gridbox_dict["lon2"]
            n_lat1, n_lat2 = north_gridbox_dict["lat1"], north_gridbox_dict["lat2"]

            # Take the mean over the lat and lon values
            # for the southern box for the ensemble member
            try:
                south_gridbox_mean = member.sel(
                    lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)
                ).mean(dim=["lat", "lon"])
                north_gridbox_mean = member.sel(
                    lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)
                ).mean(dim=["lat", "lon"])
            except Exception as e:
                print(f"Error taking gridbox mean: {e}")
                sys.exit()

            # Calculate the NAO index for the ensemble member
            try:
                nao_index = south_gridbox_mean - north_gridbox_mean
            except Exception as e:
                print(f"Error calculating NAO index: {e}")
                sys.exit()

            # Extract the years
            years = nao_index.time.dt.year.values

            # Associate the attributes with the NAO index
            nao_index.attrs = attributes

            # If model is not in the ensemble_members_nao_anoms
            # then add it to the ensemble_members_nao_anoms
            if model not in ensemble_members_nao_anoms:
                ensemble_members_nao_anoms[model] = []

            # Append the ensemble member to the list of ensemble members
            ensemble_members_nao_anoms[model].append(nao_index)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    return ensemble_members_nao_anoms, years, ensemble_members_count


def calculate_field_stats(
    observed_data,
    model_data,
    models,
    variable,
    lag=None,
    NAO_matched=False,
    measure="acc",
    matched_var_ensemble_members=None,
):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. Before calculating the spatial correlations between the two datasets.

    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    variable (str): The variable to be plotted.
    lag (int, optional): The lag to be used for the spatial correlations, by default None.
    NAO_matched (bool, optional): Whether to use the NAO matched model data, by default False.
    measure (str, optional): The measure to be used for the spatial correlations, by default 'acc'.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    # try:
    # Process the model data and calculate the ensemble mean
    if type(model_data) == dict:
        if lag is None:
            (
                ensemble_mean,
                lat,
                lon,
                years,
                ensemble_members_count,
                years_constrained,
                ensemble_members,
            ) = process_model_data_for_plot(model_data, models)
        else:
            (
                ensemble_mean,
                lat,
                lon,
                years,
                ensemble_members_count,
                years_constrained,
                ensemble_members,
            ) = process_model_data_for_plot(model_data, models, lag=lag)

            # Select only the constrained years for the obs
            observed_data = observed_data.sel(
                time=observed_data.time.dt.year.isin(years_constrained)
            )
    else:
        print("The type of model data is:", type(model_data))

        # Set the ensemble mean to the model data
        ensemble_mean = model_data

        # Extract the lat and lon values
        lat = ensemble_mean.lat.values
        lon = ensemble_mean.lon.values

        # Extract the years
        years = ensemble_mean.time.dt.year.values

        # Set the ensemble members count to 1
        ensemble_members_count = None

    # Debug the model data
    # #print("ensemble mean within spatial correlation function:", ensemble_mean)
    # print("shape of ensemble mean within spatial correlation function:", np.shape(ensemble_mean))

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
    # print('observed years', obs_years)
    # print('model years', years)

    # If NAO_matched is false
    if NAO_matched == False:

        # Find the years that are in both the observed and model data
        years_in_both = np.intersect1d(obs_years, years)

        # print('years in both', years_in_both)

        # Select only the years that are in both the observed and model data
        observed_data = observed_data.sel(
            time=observed_data.time.dt.year.isin(years_in_both)
        )
        ensemble_mean = ensemble_mean.sel(
            time=ensemble_mean.time.dt.year.isin(years_in_both)
        )

        # Remove years with NaNs
        observed_data, ensemble_mean, _, _ = remove_years_with_nans(
            observed_data, ensemble_mean, variable
        )

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
    observed_data_array = observed_data.values
    if type(model_data) != dict and variable in ["tas", "sfcWind", "rsds"]:
        ensemble_mean_array = ensemble_mean["__xarray_dataarray_variable__"].values
    else:
        ensemble_mean_array = ensemble_mean.values

    # #print the values and shapes of the observed and model data
    print("observed data shape", np.shape(observed_data_array))
    print("model data shape", np.shape(ensemble_mean_array))
    # print("observed data", observed_data_array)
    # print("model data", ensemble_mean_array)

    # Check that the observed data and ensemble mean have the same shape
    if observed_data_array.shape != ensemble_mean_array.shape:
        print("Observed data and ensemble mean must have the same shape.")
        print("observed data shape", np.shape(observed_data_array))
        print("model data shape", np.shape(ensemble_mean_array))
        print(f"variable = {variable}")
        if variable in ["var131", "var132", "ua", "va", "Wind", "wind"]:
            print("removing the vertical dimension")
            # using the .squeeze() method
            ensemble_mean_array = ensemble_mean_array.squeeze()
            print(
                "model data shape after removing vertical dimension",
                np.shape(ensemble_mean_array),
            )
            print("observed data shape", np.shape(observed_data_array))

    if measure == "acc":
        # Calculate the correlations between the observed and model data
        rfield, pfield = calculate_correlations(
            observed_data_array, ensemble_mean_array, obs_lat, obs_lon
        )

        # Set up the variable names
        stat_field = rfield
        pfield = pfield
    elif measure == "msss":
        # Calculate the RMSE between the observed and model data
        rmse, rmse_pfield = calculate_msss(
            observed_data_array, ensemble_mean_array, obs_lat, obs_lon
        )

        # Set up the variable names
        stat_field = rmse
        pfield = rmse_pfield
    elif measure == "rpc":
        # Set up the ensemble members to be used
        if matched_var_ensemble_members is not None:
            ensemble_members = matched_var_ensemble_members

            # calculate the rpc field for the matched members
            rpc, rpc_pfield = calculate_rpc_field(
                observed_data_array,
                ensemble_mean_array,
                ensemble_members,
                obs_lat,
                obs_lon,
                nao_matched=True,
            )
        else:
            ensemble_members = ensemble_members

            # Calculate the rpc between the observed and model data
            # in the non-nao matched case
            rpc, rpc_pfield = calculate_rpc_field(
                observed_data_array,
                ensemble_mean_array,
                ensemble_members,
                obs_lat,
                obs_lon,
                nao_matched=False,
            )

        # Set up the variable names
        stat_field = rpc
        pfield = rpc_pfield
    else:
        raise ValueError("Invalid measure")

    return (
        stat_field,
        pfield,
        obs_lons_converted,
        lons_converted,
        ensemble_members_count,
    )


# TODO: define a new function called calculate_correlations_timeseries
# which will calculate the time series for obs and model 1D arrays for each grid box
def calculate_correlations_timeseries(
    observed_data, model_data, models, variable, region
):
    """
    Calculates the correlation coefficients and p-values between the observed and model data for the given
    models, variable, and region.

    Args:
        observed_data (pandas.DataFrame): The observed data.
        model_data (dict): A dictionary containing the model data for each model.
        models (list): A list of model names to calculate correlations for.
        variable (str): The variable to calculate correlations for.
        region (str): The region to calculate correlations for.

    Returns:
        dict: A dictionary containing the correlation coefficients and p-values for each model.
    """

    # First check the dimensions of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("model data shape", np.shape(model_data))

    # Print the region being processed
    print("region being processed in calculate_correlations_timeseries", region)

    # Model data still needs to be processed to a 1D array
    # this is done by using process_model_data_for_plot_timeseries
    ensemble_mean, model_years, ensemble_members_count = (
        process_model_data_for_plot_timeseries(model_data, models, region)
    )

    # Print the shape of the ensemble mean
    print("ensemble mean shape", np.shape(ensemble_mean))

    # Find the years that are in both the observed and model data
    obs_years = observed_data.time.dt.year.values
    # Find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, model_years)

    # Select only the years that are in both the observed and model data
    observed_data = observed_data.sel(
        time=observed_data.time.dt.year.isin(years_in_both)
    )
    ensemble_mean = ensemble_mean.sel(
        time=ensemble_mean.time.dt.year.isin(years_in_both)
    )

    # Remove years with NaNs
    observed_data, ensemble_mean, obs_years, model_years = remove_years_with_nans(
        observed_data, ensemble_mean, variable
    )

    # Convert both the observed and model data to numpy arrays
    observed_data_array = observed_data.values
    ensemble_mean_array = ensemble_mean.values

    # Check that the observed data and ensemble mean have the same shape
    if observed_data_array.shape != ensemble_mean_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    # Using the new function calculate_correlations_1D
    r, p = calculate_correlations_1D(observed_data_array, ensemble_mean_array)

    # Return the correlation coefficients and p-values
    return (
        r,
        p,
        ensemble_mean_array,
        observed_data_array,
        ensemble_members_count,
        obs_years,
        model_years,
    )


# Define a new function to calculate the correlations between the observed and model data
# for the NAO index time series
def calculate_nao_correlations(obs_nao, model_nao, variable):
    """
    Calculates the correlation coefficients between the observed North Atlantic Oscillation (NAO) index and the NAO indices
    of multiple climate models.

    Args:
        obs_nao (array-like): The observed NAO index values.
        model_nao (dict): A dictionary containing the NAO index values for each climate model.
        models (list): A list of strings representing the names of the climate models.

    Returns:
        A dictionary containing the correlation coefficients between the observed NAO index and the NAO indices of each
        climate model.
    """

    # First check the dimensions of the observed and model data
    print("observed data shape", np.shape(obs_nao))
    print("model data shape", np.shape(model_nao))

    # Find the years that are in both the observed and model data
    obs_years = obs_nao.time.dt.year.values
    model_years = model_nao.time.dt.year.values

    # print the years
    print("observed years", obs_years)
    print("model years", model_years)

    # Find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, model_years)

    # Select only the years that are in both the observed and model data
    obs_nao = obs_nao.sel(time=obs_nao.time.dt.year.isin(years_in_both))
    model_nao = model_nao.sel(time=model_nao.time.dt.year.isin(years_in_both))

    # Remove years with NaNs
    obs_nao, model_nao, obs_years, model_years = remove_years_with_nans(
        obs_nao, model_nao, variable
    )

    # Convert both the observed and model data to numpy arrays
    obs_nao_array = obs_nao.values
    model_nao_array = model_nao.values

    # Check that the observed data and ensemble mean have the same shape
    if obs_nao_array.shape != model_nao_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    # Using the new function calculate_correlations_1D
    r, p = calculate_correlations_1D(obs_nao_array, model_nao_array)

    # Return the correlation coefficients and p-values
    return r, p, model_nao_array, obs_nao_array, model_years, obs_years


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
        print("observed data shape", np.shape(observed_data))
        print("model data shape", np.shape(model_data))

        # Loop over the latitudes and longitudes
        for y in range(len(obs_lat)):
            for x in range(len(obs_lon)):
                # set up the obs and model data
                obs = observed_data[:, y, x]
                mod = model_data[:, y, x]

                # # Print the obs and model data
                # print("observed data", obs)
                # print("model data", mod)

                # If all of the values in the obs and model data are NaN
                if np.isnan(obs).all() or np.isnan(mod).all():
                    # #print a warning
                    # print("Warning: All NaN values detected in the data.")
                    # print("Skipping this grid point.")
                    # print("")

                    # Set the correlation coefficient and p-value to NaN
                    rfield[y, x], pfield[y, x] = np.nan, np.nan

                    # Continue to the next grid point
                    continue

                # If there are any NaN values in the obs or model data
                if np.isnan(obs).any() or np.isnan(mod).any():
                    # #print a warning
                    print("Warning: NaN values detected in the data.")
                    print("Setting rfield and pfield to NaN.")

                    # Set the correlation coefficient and p-value to NaN
                    rfield[y, x], pfield[y, x] = np.nan, np.nan

                    # Continue to the next grid point
                    continue

                # Calculate the correlation coefficient and p-value
                r, p = stats.pearsonr(obs, mod)

                # #print the correlation coefficient and p-value
                # #print("correlation coefficient", r)
                # #print("p-value", p)

                # If the correlation coefficient is negative, set the p-value to NaN
                # if r < 0:
                # p = np.nan

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p

        # #print the range of the correlation coefficients and p-values
        # to 3 decimal places
        # print(f"Correlation coefficients range from {rfield.min():.3f} to {rfield.max():.3f}")
        # print(f"P-values range from {pfield.min():.3f} to {pfield.max():.3f}")

        # Return the correlation coefficients and p-values
        return rfield, pfield

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        sys.exit()


# Define a new function to calculate the one dimensional correlations
# between the observed and model data


def calculate_correlations_1D(observed_data, model_data):
    """
    Calculates the correlations between the observed and model data.

    Parameters:
    observed_data (numpy.ndarray): The processed observed data.
    model_data (numpy.ndarray): The processed model data.

    Returns:
    r (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    p (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """

    # Initialize empty arrays for the spatial correlations and p-values
    r = []
    p = []

    # Verify that the observed and model data have the same shape
    if observed_data.shape != model_data.shape:
        raise ValueError("Observed data and model data must have the same shape.")

    # Verify that they don't contain all NaN values
    if np.isnan(observed_data).all() or np.isnan(model_data).all():
        # #print a warning
        print("Warning: All NaN values detected in the data.")
        print("exiting the script")
        sys.exit()

    # Calculate the correlation coefficient and p-value
    r, p = stats.pearsonr(observed_data, model_data)

    # return the correlation coefficient and p-value
    return r, p


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

    # print("var name for obs", obs_var_name)

    for year in observed_data.time.dt.year.values[::-1]:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # print("data type", (type(data)))
        # print("data vaues", data)
        # print("data shape", np.shape(data))

        # If there are any NaN values in the data
        if np.isnan(data.values).any():
            # If there are only NaN values in the data
            if np.isnan(data.values).all():
                # Select the year from the observed data
                observed_data = observed_data.sel(
                    time=observed_data.time.dt.year != year
                )

                # for the model data
                ensemble_mean = ensemble_mean.sel(
                    time=ensemble_mean.time.dt.year != year
                )

                print(year, "all NaN values for this year")
        # if there are no NaN values in the data for a year
        # then #print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no NaN values for this year")

            # exit the loop
            break

    # Set up the years to be returned
    obs_years = observed_data.time.dt.year.values
    model_years = ensemble_mean.time.dt.year.values

    return observed_data, ensemble_mean, obs_years, model_years


# plot the correlations and p-values


def plot_correlations(
    models,
    rfield,
    pfield,
    obs,
    variable,
    region,
    season,
    forecast_range,
    plots_dir,
    obs_lons_converted,
    lons_converted,
    azores_grid,
    iceland_grid,
    uk_n_box,
    uk_s_box,
    ensemble_members_count=None,
    p_sig=0.05,
):
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
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Extract the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # Extract the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box["lon1"], uk_n_box["lon2"]
    uk_n_lat1, uk_n_lat2 = uk_n_box["lat1"], uk_n_box["lat2"]

    # Extract the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box["lon1"], uk_s_box["lon2"]
    uk_s_lat1, uk_s_lat2 = uk_s_box["lat1"], uk_s_box["lat2"]

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
    if region == "global":
        lats = obs.lat
        lons = lons_converted
    # if the region is not global
    elif region == "north-atlantic":
        lats = obs.lat
        lons = lons_converted
    else:
        # print("Error: region not found")
        sys.exit()

    # Set the font size for the plots
    plt.rcParams.update({"font.size": 12})

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
    ax.plot(
        [azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1],
        [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1],
        color="green",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1],
        [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1],
        color="green",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )

    # # Add green lines outlining the northern and southern UK index boxes
    ax.plot(
        [uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1],
        [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1],
        color="green",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    # ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=ccrs.PlateCarree())

    # Add filled contours
    # Contour levels
    clevs = np.arange(-1, 1.1, 0.1)
    # Contour levels for p-values
    clevs_p = np.arange(0, 1.1, 0.1)
    # Plot the filled contours
    cf = plt.contourf(
        lons, lats, rfield, clevs, cmap="RdBu_r", transform=ccrs.PlateCarree()
    )

    # If the variables is 'tas'
    # then we want to invert the stippling
    # so that stippling is plotted where there is no significant correlation
    if variable == "tas":
        # replace values in pfield that are less than 0.05 with nan
        pfield[pfield < p_sig] = np.nan
    else:
        # replace values in pfield that are greater than 0.05 with nan
        pfield[pfield > p_sig] = np.nan

    # #print the pfield
    # #print("pfield mod", pfield)

    # Add stippling where rfield is significantly different from zero
    plt.contourf(
        lons, lats, pfield, hatches=["...."], alpha=0, transform=ccrs.PlateCarree()
    )

    # Add colorbar
    cbar = plt.colorbar(cf, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Correlation Coefficient")

    # extract the model name from the list
    # given as ['model']
    # we only want the model name
    # if the length of the list is 1
    # then the model name is the first element
    if len(models) == 1:
        model = models[0]
    elif len(models) > 1:
        models = "multi-model mean"
    else:
        # print("Error: model name not found")
        sys.exit()

    # Set up the significance threshold
    # if p_sig is 0.05, then sig_threshold is 95%
    sig_threshold = int((1 - p_sig) * 100)

    # Extract the number of ensemble members from the ensemble_members_count dictionary
    # if the ensemble_members_count is not None
    if ensemble_members_count is not None:
        total_no_members = sum(ensemble_members_count.values())

    # Add title
    plt.title(
        f"{models} {variable} {region} {season} {forecast_range} Correlation Coefficients, p < {p_sig} ({sig_threshold}%), N = {total_no_members}"
    )

    # set up the path for saving the figure
    fig_name = f"{models}_{variable}_{region}_{season}_{forecast_range}_N_{total_no_members}_p_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Function for plotting the results for all of the models as 12 subplots


def plot_correlations_subplots(
    models,
    obs,
    variable_data,
    variable,
    region,
    season,
    forecast_range,
    plots_dir,
    azores_grid,
    iceland_grid,
    uk_n_box,
    uk_s_box,
    p_sig=0.05,
):
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
    plt.rcParams.update({"font.size": 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # Set up the lats and lons for the northern UK index box
    uk_n_lon1, uk_n_lon2 = uk_n_box["lon1"], uk_n_box["lon2"]
    uk_n_lat1, uk_n_lat2 = uk_n_box["lat1"], uk_n_box["lat2"]

    # Set up the lats and lons for the southern UK index box
    uk_s_lon1, uk_s_lon2 = uk_s_box["lon1"], uk_s_box["lon2"]
    uk_s_lat1, uk_s_lat2 = uk_s_box["lat1"], uk_s_box["lat2"]

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
        fig, axs = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(18, 12),
            subplot_kw={"projection": proj},
            gridspec_kw={"wspace": 0.1},
        )
        # Remove the last subplot
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 11:
        fig, axs = plt.subplots(
            nrows=4,
            ncols=3,
            figsize=(18, 16),
            subplot_kw={"projection": proj},
            gridspec_kw={"wspace": 0.1},
        )
        axs[-1, -1].remove()
        # Set up where to plot the title
        title_index = 1
    elif nmodels == 12:
        fig, axs = plt.subplots(
            nrows=4,
            ncols=3,
            figsize=(18, 16),
            subplot_kw={"projection": proj},
            gridspec_kw={"wspace": 0.1},
        )
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
        # print("Processing model:", model)

        # Convert the model to a single index list
        model = [model]

        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = (
            calculate_field_stats(obs, variable_data, model, variable)
        )

        # Set up the converted lons
        lons_converted = lons_converted - 180

        # Set up the lats and lons
        # if the region is global
        if region == "global":
            lats = obs.lat
            lons = lons_converted
        # if the region is not global
        elif region == "north-atlantic":
            lats = obs.lat
            lons = lons_converted
        else:
            # print("Error: region not found")
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
        ax.plot(
            [azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1],
            [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )
        ax.plot(
            [iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1],
            [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )

        # # Add green lines outlining the northern and southern UK index boxes
        # ax.plot([uk_n_lon1, uk_n_lon2, uk_n_lon2, uk_n_lon1, uk_n_lon1], [uk_n_lat1, uk_n_lat1, uk_n_lat2, uk_n_lat2, uk_n_lat1], color='green', linewidth=2, transform=proj)
        # ax.plot([uk_s_lon1, uk_s_lon2, uk_s_lon2, uk_s_lon1, uk_s_lon1], [uk_s_lat1, uk_s_lat1, uk_s_lat2, uk_s_lat2, uk_s_lat1], color='green', linewidth=2, transform=proj)

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap="RdBu_r", transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == "tas":
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=["...."], alpha=0, transform=proj)

        # Add title
        # ax.set_title(f"{model} {variable} {region} {season} {forecast_range} Correlation Coefficients")

        # extract the model name from the list
        if len(model) == 1:
            model = model[0]
        elif len(model) > 1:
            model = "all_models"
        else:
            # print("Error: model name not found")
            sys.exit()

        # Add textbox with model name
        ax.text(
            0.05,
            0.95,
            model,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add the contourf object to the list
        cf_list.append(cf)

        # If this is the centre subplot on the first row, set the title for the figure
        if i == title_index:
            # Add title
            ax.set_title(
                f"{variable} {region} {season} years {forecast_range} Correlation Coefficients, p < {p_sig} ({sig_threshold}%)",
                fontsize=12,
            )

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(
        cf_list[0],
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=fig.axes,
        shrink=0.8,
    )
    cbar.set_label("Correlation Coefficient")

    # Specify a tight layout
    # plt.tight_layout()

    # set up the path for saving the figure
    fig_name = f"{variable}_{region}_{season}_{forecast_range}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # # Adjust the vertical spacing between the plots
    # plt.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

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
        # print("Error: variable not found")
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
        # print("Error: variable not found")
        sys.exit()
    return obs_var_name


def calculate_spatial_correlations_bootstrap(
    observed_data,
    model_data,
    models,
    variable,
    n_bootstraps=1000,
    experiment=None,
    lag=None,
    matched_var_ensemble_members=None,
    ensemble_mean=None,
    measure=None,
):
    """
    The method involves creating 1,000 bootstrapped hindcasts from a finite ensemble size and a finite number of validation years.
    The steps involved in creating the bootstrapped hindcasts are as follows:

    1) Randomly select N cases (validation years) with replacement.
        To take autocorrelation into account, this is done in blocks of five consecutive years.
    2) For each case, randomly select M ensemble members with replacement.
        Compute the ensemble mean from these M samples.
    3) Compute the evaluation metrics (ACC, MSSS, RPC, and skill difference) with
        the resultant ensemble mean prediction.
    4) Repeat steps 1-3 1,000 times to create a sample distribution of the
        evaluation metrics.

    For the ACC and MSSS, the p-value is defined as the ratio of negative values from the
        bootstrapped sample distribution on the basis of a one-tailed test of the hypothesis
            that the prediction skill is greater than 0.

    Arguments:
        observed_data (xarray.core.dataset.Dataset): The processed observed data.
        model_data (dict): The processed model data.
        models (list): The list of models to be plotted.
        variable (str): The variable name.
        n_bootstraps (int): The number of bootstraps to perform. Default is 1000.

    Returns:
        rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
        pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data. Bootstrapped.
    """

    # Process the model data into an array
    if type(model_data) == dict:
        print("the type of model data is a dictionary")
        if lag is None:
            (
                ensemble_mean,
                lat,
                lon,
                years,
                ensemble_members_count,
                years_constrained,
                ensemble_members,
            ) = process_model_data_for_plot(model_data, models)
        else:
            print("Applying lag")
            (
                ensemble_mean,
                lat,
                lon,
                years,
                ensemble_members_count,
                years_constrained,
                ensemble_members,
            ) = process_model_data_for_plot(model_data, models, lag=lag)

            # Select only the constrained years for the obs
            observed_data = observed_data.sel(
                time=observed_data.time.dt.year.isin(years_constrained)
            )
    else:
        print("the type of model data is: ", type(model_data))
        if type(matched_var_ensemble_members) == None and type(ensemble_mean) == None:
            raise AttributeError(
                "matched_var_ensemble_members and ensemble_mean must be specified if model_data is not a dictionary"
            )

        # Set the ensemble members as the matched_var_ensemble_members
        ensemble_members = matched_var_ensemble_members[
            "__xarray_dataarray_variable__"
        ].values

        # # Extract the lat and lon values
        # lat = ensemble_mean.lat.values
        # lon = ensemble_mean.lon.values

        # # Extract the years
        # years = ensemble_mean.time.dt.year.values

        # Set the ensemble members count to None
        ensemble_members_count = None

    # if observed data is not a numpy array
    if type(observed_data) != np.ndarray:
        print("observed data is not a numpy array")
        # convert observed data to a numpy array
        observed_data = observed_data.values
        # if the experiment is dcppA-hindcast

        # # constrain the years to the years that are in both the observed and model data
        # observed_data = observed_data[3:, :, :]

    # # Print the types of the observed and model data
    # print("observed data type", type(observed_data))
    # print("model data type", type(model_data))

    # Print the shapes of the observed and model data
    print("observed data shape", np.shape(observed_data))
    print("model data shape", np.shape(ensemble_members))

    # print the values of the observed and model data
    # print("observed data", observed_data)
    # print("model data", model_data)

    # Check that the observed and model data have the same type
    if type(observed_data) != type(ensemble_members):
        raise ValueError("Observed data and model data must have the same type.")

    # # Print the years extracted from the observed and model data
    # print("observed years", obs_years)
    # print("model years", model_years)

    # Print the values of each to check
    print("observed data year constrained", np.shape(observed_data))
    print("model data year constrained", np.shape(ensemble_members))

    # Now we want to check that there are no NaNs in the observed and model data
    if np.isnan(observed_data).any():
        raise ValueError("Observed data contains NaNs.")

    if np.isnan(ensemble_members).any():
        raise ValueError("Model data contains NaNs.")

    if matched_var_ensemble_members is None:
        # Now we want to check that the observed and model data have the same shape
        # for all dimensions of the observed data
        # and the final 3 dimensions of the model data
        model_data_shape = ensemble_members[0, :, :, :]
    else:
        model_data_shape = ensemble_members[:, 0, :, :]

    # for brevity set up the lats and lons
    lats = observed_data[0, :, 0]
    lons = observed_data[0, 0, :]

    print("observed data shape", np.shape(observed_data))
    print("model data shape", np.shape(model_data_shape))
    # if the shapes are not the same
    if observed_data.shape != model_data_shape.shape:
        raise ValueError("Observed data and model data must have the same shape.")

    # create an empty array for the p-values
    # dim = (1000, lat, lon)
    pfield_dist = np.empty(
        [n_bootstraps, len(observed_data[0, :, 0]), len(observed_data[0, 0, :])]
    )
    # create an empty array for the correlation coefficients
    rfield_dist = np.empty(
        [n_bootstraps, len(observed_data[0, :, 0]), len(observed_data[0, 0, :])]
    )

    # Print the shapes of the pfield and rfield arrays
    print("pfield array shape", np.shape(pfield_dist))
    print("rfield array shape", np.shape(rfield_dist))

    # Print the types of the pfield and rfield arrays
    print("pfield array type", type(pfield_dist))
    print("rfield array type", type(rfield_dist))

    # # Take the time mean of the observed data
    # observed_data_tm = np.mean(observed_data, axis=0)

    # Extract the number of validation years
    # this is the second dimension of the model data
    if matched_var_ensemble_members is None:
        n_validation_years = len(ensemble_members[0, :, 0, 0])

        # Extract the number of ensemble members
        # this is the first dimension of the model data
        m_ensemble_members = len(ensemble_members[:, 0, 0, 0])
    else:
        # Swap the axes of the matched_var_ensemble_members
        # so that the ensemble members axis is the first axis
        matched_var_ensemble_members = np.swapaxes(matched_var_ensemble_members, 0, 1)

        # Extract the number of validation years
        # this is the second dimension of the model data
        n_validation_years = len(matched_var_ensemble_members[0, :, 0, 0])

        # Extract the number of ensemble members
        # this is the first dimension of the model data
        m_ensemble_members = len(matched_var_ensemble_members[:, 0, 0, 0])

    # set up the block size for the autocorrelation
    block_size = 5  # years

    # Save the original model data
    model_data_original = model_data.copy()

    # print the number of validation years
    print("number of validation years", n_validation_years)

    # print the number of ensemble members
    print("number of ensemble members", m_ensemble_members)

    # First we want to loop over the bootstraps
    for i in range(n_bootstraps):
        # Randomly select N cases (validation years) with replacement.
        # To take autocorrelation into account, this is done in blocks of five consecutive years.
        # Create

        # # print the number bootstrap
        # print("bootstrap number", i)

        # Randomly select block start indices
        block_starts = resample(
            range(0, n_validation_years - block_size + 1, block_size),
            n_samples=n_validation_years // block_size,
            replace=True,
        )

        # Create indices for the entire blocks
        block_indices = []
        for start in block_starts:
            block_indices.extend(range(start, start + block_size))

        # Ensure we have exactly N indices (with replacement)
        if len(block_indices) < n_validation_years:
            block_indices.extend(
                resample(
                    block_indices,
                    n_samples=n_validation_years - len(block_indices),
                    replace=True,
                )
            )

        # Create a mask for the selected block indices
        mask = np.zeros(n_validation_years, dtype=bool)
        mask[block_indices] = True

        # Apply the mask to select the corresponding block of data for the model data
        n_mask_model_data = ensemble_members[:, mask, :, :]

        # Apply the mask to select the corresponding block of data for the observed data
        n_mask_observed_data = observed_data[mask, :, :]

        ensemble_resampled = resample(
            n_mask_model_data, n_samples=m_ensemble_members, replace=True
        )

        # # Print the dimensions of the ensemble resampled
        # print("ensemble resampled shape", np.shape(ensemble_resampled))
        # print("model data original shape masked", np.shape(model_data_original[:, mask, :, :]))

        # # Check if ensemble_resampled is different from model_data
        # if not np.array_equal(ensemble_resampled, model_data_original[:, mask, :, :]):
        #     print("Ensemble has been resampled")
        # else:
        #     print("Ensemble has not been resampled")

        # Calculate the ensemble mean
        ensemble_mean = np.mean(ensemble_resampled, axis=0)

        # if the measure is acc
        if measure == "acc":
            # Call the function to get the r and p fields
            rfield, _ = calculate_correlations(
                n_mask_observed_data, ensemble_mean, lats, lons
            )
        # if the measure is msss
        elif measure == "msss":
            msss_field, _ = calculate_msss(
                n_mask_observed_data, ensemble_mean, lats, lons
            )

            # Set the rfield to the msss_field
            rfield = msss_field
        elif measure == "rpc":
            rpc_field, _ = calculate_rpc_field(
                n_mask_observed_data,
                ensemble_mean,
                ensemble_resampled,
                lats,
                lons,
                nao_matched=False,
            )

            # Set the rfield to the rpc_field
            rfield = rpc_field
        else:
            print("Error: measure not found")

        # append the correlation coefficients and p-values to the arrays
        rfield_dist[i, :, :] = rfield

    # Print the shapes of the pfield and rfield arrays
    print("rfield array shape", np.shape(rfield_dist))

    # Print the types of the pfield and rfield arrays
    print("rfield array type", type(rfield_dist))

    # Now we want to obtain the p-values for the correlations
    # first create an empty array for the p-values
    pfield_bootstrap = np.empty([len(lats), len(lons)])

    # if the measure is ACC or MSSS
    # we want to obtain the p-value from the ratio of negative values from the bootstrapped
    # sample distribution on the basis of a one-tailed test of the hypothesis that the prediction skill is greater than 0.
    if measure == "acc" or measure == "msss":
        # Now loop over the lats and lons
        for y in range(len(lats)):
            # print("y", y)
            for x in range(len(lons)):
                # print("x", x)
                # # print the shape of the rfield_dist array
                # print("rfield_dist shape", np.shape(rfield_dist))
                # set up the rfield_dist and pfield_dist
                rfield_sample = rfield_dist[:, y, x]

                # Calculate the p-value
                pfield_bootstrap[y, x] = np.sum(rfield_sample < 0) / n_bootstraps
    elif measure == "rpc":
        # Now loop over the lats and lons
        for y in range(len(lats)):
            for x in range(len(lons)):
                # Calculate the 2.5% and 97.5% percentiles of the bootstrapped sample distribution
                pct_2p5, pct_97p5 = np.percentile(rfield_dist, [2.5, 97.5])

                # Check if the percentiles cross 1
                if pct_2p5 > 1 or pct_97p5 < 1:
                    # If the percentiles do not cross 1, the RPC is significantly different from 1
                    pfield_bootstrap[y, x] = 1
                else:
                    # If the percentiles cross 1, the RPC is not significantly different from 1
                    pfield_bootstrap[y, x] = 0
    else:
        print("Error: measure not found")

    # Print the shape of the pfield_bootstrap array
    print("pfield_bootstrap shape", np.shape(pfield_bootstrap))

    # Print the type of the pfield_bootstrap array
    print("pfield_bootstrap type", type(pfield_bootstrap))

    # Return the p-values
    return pfield_bootstrap


def forecast_stats(obs, forecast1, forecast2, no_boot=1000):
    """
    Assess and compares two forecasts, using a block bootstrap for uncertanties.

    Based on Doug Smith's 'fcsts_assess' function.

    Inputs:

        obs[time, lat, lon] (array) = timeseries of observations

        forecast1[member, time] = forecast1 ensemble

        forecast2[member, time] = forecast2 ensemble

        nboot = number of bootstraps = no. of bootstrap samples to use.
                                        Default is 1000.

    Outputs:

        corr1: correlation between forecast1 ensemble mean and observations

        corr1_min, corr1_max, corr1_p: 5% to 95% uncertainties and p value

        corr2: correlation between forecast2 ensemble mean and observations

        corr2_min, corr2_max, corr2_p: 5% to 95% uncertainties and p value

        corr10: correlation between forecast1 ensemble mean and observations for 10 ensemble members

        corr10_min, corr10_max, corr10_p: 5% to 95% uncertainties and p value

        msss1: mean squared skill score between forecast1 ensemble mean and observations

        msss1_min, msss1_max, msss1_p: 5% to 95% uncertainties and p value

        rpc1: ratio of predictable components for forecast1

        rpc1_min, rpc1_max, rpc1_p: 5% to 95% uncertainties and p value

        rpc2: ratio of predictable components for forecast2

        rpc2_min, rpc2_max, rpc2_p: 5% to 95% uncertainties and p value

        corr_diff: corr1 - corr2

        corr_diff_min, corr_diff_max, corr_diff_p: 5% to 95% uncertainties and p value

        partial_r: partial correlation between obs and forecast1 ensemble mean, after removing
                    influence of forecast2 ensemble mean.

        partial_r_min, partial_r_max, partial_r_p: 5% to 95% uncertainties and p value

        partial_r_bias: bias in partial correlation

        nens1: number of ensemble members in forecast1

        nens2: number of ensemble members in forecast2

        sigo: standard deviation of observations

        obs_resid: residual after regressing out forecast2 ensemble mean from observations

        sigo_resid: standard deviation of obs_resid

        forecast1_em_resid: residual after regressing out forecast2 ensemble mean from forecast1 ensemble mean

        f1_ts: forecast1 ensemble mean timeseries

        f2_ts: forecast2 ensemble mean timeseries

        f10_ts: forecast1 ensemble mean timeseries for 10 ensemble members

        o_ts: observations timeseries

    """

    # Set up the dictionary for the outputs
    # missing data indicator
    mdi = -9999.0

    # Set up the dictionary for the outputs
    forecasts_stats = {
        "corr1": mdi,
        "corr1_min": mdi,
        "corr1_max": mdi,
        "corr1_p": mdi,
        "corr2": mdi,
        "corr2_min": mdi,
        "corr2_max": mdi,
        "corr2_p": mdi,
        "corr10": mdi,
        "corr10_min": mdi,
        "corr10_max": mdi,
        "corr10_p": mdi,
        "msss1": mdi,
        "msss1_min": mdi,
        "msss1_max": mdi,
        "msss1_p": mdi,
        "corr12": mdi,
        "corr12_min": mdi,
        "corr12_max": mdi,
        "corr12_p": mdi,
        "rpc1": mdi,
        "rpc1_min": mdi,
        "rpc1_max": mdi,
        "rpc1_p": mdi,
        "rpc2": mdi,
        "rpc2_min": mdi,
        "rpc2_max": mdi,
        "rpc2_p": mdi,
        "corr_diff": mdi,
        "corr_diff_min": mdi,
        "corr_diff_max": mdi,
        "corr_diff_p": mdi,
        "partialr": mdi,
        "partialr_min": mdi,
        "partialr_max": mdi,
        "partialr_p": mdi,
        "partialr_bias": mdi,
        "nens1": mdi,
        "nens2": mdi,
        "sigo": mdi,
        "sigo_resid": mdi,
        "obs_resid": [],
        "fcst1_em_resid": [],
        "f1_ts": [],
        "f2_ts": [],
        "f10_ts": [],
        "o_ts": [],
        "f1_ts_short": [],
        "o_ts_short": [],
        "corr1_short": mdi,
        "corr1_p_short": mdi,
    }

    # Set up the number of times from the obs
    # the size of the first dimension of the obs
    n_times = obs.shape[0]

    # N_times short
    n_times_short = n_times - 10

    # Set up the number of lats and lons
    n_lats = obs.shape[1]
    n_lons = obs.shape[2]

    # Extract the number of ensemble members for the first forecast
    nens1 = forecast1.shape[0]

    # Extract the number of ensemble members for the second forecast
    # also divide this into two halves
    nens2 = np.shape(forecast2)[0]
    nens2_2 = int(nens2 / 2 + 1)

    # Set up the number of bootstraps
    nboot = no_boot

    # Set up the shapes of the arrays to be filled
    r_partial_boot = np.zeros([nboot, n_lats, n_lons])
    r_partial_bias_boot = np.zeros([nboot, n_lats, n_lons])

    r1o_boot = np.zeros([nboot, n_lats, n_lons])
    r1o_boot_short = np.zeros([nboot, n_lats, n_lons])
    r2o_boot = np.zeros([nboot, n_lats, n_lons])
    r12_boot = np.zeros([nboot, n_lats, n_lons])

    # sig_f1 = np.zeros([nboot, n_lats, n_lons]) ; sig_f2 = np.zeros([nboot, n_lats, n_lons])

    rdiff_boot = np.zeros([nboot, n_lats, n_lons])
    rpc1_boot = np.zeros([nboot, n_lats, n_lons])
    rpc2_boot = np.zeros([nboot, n_lats, n_lons])

    r_ens_10_boot = np.zeros([nboot, n_lats, n_lons])
    msss1_boot = np.zeros([nboot, n_lats, n_lons])

    # Set up the block length for the block bootstrap
    block_length = 5

    # Set up the number of blocks to be used
    n_blocks = int(n_times / block_length)

    # if the nblocks * block_length is less than n_times
    # add one to the number of blocks
    if n_blocks * block_length < n_times:
        n_blocks = n_blocks + 1

    # set up the indexes
    # for the time - time needs to be the same for all forecasts and obs
    index_time = range(n_times - block_length + 1)

    # Create a short time index
    index_time_short = range(n_times_short - block_length + 1)

    # For the members
    index_ens1 = range(nens1)
    index_ens2 = range(nens2)

    # Loop over the bootstraps
    for iboot in np.arange(nboot):

        print("bootstrap index", iboot)
        # Select ensemble members and the starting indicies for the blocks
        # for the first forecast just use the raw data
        if iboot == 0:
            index_ens1_this = index_ens1

            index_ens2_this = index_ens2

            # normal order of time
            index_time_this = range(0, n_times, block_length)

            # normal order of time for short time
            index_time_short_this = range(0, n_times_short, block_length)

        else:  # pick random samples
            # Create an array containing random indices

            index_ens1_this = np.array([random.choice(index_ens1) for _ in index_ens1])

            index_ens2_this = np.array([random.choice(index_ens2) for _ in index_ens2])

            # Create an array containing random indices for the blocks
            index_time_this = np.array(
                [random.choice(index_time) for _ in range(n_blocks)]
            )

            # Create an array containing random indices for the blocks
            # For the short time
            index_time_short_this = np.array(
                [random.choice(index_time_short) for _ in range(n_blocks)]
            )

        # Create am empty array to store the observations
        obs_boot = np.zeros([n_times, n_lats, n_lons])
        obs_boot_short = np.zeros([n_times_short, n_lats, n_lons])

        # Create an empty array to store the first forecast
        fcst1_boot = np.zeros([nens1, n_times, n_lats, n_lons])
        fcst1_boot_short = np.zeros([nens1, n_times_short, n_lats, n_lons])

        fcst2_boot = np.zeros([nens2, n_times, n_lats, n_lons])

        # Create an empty array for the 10 member forecast
        fcst10_boot = np.zeros([10, n_times, n_lats, n_lons])

        # Loop over the blocks
        # First set the time to 0
        itime = 0

        for ithis in index_time_this:

            # Set up the individual block index
            index_block = np.arange(ithis, ithis + block_length)

            # If the block index is greater than the number of times, then reduce the block index
            index_block[(index_block > n_times - 1)] = (
                index_block[(index_block > n_times - 1)] - n_times
            )

            # Select a subset of indices for the block
            index_block = index_block[: min(block_length, n_times - itime)]

            # Loop over the block indices
            for iblock in index_block:

                # print("block index", iblock)
                # print("time index", itime)
                # print("shape of obs_boot", np.shape(obs_boot))
                # print("shape of obs", np.shape(obs))

                # Extract the observations for the block
                obs_boot[itime, :, :] = obs[iblock, :, :]

                # Extract the first forecast for the block and random ensemble members
                fcst1_boot[:, itime, :, :] = forecast1[index_ens1_this, iblock, :, :]

                # Extract the second forecast for the block and random ensemble members
                fcst2_boot[:, itime, :, :] = forecast2[index_ens2_this, iblock, :, :]

                # Extract the 10 member forecast for the block and random ensemble members
                fcst10_boot[:, itime, :, :] = forecast1[
                    index_ens1_this[0:10], iblock, :, :
                ]

                # Increment the time
                itime += 1

        # Loop over the short time index
        itime_short = 0

        for ithis in index_time_short_this:

            # Set up the individual block index
            index_block_short = np.arange(ithis, ithis + block_length)

            # If the block index is greater than the number of times, then reduce the block index
            index_block_short[(index_block_short > n_times_short - 1)] = (
                index_block_short[(index_block_short > n_times_short - 1)]
                - n_times_short
            )

            # Select a subset of indices for the block
            index_block_short = index_block_short[
                : min(block_length, n_times_short - itime_short)
            ]

            # Loop over the block indices
            for iblock in index_block_short:

                # Extract the observations for the block
                obs_boot_short[itime_short, :, :] = obs[iblock, :, :]

                # Extract the first forecast for the block and random ensemble members
                fcst1_boot_short[:, itime_short, :, :] = forecast1[
                    index_ens1_this, iblock, :, :
                ]

                # Increment the time
                itime_short += 1

        # Process the stats
        o = obs_boot
        o_short = obs_boot_short

        print("shape of obs_boot", np.shape(o))
        # print("value of obs_boot", o)

        # Get the ensemble mean forecast
        # Should these be calculated for each lat lon, or does numpy do that for me?
        f1 = np.mean(fcst1_boot, axis=0)
        f1_short = np.mean(fcst1_boot_short, axis=0)
        f2 = np.mean(fcst2_boot, axis=0)

        # Get the 10 member ensemble mean forecast
        f10 = np.mean(fcst10_boot, axis=0)

        # TODO: Extract the first bootstrap value of f1, f2 and f10
        # For the ensemble mean time series plots
        if iboot == 0:
            f1_ts = f1
            f1_short_i1 = f1_short
            f2_ts = f2
            f10_ts = f10

            o_ts = o
            o_ts_short_i1 = o_short

        # Compute the bias by removing independent estimates of f2 - the historical ensemble mean
        # first half of the ensemble
        f2_1 = np.mean(fcst2_boot[0:nens2_2, :, :, :], axis=0)

        # Compute the second sample for the second half od the ensemble
        # second half of the ensemble
        f2_2 = np.mean(fcst2_boot[nens2_2:-1, :, :, :], axis=0)

        # Loop over the gridpoints to calculate the correlations
        for lat in range(n_lats):
            for lon in range(n_lons):
                # Extract the forecasts and obs
                f1_cell = f1[:, lat, lon]

                # f1_cell for the short period
                f1_cell_short = f1_short[:, lat, lon]

                f2_cell = f2[:, lat, lon]
                f10_cell = f10[:, lat, lon]
                o_cell = o[:, lat, lon]

                # o_cell for the short period
                o_cell_short = o_short[:, lat, lon]

                # If all the values of o_cell are 0
                if np.all(o_cell == 0.0):
                    # Print a warning
                    print("Warning: all values of o_cell are 0 at lat", lat, "lon", lon)
                    print(
                        "Setting all values of the correlations to NaN at lat",
                        lat,
                        "lon",
                        lon,
                    )
                    # Set all the values of the correlations to NaN
                    r1o_boot[iboot, lat, lon] = np.nan
                    r2o_boot[iboot, lat, lon] = np.nan
                    r12_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the differences in correlations to NaN
                    rdiff_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the MSSS to NaN
                    msss1_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the RPCs to NaN
                    rpc1_boot[iboot, lat, lon] = np.nan
                    rpc2_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the 10 member ensemble mean correlations to NaN
                    r_ens_10_boot[iboot, lat, lon] = np.nan

                    # set the r partial correlations to NaN
                    r_partial_boot[iboot, lat, lon] = np.nan

                    # Set the r partial bias to NaN
                    r_partial_bias_boot[iboot, lat, lon] = np.nan

                    # Continue to the next lat lon
                    continue
                # elif f1_cell contains NaNs
                elif (
                    np.isnan(f1_cell).any()
                    or np.isnan(f2_cell).any()
                    or np.isnan(o_cell).any()
                    or np.isnan(f10_cell).any()
                ):
                    # Print a warning
                    # print("Warning: f1_cell contains NaNs at lat", lat, "lon", lon)
                    # print(
                    #     "Setting all values of the correlations to NaN at lat",
                    #     lat,
                    #     "lon",
                    #     lon,
                    # )
                    # Set all the values of the correlations to NaN
                    r1o_boot[iboot, lat, lon] = np.nan
                    r2o_boot[iboot, lat, lon] = np.nan
                    r12_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the differences in correlations to NaN
                    rdiff_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the MSSS to NaN
                    msss1_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the RPCs to NaN
                    rpc1_boot[iboot, lat, lon] = np.nan
                    rpc2_boot[iboot, lat, lon] = np.nan

                    # Set all the values of the 10 member ensemble mean correlations to NaN
                    r_ens_10_boot[iboot, lat, lon] = np.nan

                    # set the r partial correlations to NaN
                    r_partial_boot[iboot, lat, lon] = np.nan

                    # Set the r partial bias to NaN
                    r_partial_bias_boot[iboot, lat, lon] = np.nan

                    # Continue to the next lat lon
                    continue

                # Extract the forecasts and obs for the independent estimates
                f2_1_cell = f2_1[:, lat, lon]
                f2_2_cell = f2_2[:, lat, lon]

                # Perform the correlations
                r12, _ = pearsonr(f1_cell, f2_cell)
                r1o, _ = pearsonr(f1_cell, o_cell)
                r2o, _ = pearsonr(f2_cell, o_cell)
                r_ens_10_boot[iboot, lat, lon], _ = pearsonr(f10_cell, o_cell)

                # Perform the correlations for the short period
                r1o_short, _ = pearsonr(f1_cell_short, o_cell_short)

                # Assign values to bootstrap arrays
                r1o_boot[iboot, lat, lon] = r1o
                r2o_boot[iboot, lat, lon] = r2o
                r12_boot[iboot, lat, lon] = r12
                r1o_boot_short[iboot, lat, lon] = r1o_short

                # Difference in correlations
                rdiff_boot[iboot, lat, lon] = r1o - r2o

                # Calculate the mean squared skill score
                msss1_boot[iboot, lat, lon] = msss(o_cell, f1_cell)

                # Calculate the variance of th noise for forecast1
                # var_noise_f1 = np.var(fcst1_boot[:, :, lat, lon] - f1, axis=0)

                # Calculate the standard deviations of the forecasts1 and 2
                sig_f1 = np.std(f1_cell)
                sig_f2 = np.std(f2_cell)

                # Calculate the RPC scores
                rpc1_boot[iboot, lat, lon] = r1o / (
                    sig_f1 / np.std(fcst1_boot[:, :, lat, lon])
                )

                rpc2_boot[iboot, lat, lon] = r2o / (
                    sig_f2 / np.std(fcst2_boot[:, :, lat, lon])
                )

                # Calculate the biased partial correlation - full ensemble, no seperation
                # Set up the denominator for this
                denom_sq = (1.0 - r2o**2) * (1.0 - r12**2)

                # Set up the numerator
                num = r1o - r12 * r2o

                # Calculate the partial correlation
                r_partial_boot[iboot, lat, lon] = num / np.sqrt(denom_sq)

                # Compute the correlations for the independent estimates
                r12_1, _ = pearsonr(f1_cell, f2_1_cell)

                r2o_1, _ = pearsonr(f2_1_cell, o_cell)

                r2o_2, _ = pearsonr(f2_2_cell, o_cell)

                # Compute the standard deviations of the independent estimates
                sig_o = np.std(o_cell)
                sig_f2_1 = np.std(f2_1_cell)
                sig_f2_2 = np.std(f2_2_cell)

                # Calculate the residuals for the forecast using the first half of the ensemble
                res_f1 = f1_cell - r12_1 * f2_1_cell * (sig_f1 / sig_f2_1)

                # Calculate the residuals for the observations using the first half of the ensemble
                res_o_1 = o_cell - r2o_1 * f2_1_cell * (sig_o / sig_f2_1)

                # Calculate the residuals for the observations using the second half of the ensemble
                res_o_2 = o_cell - r2o_2 * f2_2_cell * (sig_o / sig_f2_2)

                # Calculate the correlations for the biased partial correlation - same half of members
                # correlations between first half of members forecast and obs residuals
                rp_biased, _ = pearsonr(res_f1, res_o_1)

                # Calculate the correlations for the unbiased partial correlation - different half of members
                # correlations between first half of members forecast and obs residuals for the second half of members
                rp_unbiased, _ = pearsonr(res_f1, res_o_2)

                # Calculate the r_partial_bias
                r_partial_bias_boot[iboot, lat, lon] = rp_biased - rp_unbiased

    # Append the stats - are these the right shape for the dictionary?
    # TODO: fix these to be the right shape for lat lon
    # correlation between forecast1 ensemble mean and observations for non-bootstrapped data
    forecasts_stats["corr1"] = r1o_boot[0]

    # short correlations
    forecasts_stats["corr1_short"] = r1o_boot_short[0]

    forecasts_stats["corr1_min"] = np.percentile(r1o_boot, 5, axis=0)  # 5% uncertainty

    forecasts_stats["corr1_max"] = np.percentile(
        r1o_boot, 95, axis=0
    )  # 95% uncertainty

    # Initialize the count of values arrays
    count_vals_r1o = np.zeros([n_lats, n_lons])
    count_vals_r1o_short = np.zeros([n_lats, n_lons])
    count_vals_r2o = np.zeros([n_lats, n_lons])

    count_vals_r_ens_10 = np.zeros([n_lats, n_lons])
    count_vals_msss1 = np.zeros([n_lats, n_lons])

    count_vals_r12 = np.zeros([n_lats, n_lons])
    count_vals_rdiff = np.zeros([n_lats, n_lons])

    count_vals_rpc1 = np.zeros([n_lats, n_lons])
    count_vals_rpc2 = np.zeros([n_lats, n_lons])

    count_vals_r_partial = np.zeros([n_lats, n_lons])

    # Initialize the correlation arrays
    r1o_p = np.zeros([n_lats, n_lons])
    r1o_p_short = np.zeros([n_lats, n_lons])
    r2o_p = np.zeros([n_lats, n_lons])

    r_ens_10_p = np.zeros([n_lats, n_lons])
    msss1_p = np.zeros([n_lats, n_lons])

    r12_p = np.zeros([n_lats, n_lons])
    rdiff_p = np.zeros([n_lats, n_lons])

    rpc1_p = np.zeros([n_lats, n_lons])
    rpc2_p = np.zeros([n_lats, n_lons])

    r_partial_p = np.zeros([n_lats, n_lons])

    # TODO: Modify this to include significant negative correlations as well
    # TODO: Read through Doug's paper to figure out what is going on here
    for lat in range(n_lats):
        for lon in range(n_lons):
            # Extract the forecasts and obs bootstrapped data for the cell
            r1o_boot_cell = r1o_boot[:, lat, lon]
            r1o_boot_cell_short = r1o_boot_short[:, lat, lon]
            r2o_boot_cell = r2o_boot[:, lat, lon]

            r_ens_10_boot_cell = r_ens_10_boot[:, lat, lon]

            msss1_boot_cell = msss1_boot[:, lat, lon]

            r12_boot_cell = r12_boot[:, lat, lon]

            rdiff_boot_cell = rdiff_boot[:, lat, lon]

            rpc1_boot_cell = rpc1_boot[:, lat, lon]
            rpc2_boot_cell = rpc2_boot[:, lat, lon]

            r_partial_boot_cell = r_partial_boot[:, lat, lon]

            # Calculate the p-values
            # TODO: add a function to calculate the p-values here and do count vals thing
            # Process the count_vals
            count_vals_r1o[lat, lon] = np.sum(
                i < 0.0 for i in r1o_boot_cell
            )  # count of negative values

            count_vals_r1o_short[lat, lon] = np.sum(
                i < 0.0 for i in r1o_boot_cell_short
            )  # count of negative values

            count_vals_r2o[lat, lon] = np.sum(
                i < 0.0 for i in r2o_boot_cell
            )  # count of negative values

            count_vals_r_ens_10[lat, lon] = np.sum(
                i < 0.0 for i in r_ens_10_boot_cell
            )  # count of negative values

            count_vals_msss1[lat, lon] = np.sum(
                i < 0.0 for i in msss1_boot_cell
            )  # count of negative values

            count_vals_r12[lat, lon] = np.sum(
                i < 0.0 for i in r12_boot_cell
            )  # count of negative values

            count_vals_rdiff[lat, lon] = np.sum(
                i < 0.0 for i in rdiff_boot_cell
            )  # count of negative values

            # count of values less than 1 fo RPC
            count_vals_rpc1[lat, lon] = np.sum(i < 1.0 for i in rpc1_boot_cell)

            # count of values less than 1 fo RPC
            count_vals_rpc2[lat, lon] = np.sum(i < 1.0 for i in rpc2_boot_cell)

            count_vals_r_partial[lat, lon] = np.sum(
                i < 0.0 for i in r_partial_boot_cell
            )  # count of negative values

            # Calculate the p-values
            r1o_p[lat, lon] = float(count_vals_r1o[lat, lon]) / nboot

            r1o_p_short[lat, lon] = float(count_vals_r1o_short[lat, lon]) / nboot

            r2o_p[lat, lon] = float(count_vals_r2o[lat, lon]) / nboot

            r_ens_10_p[lat, lon] = float(count_vals_r_ens_10[lat, lon]) / nboot

            msss1_p[lat, lon] = float(count_vals_msss1[lat, lon]) / nboot

            r12_p[lat, lon] = float(count_vals_r12[lat, lon]) / nboot

            rdiff_p[lat, lon] = float(count_vals_rdiff[lat, lon]) / nboot

            rpc1_p[lat, lon] = float(count_vals_rpc1[lat, lon]) / nboot

            rpc2_p[lat, lon] = float(count_vals_rpc2[lat, lon]) / nboot

            r_partial_p[lat, lon] = float(count_vals_r_partial[lat, lon]) / nboot

    # Append the p-values to the dictionary
    forecasts_stats["corr1_p"] = r1o_p
    forecasts_stats["corr1_p_short"] = r1o_p_short
    forecasts_stats["corr2_p"] = r2o_p

    forecasts_stats["corr10_p"] = r_ens_10_p
    forecasts_stats["msss1_p"] = msss1_p

    forecasts_stats["corr12_p"] = r12_p
    forecasts_stats["corr_diff_p"] = rdiff_p

    forecasts_stats["rpc1_p"] = rpc1_p
    forecasts_stats["rpc2_p"] = rpc2_p

    forecasts_stats["partialr_p"] = r_partial_p

    # correlation between forecast2 ensemble mean and observations for non-bootstrapped data
    forecasts_stats["corr2"] = r2o_boot[0]

    forecasts_stats["corr2_min"] = np.percentile(r2o_boot, 5, axis=0)  # 5% uncertainty

    forecasts_stats["corr2_max"] = np.percentile(
        r2o_boot, 95, axis=0
    )  # 95% uncertainty

    # correlation between 10 member forecast ensemble mean and observations for non-bootstrapped data
    forecasts_stats["corr10"] = np.percentile(r_ens_10_boot, 50, axis=0)

    forecasts_stats["corr10_min"] = np.percentile(
        r_ens_10_boot, 5, axis=0
    )  # 5% uncertainty

    forecasts_stats["corr10_max"] = np.percentile(
        r_ens_10_boot, 95, axis=0
    )  # 95% uncertainty

    # mean squared skill score between forecast1 ensemble mean and observations for non-bootstrapped data
    forecasts_stats["msss1"] = msss1_boot[0]

    forecasts_stats["msss1_min"] = np.percentile(
        msss1_boot, 5, axis=0
    )  # 5% uncertainty

    forecasts_stats["msss1_max"] = np.percentile(
        msss1_boot, 95, axis=0
    )  # 95% uncertainty

    # correlation between forecast1 and forecast2 ensemble means for non-bootstrapped data
    forecasts_stats["corr12"] = r12_boot[0]

    forecasts_stats["corr12_min"] = np.percentile(r12_boot, 5, axis=0)  # 5% uncertainty

    forecasts_stats["corr12_max"] = np.percentile(
        r12_boot, 95, axis=0
    )  # 95% uncertainty

    # corr1 - corr2 for non-bootstrapped data
    forecasts_stats["corr_diff"] = rdiff_boot[0]

    forecasts_stats["corr_diff_min"] = np.percentile(
        rdiff_boot, 5, axis=0
    )  # 5% uncertainty

    forecasts_stats["corr_diff_max"] = np.percentile(
        rdiff_boot, 95, axis=0
    )  # 95% uncertainty

    # ratio of predictable components for forecast1 for non-bootstrapped data
    forecasts_stats["rpc1"] = rpc1_boot[0]

    forecasts_stats["rpc1_min"] = np.percentile(rpc1_boot, 5, axis=0)  # 5% uncertainty

    forecasts_stats["rpc1_max"] = np.percentile(
        rpc1_boot, 95, axis=0
    )  # 95% uncertainty

    # ratio of predictable components for forecast2 for non-bootstrapped data
    forecasts_stats["rpc2"] = rpc2_boot[0]

    forecasts_stats["rpc2_min"] = np.percentile(rpc2_boot, 5, axis=0)  # 5% uncertainty

    forecasts_stats["rpc2_max"] = np.percentile(
        rpc2_boot, 95, axis=0
    )  # 95% uncertainty

    # Adjusted partial correlation

    adjust_bias = np.percentile(r_partial_bias_boot, 50, axis=0)  # 50% uncertainty

    r_partial_boot = r_partial_boot - adjust_bias  # adjust for bias

    # bias in partial correlation
    forecasts_stats["partialr_bias"] = adjust_bias

    # partial correlation between obs and forecast1 ensemble mean for non-bootstrapped data
    forecasts_stats["partialr"] = r_partial_boot[0]

    forecasts_stats["partialr_min"] = np.percentile(
        r_partial_boot, 5, axis=0
    )  # 5% uncertainty

    forecasts_stats["partialr_max"] = np.percentile(
        r_partial_boot, 95, axis=0
    )  # 95% uncertainty

    # Calculate the residuals for the observations

    f1 = np.mean(forecast1, axis=0)
    f2 = np.mean(forecast2, axis=0)

    # Initialize arrays for the standard deviations
    sig1 = np.zeros([n_lats, n_lons])
    sig2 = np.zeros([n_lats, n_lons])

    sigo = np.zeros([n_lats, n_lons])

    sigo_resid = np.zeros([n_lats, n_lons])

    obs_resid = np.zeros([n_times, n_lats, n_lons])

    fcst1_em_resid = np.zeros([n_times, n_lats, n_lons])

    # Loop over the gridpoints to calculate the correlations
    for lat in range(n_lats):
        for lon in range(n_lons):
            # Extract the forecasts and obs for the cell
            f1_cell = f1[:, lat, lon]
            f2_cell = f2[:, lat, lon]

            # NOTE: This needs to be obs from the arguments
            # Not o from the bootstrapped data
            o_cell = obs[:, lat, lon]

            # Calculate the standard deviations of the forecasts1 and 2
            sig1_cell = np.std(f1_cell)
            sig2_cell = np.std(f2_cell)

            # Calculate the standard deviation of the observations
            sigo_cell = np.std(o_cell)

            # Append the standard deviations to the arrays
            sig1[lat, lon] = sig1_cell
            sig2[lat, lon] = sig2_cell

            sigo[lat, lon] = sigo_cell

            # Calculate the residuals for the observations
            obs_resid[:, lat, lon] = o_cell - r2o_boot[0, lat, lon] * f2_cell * (
                sigo_cell / sig2_cell
            )

            # Calculate the residuals for the forecast1 ensemble mean
            fcst1_em_resid[:, lat, lon] = f1_cell - r12_boot[0, lat, lon] * f2_cell * (
                sig1_cell / sig2_cell
            )

            # Calculate the standard deviation of the
            # residuals for the observations
            sigo_resid[lat, lon] = np.nanstd(obs_resid[:, lat, lon])

    # Append the standard deviations to the dictionary
    forecasts_stats["sigo"] = sigo
    forecasts_stats["sigo_resid"] = sigo_resid

    # Append the residuals to the dictionary
    forecasts_stats["obs_resid"] = obs_resid

    forecasts_stats["fcst1_em_resid"] = fcst1_em_resid

    # Append the ensemble members count to the dictionary
    forecasts_stats["nens1"] = nens1
    forecasts_stats["nens2"] = nens2

    # Append the forecasts to the dictionary
    forecasts_stats["f1_ts"] = f1_ts
    forecasts_stats["f2_ts"] = f2_ts

    # Add the short time series for the first forecast
    forecasts_stats["f1_ts_short"] = f1_short_i1

    # Add the short time series for the observations
    forecasts_stats["o_ts_short"] = o_ts_short_i1

    forecasts_stats["f10_ts"] = f10_ts

    forecasts_stats["o_ts"] = o_ts

    # Return the forecasts_stats dictionary
    return forecasts_stats


# Define a function which will calculate the mean squared skill score


def msss(obs, forecast):
    """

    Calculate the mean squared skill score. Given the observations and forecast.

    Inputs:
        obs[time] = time series of observations
        forecast[time] = time series of forecast

    Outputs:
        msss = mean squared skill score

    """

    # Extract the number of times
    n_times = len(obs)

    # Calculate the numerator
    numerator = (forecast - obs) ** 2

    # Calculate the denominator
    denominator = (obs - np.mean(obs)) ** 2

    # Calculate the mean squared skill score
    msss = 1 - (np.sum(numerator) / np.sum(denominator))

    # Return the mean squared skill score
    return msss


# Write a new function which will plot a series of subplots
# for the same variable, region and forecast range (e.g. psl global years 2-9)
# but with different seasons (e.g. DJFM, MAM, JJA, SON)
# TODO: this doesn't include bootstrapped p values
def plot_seasonal_correlations(
    models,
    observations_path,
    variable,
    region,
    region_grid,
    forecast_range,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    obs_var_name,
    azores_grid,
    iceland_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
    north_sea_grid=None,
    central_europe_grid=None,
    snao_south_grid=None,
    snao_north_grid=None,
):
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
    north_sea_grid : dict, optional
        Dictionary of North Sea grid. The default is None.
    central_europe_grid : dict, optional
        Dictionary of Central Europe grid. The default is None.

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
    ax_labels = ["A", "B", "C", "D"]

    # Loop over the seasons
    for i in range(len(seasons_list_obs)):

        # Print the season(s) being processed
        print("obs season", seasons_list_obs[i])
        print("mod season", seasons_list_mod[i])

        # Process the observations
        obs = process_observations(
            variable,
            region,
            region_grid,
            forecast_range,
            seasons_list_obs[i],
            observations_path,
            obs_var_name,
        )

        # Print the shape of the observations
        print("obs shape", np.shape(obs))

        # Load and process the model data
        model_datasets = load_data(
            dic.base_dir, models, variable, region, forecast_range, seasons_list_mod[i]
        )
        # Process the model data
        model_data, model_time = process_data(model_datasets, variable)

        # Print the shape of the model data
        print("model shape", np.shape(model_data))

        # If the variable is 'rsds'
        # divide the obs data by 86400 to convert from J/m2 to W/m2
        if variable == "rsds":
            obs /= 86400

        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = (
            calculate_field_stats(obs, model_data, models, variable)
        )

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
    plt.rcParams.update({"font.size": 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # If the north sea grid is not None
    if north_sea_grid is not None:
        # Set up the lats and lons for the north sea grid
        north_sea_lon1, north_sea_lon2 = north_sea_grid["lon1"], north_sea_grid["lon2"]
        north_sea_lat1, north_sea_lat2 = north_sea_grid["lat1"], north_sea_grid["lat2"]

    # If the central europe grid is not None
    if central_europe_grid is not None:
        # Set up the lats and lons for the central europe grid
        central_europe_lon1, central_europe_lon2 = (
            central_europe_grid["lon1"],
            central_europe_grid["lon2"],
        )
        central_europe_lat1, central_europe_lat2 = (
            central_europe_grid["lat1"],
            central_europe_grid["lat2"],
        )

    # If the snao south grid is not None
    if snao_south_grid is not None:
        # Set up the lats and lons for the snao south grid
        snao_south_lon1, snao_south_lon2 = (
            snao_south_grid["lon1"],
            snao_south_grid["lon2"],
        )
        snao_south_lat1, snao_south_lat2 = (
            snao_south_grid["lat1"],
            snao_south_grid["lat2"],
        )

    # If the snao north grid is not None
    if snao_north_grid is not None:
        # Set up the lats and lons for the snao north grid
        snao_north_lon1, snao_north_lon2 = (
            snao_north_grid["lon1"],
            snao_north_grid["lon2"],
        )
        snao_north_lat1, snao_north_lat2 = (
            snao_north_grid["lat1"],
            snao_north_grid["lat2"],
        )

    # # subtract 180 from all of the azores and iceland lons
    # azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    # iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # Set up the fgure size and subplot parameters
    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{variable} {region} {forecast_range} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.90)

    # Set up the significance thresholdf
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
        obs_lons_converted, lons_converted = (
            obs_lons_converted_list[i],
            lons_converted_list[i],
        )

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
        # ax.plot([azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1], [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1], color='green', linewidth=2, transform=proj)
        # ax.plot([iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1], [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1], color='green', linewidth=2, transform=proj)

        # if the north sea grid is not None
        if north_sea_grid is not None:
            # Add green lines outlining the North Sea grid
            ax.plot(
                [
                    north_sea_lon1,
                    north_sea_lon2,
                    north_sea_lon2,
                    north_sea_lon1,
                    north_sea_lon1,
                ],
                [
                    north_sea_lat1,
                    north_sea_lat1,
                    north_sea_lat2,
                    north_sea_lat2,
                    north_sea_lat1,
                ],
                color="green",
                linewidth=2,
                transform=proj,
            )

        # if the central europe grid is not None
        if central_europe_grid is not None:
            # Add green lines outlining the Central Europe grid
            ax.plot(
                [
                    central_europe_lon1,
                    central_europe_lon2,
                    central_europe_lon2,
                    central_europe_lon1,
                    central_europe_lon1,
                ],
                [
                    central_europe_lat1,
                    central_europe_lat1,
                    central_europe_lat2,
                    central_europe_lat2,
                    central_europe_lat1,
                ],
                color="green",
                linewidth=2,
                transform=proj,
            )

        # if the snao south grid is not None
        if snao_south_grid is not None:
            # Add green lines outlining the SNAO south grid
            ax.plot(
                [
                    snao_south_lon1,
                    snao_south_lon2,
                    snao_south_lon2,
                    snao_south_lon1,
                    snao_south_lon1,
                ],
                [
                    snao_south_lat1,
                    snao_south_lat1,
                    snao_south_lat2,
                    snao_south_lat2,
                    snao_south_lat1,
                ],
                color="cyan",
                linewidth=2,
                transform=proj,
            )

        # if the snao north grid is not None
        if snao_north_grid is not None:
            # Add green lines outlining the SNAO north grid
            ax.plot(
                [
                    snao_north_lon1,
                    snao_north_lon2,
                    snao_north_lon2,
                    snao_north_lon1,
                    snao_north_lon1,
                ],
                [
                    snao_north_lat1,
                    snao_north_lat1,
                    snao_north_lat2,
                    snao_north_lat2,
                    snao_north_lat1,
                ],
                color="cyan",
                linewidth=2,
                transform=proj,
            )

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap="RdBu_r", transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == "tas":
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=["...."], alpha=0, transform=proj)

        # Add a textbox with the season name
        ax.text(
            0.05,
            0.95,
            season,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # # Add a textbox with the number of ensemble members in the bottom right corner
        # ax.text(0.95, 0.05, f"N = {ensemble_members_count_list[i]}", transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbox in the bottom right with the figure letter
        # extract the figure letter from the ax_labels list
        fig_letter = ax_labels[i]
        ax.text(
            0.95,
            0.05,
            fig_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # # Set up the text for the subplot
        # ax.text(-0.1, 1.1, key, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

        # Add the contourf object to the list
        cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(
        cf_list[0],
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=fig.axes,
        shrink=0.8,
    )
    cbar.set_label("correlation coefficients")

    # print("ax_labels shape", np.shape(ax_labels))
    # for i, ax in enumerate(axs):
    #     # Add the label to the bottom left corner of the subplot
    #     ax.text(0.05, 0.05, ax_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    # Set up the path for saving the figure
    fig_name = f"{variable}_{region}_{forecast_range}_{experiment}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Now define a function which will calculate the RPC scores
def calculate_rpc_field(
    obs, model_mean, model_members, obs_lat, obs_lon, nao_matched=False
):
    """
    Calculates the RPC scores for the model data.
    """

    # Check that the shapes of the time dimensions align
    if np.shape(obs)[0] != np.shape(model_mean)[0]:
        raise ValueError(
            f"Time dimensions do not align: obs = {np.shape(obs)[0]}, model = {np.shape(model_mean)[0]}"
        )

    # Initialise empty arrays for the RPC and the significance
    rpc_field = np.empty([len(obs_lat), len(obs_lon)])
    p_field = np.empty([len(obs_lat), len(obs_lon)])

    # if nao_matched is true
    if nao_matched:
        # Extract the values from the model members
        model_members = model_members["__xarray_dataarray_variable__"].values

        # Print the shape of the model members
        print("model members shape", np.shape(model_members))

    # Loop over the lats
    for y in range(len(obs_lat)):
        for x in range(len(obs_lon)):
            # Set up the obs and model data for this lat/lon
            obs_point = obs[:, y, x]
            model_mean_point = model_mean[:, y, x]
            model_members_point = model_members[:, :, y, x]

            # # Swap round the axes of the model members
            # # BUG: What if axis have been swapped already? for bootstrapping
            # model_members_point = np.swapaxes(model_members_point, 0, 1)

            # If all of the values in the obs and model data are NaN
            if np.isnan(obs_point).all() or np.isnan(model_mean_point).all():
                # #print a warning
                # print("Warning: All NaN values detected in the data.")
                # print("Skipping this grid point.")
                # print("")

                # Set the RPC and p value to NaN
                rpc_field[y, x] = np.nan
                p_field[y, x] = np.nan

                # Skip to the next lat/lon
                continue

            # If some of the values in the obs and model data are NaN
            if np.isnan(obs_point).any() or np.isnan(model_mean_point).any():
                # #print a warning
                # print("Warning: Some NaN values detected in the data.")
                # print("Skipping this grid point.")
                # print("")

                # Set the RPC and p value to NaN
                rpc_field[y, x] = np.nan
                p_field[y, x] = np.nan

                # Skip to the next lat/lon
                continue

            # Calculate the ACC
            acc, _ = stats.pearsonr(obs_point, model_mean_point)

            # Calculate the standard deviation of the predictable signal for the forecasts
            sigma_fsig = np.std(model_mean_point)

            # Calculate the total standard deviation of the forecasts
            sigma_ftot = np.nanstd(model_members_point)

            # Where acc is negative, set rpc field to nan
            if acc < 0:
                rpc_field[y, x] = np.nan
                p_field[y, x] = np.nan
                continue

            # Calculate the rpc score
            rpc = acc / (sigma_fsig / sigma_ftot)

            # APpend the rpc score to the array
            rpc_field[y, x] = rpc

            # FIXME: Bootstrap this instead
            # Calculate the p value
            # Where rpc values are significantly greater than 1
            # using a two-tailed t-test
            # Calculate the number of degrees of freedom
            # equal to the number of years minus 1
            dof = len(obs_point) - 1

            # Calculate the t-statistic
            # equal to the rpc divided by the square root of the rpc divided by the degrees of freedom
            t_stat = rpc / np.sqrt(rpc / dof)

            # Calculate the p value
            # equal to 1 minus the cdf of the t-statistic
            # multiplied by 2 to make it a two-tailed test
            p_field[y, x] = 1 - stats.t.cdf(t_stat, dof) * 2

    # Return the RPC and p value
    return rpc_field, p_field


# Define a new function for calculating the rmse
def calculate_msss(obs, model_data, obs_lat, obs_lon):
    """
    Calculates the mean squared skill score for the model data.
    """

    # Check that the shapes of the time dimensions align
    if np.shape(obs)[0] != np.shape(model_data)[0]:
        raise ValueError(
            f"Time dimensions do not align: obs = {np.shape(obs)[0]}, model = {np.shape(model_data)[0]}"
        )

    # Initialise empty arrays for the MSSS and the significance
    msss_field = np.empty([len(obs_lat), len(obs_lon)])
    p_field = np.empty([len(obs_lat), len(obs_lon)])

    # Loop over the lats
    for y in range(len(obs_lat)):
        for x in range(len(obs_lon)):
            # Extract the obs and model data for this lat/lon
            obs_point = obs[:, y, x]
            model_point = model_data[:, y, x]

            # If all of the obs and model data are nan
            if np.isnan(obs_point).all() and np.isnan(model_point).all():
                # Set the MSSS and p value to nan
                msss_field[y, x] = np.nan
                p_field[y, x] = np.nan
                # Skip to the next lat/lon
                continue

            # If some of the values are nan
            if np.isnan(obs_point).any() or np.isnan(model_point).any():
                # print("Warning: some values are nan")
                # Set the MSSS and p value to nan
                msss_field[y, x] = np.nan
                p_field[y, x] = np.nan
                # Skip to the next lat/lon
                continue

            # Calculate the mean of the obs
            obs_point_mean = np.mean(obs_point)

            # Calculate the numerator
            # Sum over the time dimension
            # of the squared difference between the obs and model
            numerator = np.sum((model_point - obs_point) ** 2)

            # Calculate the denominator
            # Sum over the time dimension
            # of the squared difference between the obs and the obs mean
            denominator = np.sum((obs_point - obs_point_mean) ** 2)

            # Calculate the MSSS
            # and store in the array
            msss_field[y, x] = 1 - (numerator / denominator)

            # FIXME: Do this with bootstrapping instead
            # Calculate the p value
            # where MSSS is significantly different from zero, to the 95% level
            # using a two-tailed t-test
            # Calculate the number of degrees of freedom
            # equal to the number of years minus 1
            dof = len(obs_point) - 1
            # Calculate the t-statistic
            # equal to the MSSS divided by the square root of the MSSS divided by the degrees of freedom
            t_stat = msss_field[y, x] / np.sqrt(msss_field[y, x] / dof)

            # Calculate the p value
            # equal to 1 minus the cdf of the t-statistic
            # multiplied by 2 to make it a two-tailed test
            p_field[y, x] = 1 - stats.t.cdf(t_stat, dof) * 2

    # Return the MSSS and p value
    return msss_field, p_field


# Plot the seasonal correlations for the raw ensemble, the lagged ensemble and the NAO-matched ensemble
# TODO: work the bootstrapped p values into this function
# TODO: get this to work for MSESS and RPC
def plot_seasonal_correlations_raw_lagged_matched(
    models,
    observations_path,
    model_variable,
    obs_variable,
    obs_path,
    region,
    region_grid,
    forecast_range,
    start_year,
    end_year,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    save_dir,
    obs_var_name,
    azores_grid,
    iceland_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
    bootstrapped_pval=False,
    lag=4,
    no_subset_members=20,
    measure="acc",
    north_atlantic=False,
):
    """
    Plots the spatial correlation coefficients and p-values for the raw ensemble, the lagged ensemble and the NAO-matched ensemble.
    For the same variable, region and forecast range (e.g. tas global years 2-9) but with different seasons (e.g. DJFM, MAM, JJA, SON).

    Arguments:
    - models: a list of strings with the names of the models to be plotted.
    - observations_path: a string with the path to the observations file.
    - model_variable: a string with the name of the variable to be plotted for the models.
    - obs_variable: a string with the name of the variable to be plotted for the observations.
    - obs_path: a string with the path to the observations file.
    - region: a string with the name of the region to be plotted.
    - region_grid: a string with the name of the grid to be used for the region.
    - forecast_range: a string with the forecast range to be plotted.
    - start_year: an integer with the start year of the forecast range.
    - end_year: an integer with the end year of the forecast range.
    - seasons_list_obs: a list of strings with the seasons to be plotted for the observations.
    - seasons_list_mod: a list of strings with the seasons to be plotted for the models.
    - plots_dir: a string with the path to the directory where the plots will be saved.
    - save_dir: a string with the path to the directory where the processed data will be saved.
    - obs_var_name: a string with the name of the variable in the observations file.
    - azores_grid: a string with the name of the grid to be used for the Azores region.
    - iceland_grid: a string with the name of the grid to be used for the Iceland region.
    - p_sig: a float with the significance level for the p-values (default is 0.05).
    - experiment: a string with the name of the experiment to be plotted (default is 'dcppA-hindcast').
    - bootstrapped_pval: a boolean indicating whether to use bootstrapped p-values (default is False).
    - lag: an integer with the number of months to lag the data (default is 4).
    - no_subset_members: an integer with the number of members to use for the NAO-matched ensemble (default is 20).
    - measure: a string with the name of the measure to be plotted (default is 'correlation').
    - north_atlantic: a boolean indicating whether to plot the North Atlantic region (default is False).
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

    # Create the labels for the subplots
    ax_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    # Create a list of the methods to use
    methods = ["raw", "lagged", "nao_matched"]

    test_methods = ["nao_matched"]

    test_season = ["DJFM"]

    # Loop over the methods
    for method in methods:
        # Print the method being used
        print("method", method)

        # Loop over the seasons
        for i, obs_season in enumerate(seasons_list_obs):

            # Print the season(s) being processed
            print("obs season", obs_season)
            model_season = seasons_list_mod[i]
            print("mod season", model_season)

            # Process the observations
            obs = process_observations(
                obs_variable,
                region,
                region_grid,
                forecast_range,
                obs_season,
                observations_path,
                obs_var_name,
            )

            # if the variable is 'rsds'
            # divide the obs data by 86400 to convert from J/m2 to W/m2
            if obs_variable == "rsds":
                print("converting obs to W/m2")
                obs /= 86400

            # Append the processed observations to the list
            obs_list.append(obs)

            # if the method is 'raw'
            if method == "raw":
                print("method", method)
                # Load and process the model data
                model_datasets = load_data(
                    dic.base_dir,
                    models,
                    model_variable,
                    region,
                    forecast_range,
                    model_season,
                )
                # Process the model data
                model_data, model_time = process_data(model_datasets, model_variable)

                # Calculate the spatial correlations for the model
                (
                    rfield,
                    pfield,
                    obs_lons_converted,
                    lons_converted,
                    ensemble_members_count,
                ) = calculate_field_stats(
                    obs, model_data, models, model_variable, measure=measure
                )
            elif method == "lagged":
                print("method", method)
                # Load and process the model data
                model_datasets = load_data(
                    dic.base_dir,
                    models,
                    model_variable,
                    region,
                    forecast_range,
                    model_season,
                )
                # Process the model data
                model_data, model_time = process_data(model_datasets, model_variable)

                # Calculate the spatial correlations for the model
                (
                    rfield,
                    pfield,
                    obs_lons_converted,
                    lons_converted,
                    ensemble_members_count,
                ) = calculate_field_stats(
                    obs, model_data, models, obs_variable, lag=lag, measure=measure
                )
            elif method == "nao_matched":
                print("method", method)

                # process the psl observations for the nao index
                obs_psl_anomaly = read_obs(
                    "psl",
                    region,
                    forecast_range,
                    obs_season,
                    observations_path,
                    start_year,
                    end_year,
                )

                # Load and process the model data for the NAO index
                model_datasets_psl = load_data(
                    dic.base_dir,
                    dic.psl_models,
                    "psl",
                    region,
                    forecast_range,
                    model_season,
                )
                # Process the model data
                model_data_psl, _ = process_data(model_datasets_psl, "psl")

                # Make sure that the models have the same time period for psl
                model_data_psl = constrain_years(model_data_psl, dic.psl_models)

                # Remove years containing NaNs from the observations and model data
                # and align the time periods
                obs_psl_anomaly, model_data_psl, _ = remove_years_with_nans_nao(
                    obs_psl_anomaly, model_data_psl, dic.psl_models, NAO_matched=False
                )

                # Calculate the lagged NAO index
                obs_nao, model_nao = calculate_nao_index_and_plot(
                    obs_psl_anomaly,
                    model_data_psl,
                    dic.psl_models,
                    "psl",
                    obs_season,
                    forecast_range,
                    plots_dir,
                )

                # Rescale the NAO index
                rescaled_nao, ensemble_mean_nao, ensemble_members_nao, years = (
                    rescale_nao(
                        obs_nao,
                        model_nao,
                        dic.psl_models,
                        obs_season,
                        forecast_range,
                        plots_dir,
                        lag=lag,
                    )
                )

                # Perform the NAO matching for the target variableOnao
                matched_var_ensemble_mean, matched_var_ensemble_members = (
                    nao_matching_other_var(
                        rescaled_nao,
                        model_nao,
                        models,
                        model_variable,
                        obs_variable,
                        dic.base_dir,
                        models,
                        obs_path,
                        region,
                        model_season,
                        forecast_range,
                        start_year,
                        end_year,
                        plots_dir,
                        dic.save_dir,
                        lagged_years=years,
                        lagged_nao=True,
                        no_subset_members=no_subset_members,
                    )
                )

                # Set up the no_ensemble_members variables
                ensemble_members_count = no_subset_members

                # Process the observations for the NAO-matched variables
                obs_match_var = read_obs(
                    obs_variable,
                    region,
                    forecast_range,
                    obs_season,
                    observations_path,
                    start_year,
                    end_year,
                )

                # Remove years containing NaNs from the observations and model data
                (
                    obs_match_var,
                    matched_var_ensemble_mean,
                    matched_var_ensemble_members,
                ) = remove_years_with_nans_nao(
                    obs_match_var,
                    matched_var_ensemble_mean,
                    models,
                    NAO_matched=True,
                    matched_var_ensemble_members=matched_var_ensemble_members,
                )

                # Now calculate the spatial correlations
                # TODO: include matched var ensemble members here
                rfield, pfield, obs_lons_converted, lons_converted, _ = (
                    calculate_field_stats(
                        obs_match_var,
                        matched_var_ensemble_mean,
                        models,
                        obs_variable,
                        lag=lag,
                        NAO_matched=True,
                        measure=measure,
                        matched_var_ensemble_members=matched_var_ensemble_members,
                    )
                )

            else:
                print("Error: method not found")
                sys.exit()

            # Append the r and p fields to the lists
            # for each season
            # for each method
            rfield_list.append(rfield)
            pfield_list.append(pfield)

            # Append the ensemble members count to the list
            # for each season
            # for each method
            ensemble_members_count_list.append(ensemble_members_count)

            # Append the obs_lons_converted and lons_converted to the lists
            obs_lons_converted_list.append(obs_lons_converted)
            lons_converted_list.append(lons_converted)

    # Set the font size for the plots
    plt.rcParams.update({"font.size": 8})

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # Set up the fgure size and subplot parameters
    # for a 3x4 grid of subplots
    fig, axs = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(14, 12),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{model_variable} {region} {forecast_range} {experiment} {measure} coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=10, y=0.90)

    # Set up the significance thresholdf
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # # Set up the axes
    # axs = np.empty((len(methods), len(seasons_list_obs)), dtype=object)

    # Create a list to store the contourf objects
    cf_list = []

    # Loop over the methods
    for i, method in enumerate(methods):
        # Loop over the seasons
        for j, obs_season in enumerate(seasons_list_obs):

            # Print the season(s) being plotted
            print("plotting season", obs_season)

            # Set up the index for the subplot
            index = i * 4 + j

            # Extract the obs
            obs = obs_list[index]

            # Extract the r and p fields
            rfield, pfield = rfield_list[index], pfield_list[index]

            # Extract the obs_lons_converted and lons_converted
            obs_lons_converted, lons_converted = (
                obs_lons_converted_list[index],
                lons_converted_list[index],
            )

            # Set up the converted lons
            lons_converted = lons_converted - 180

            # Set up the lats and lons
            lats = obs.lat
            lons = lons_converted

            # Set up the axes
            ax = axs[j, i]

            # Add coastlines
            ax.coastlines()

            # Add greenlines outlining the Azores and Iceland grids
            ax.plot(
                [azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1],
                [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )
            ax.plot(
                [iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1],
                [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1],
                color="green",
                linewidth=2,
                transform=proj,
            )

            # If the north_atlantic flag is true
            if north_atlantic:
                # Constrain the lats and lons to the North Atlantic region
                lats = lats[35:]
                lons = lons[40:100]

                # Constrain the rfield and pfield to the North Atlantic region
                rfield = rfield[35:, 40:100]
                pfield = pfield[35:, 40:100]

            # Add filled contours
            # Contour levels
            clevs = np.arange(-1, 1.1, 0.1)
            # Contour levels for p-values
            clevs_p = np.arange(0, 1.1, 0.1)
            # Plot the filled contours
            if measure == "acc":
                cf = ax.contourf(
                    lons, lats, rfield, clevs, cmap="RdBu_r", transform=proj
                )

                if model_variable == "tas":
                    # replace values in pfield that are less than 0.05 with nan
                    pfield[pfield < p_sig] = np.nan
                else:
                    # replace values in pfield that are greater than 0.05 with nan
                    pfield[pfield > p_sig] = np.nan

                # Add stippling where rfield is significantly different from zero
                ax.contourf(
                    lons, lats, pfield, hatches=["...."], alpha=0, transform=proj
                )
            elif measure == "msss":
                cf = ax.contourf(
                    lons,
                    lats,
                    rfield,
                    clevs,
                    cmap="RdBu_r",
                    transform=proj,
                    extend="both",
                )
            elif measure == "rpc":
                clevs = np.arange(0, 2.1, 0.1)
                cf = ax.contourf(
                    lons,
                    lats,
                    rfield,
                    clevs,
                    cmap="RdBu_r",
                    transform=proj,
                    extend="max",
                )
            else:
                raise ValueError(
                    f"measure {measure} not recognised when plotting statistics"
                )

            # FIXME: No p-values until bootstrap is done
            # # If the variables is 'tas'
            # # then we want to invert the stippling
            # # so that stippling is plotted where there is no significant correlation

            # Add a textbox with the season name
            ax.text(
                0.05,
                0.95,
                obs_season,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Add a textbox with the method name
            # in the top right
            ax.text(
                0.95,
                0.95,
                method,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Add a textbox with the figure letter
            fig_letter = ax_labels[i * 4 + j]
            ax.text(
                0.95,
                0.05,
                fig_letter,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Add the contourf object to the list
            cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(
        cf_list[0],
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=fig.axes,
        shrink=0.8,
    )
    cbar.set_label("correlation coefficients")

    # Set up the path for saving the figure
    fig_name = f"{model_variable}_{region}_{forecast_range}_{experiment}_{measure}_sig-{p_sig}_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Define a function to form the lagging of the ensemble for an array


def lag_ensemble_array(
    fcst1: np.ndarray, fcst2: np.ndarray, obs_array: np.ndarray, lag: int = 4
):
    """
    Lag the ensemble forecast and observation arrays.

    Parameters
    ----------
    fcst1 : numpy.ndarray
        The first ensemble forecast array with shape (no_members, no_years, no_lats, no_lons).
    fcst2 : numpy.ndarray
        The second ensemble forecast array with shape (no_members, no_years, no_lats, no_lons).
    obs_array : numpy.ndarray
        The observation array with shape (no_years, no_lats, no_lons).
    lag : int, optional
        The number of years to lag the ensemble forecast and observation arrays, by default 4.

    Returns
    -------
        lagged_fcst1 : numpy.ndarray
            The lagged ensemble forecast array with
                shape (no_lagged_members, no_years - (lag - 1), no_lats, no_lons).
        lagged_obs : numpy.ndarray
            The lagged observation array with
                shape (no_years - (lag - 1), no_lats, no_lons).
        lagged_fcst2 : numpy.ndarray
            The lagged ensemble forecast array with
                shape (no_lagged_members, no_years - (lag - 1), no_lats, no_lons).

    """
    # Extract the no_members
    n_members = fcst1.shape[0]
    n_years = fcst1.shape[1]

    # Extract the no_lats
    n_lats = fcst1.shape[2]
    n_lons = fcst1.shape[3]

    # Set up the no_lagged_members
    n_lagged_members = n_members * lag

    # Set up the lagged ensemble
    lagged_fcst1 = np.zeros([n_lagged_members, n_years, n_lats, n_lons])

    # Loop over the ensemble members
    for member in range(n_members):
        # Loop over the years
        for year in range(n_years):
            # If the year is less than the lag
            if year < lag - 1:
                # Set the lagged ensemble member equal to NaN
                lagged_fcst1[member, year, :, :] = np.nan

                # Also set the lagged ensemble member equal to NaN
                for lag_i in range(lag):
                    lagged_fcst1[member + (lag_i * n_members), year, :, :] = np.nan
            # Otherwise
            else:
                # Loop over the lag
                for lag_i in range(lag):
                    # Set the lagged ensemble member equal to the forecast
                    lagged_fcst1[member + (lag_i * n_members), year, :, :] = fcst1[
                        member, year - lag_i, :, :
                    ]

    # Now we have the lagged ensemble
    # The first 3 years of the lagged ensemble should be NaN
    # Remove these years
    lagged_fcst1 = lagged_fcst1[:, lag - 1 :, :, :]

    # Do the same for the observations
    lagged_obs = obs_array[lag - 1 :, :, :]

    # Do the same for the second forecast
    lagged_fcst2 = fcst2[:, lag - 1 :, :, :]

    return lagged_fcst1, lagged_obs, lagged_fcst2


# Plot seasonal correlations for the wind speed at a given level
# TODO: WRIte function for plotting wind speed correlations at a given level (850 hPa)


def plot_seasonal_correlations_wind_speed(
    shared_models,
    obs_path,
    region,
    region_grid,
    forecast_range,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    azores_grid,
    iceland_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
):
    """
    Plots the seasonal correlations between the wind speed at a given level and the observed wind speed.

    Parameters:
    shared_models (list): The list of shared models to be plotted.
    obs_path (str): The path to the observed data file.
    region (str): The region to be plotted.
    region_grid (numpy.ndarray): The grid of the region to be plotted.
    forecast_range (list): The forecast range to be plotted.
    seasons_list_obs (list): The list of seasons to be plotted for the observed data.
    seasons_list_mod (list): The list of seasons to be plotted for the model data.
    plots_dir (str): The directory where the plots will be saved.
    azores_grid (numpy.ndarray): The grid of the Azores region.
    iceland_grid (numpy.ndarray): The grid of the Iceland region.
    p_sig (float): The significance level for the correlation coefficients.
    experiment (str): The name of the experiment to be plotted.

    Returns:
    None
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
    ax_labels = ["A", "B", "C", "D"]

    # Set up the list of model variables
    model_ws_variables = ["ua", "va"]

    # Set up the list of obs variables
    obs_ws_variables = ["var131", "var132"]

    # Loop over the seasons
    for i in range(len(seasons_list_obs)):
        # Print the seasons being processed
        print("obs season", seasons_list_obs[i])
        print("mod season", seasons_list_mod[i])

        # Calculate the U and V wind components for the observations
        obs_u = process_observations(
            model_ws_variables[0],
            region,
            region_grid,
            forecast_range,
            seasons_list_obs[i],
            obs_path,
            obs_ws_variables[0],
        )
        obs_v = process_observations(
            model_ws_variables[1],
            region,
            region_grid,
            forecast_range,
            seasons_list_obs[i],
            obs_path,
            obs_ws_variables[1],
        )

        # Use a try statement to catch any errors
        try:
            # Calculate the wind speed for the observations
            obs = np.sqrt(np.square(obs_u) + np.square(obs_v))
        except Exception as e:
            print(
                "Error when trying to calculate wind speeds from the obs xarrays: ", e
            )
            sys.exit()

        # Load and process the model data
        # for the U and V wind components
        model_datasets_u = load_data(
            dic.base_dir,
            shared_models,
            model_ws_variables[0],
            region,
            forecast_range,
            seasons_list_mod[i],
        )
        model_datasets_v = load_data(
            dic.base_dir,
            shared_models,
            model_ws_variables[1],
            region,
            forecast_range,
            seasons_list_mod[i],
        )

        # Process the model data
        model_data_u, model_time_u = process_data(
            model_datasets_u, model_ws_variables[0]
        )
        model_data_v, model_time_v = process_data(
            model_datasets_v, model_ws_variables[1]
        )

        # Use a try statement to catch any errors
        try:
            # Create a dictionary to store the model data
            model_data_ws = {}

            # Loop over the models and members
            for model in shared_models:
                # Extract the model data for the u and v wind components
                model_data_u_model = model_data_u[model]
                model_data_v_model = model_data_v[model]

                # Create a list to store the ensemble members
                # for wind speed
                model_data_ws[model] = []

                no_members_model = len(model_data_u_model)

                # Loop over the ensemble members for the model
                for i in range(no_members_model):

                    # Extract the u field for the ensemble member
                    u_field = model_data_u_model[i]
                    # Extract the v field for the ensemble member
                    v_field = model_data_v_model[i]

                    # Calculate the wind speed for the ensemble member
                    ws_field = np.sqrt(np.square(u_field) + np.square(v_field))

                    # Append the wind speed field to the list
                    model_data_ws[model].append(ws_field)
        except Exception as e:
            print(
                "Error when trying to calculate wind speeds from the model data xarrays: ",
                e,
            )
            sys.exit()

        # Define a test ws variable
        windspeed_var_name = "Wind"

        # Calculate the spatial correlations for the season
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = (
            calculate_field_stats(obs, model_data_ws, shared_models, windspeed_var_name)
        )

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
    plt.rcParams.update({"font.size": 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the variable
    variable = "850_Wind"

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # Set up the fgure size and subplot parameters
    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{variable} {region} {forecast_range} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.90)

    # Set up the significance thresholdf
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
        obs_lons_converted, lons_converted = (
            obs_lons_converted_list[i],
            lons_converted_list[i],
        )

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
        ax.plot(
            [azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1],
            [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )
        ax.plot(
            [iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1],
            [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap="RdBu_r", transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable == "tas":
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

        # Add stippling where rfield is significantly different from zero
        ax.contourf(lons, lats, pfield, hatches=["...."], alpha=0, transform=proj)

        # Add a textbox with the season name
        ax.text(
            0.05,
            0.95,
            season,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # # Add a textbox with the number of ensemble members in the bottom right corner
        # ax.text(0.95, 0.05, f"N = {ensemble_members_count_list[i]}", transform=ax.transAxes, fontsize=10, va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.5))

        # Add a textbox in the bottom right with the figure letter
        # extract the figure letter from the ax_labels list
        fig_letter = ax_labels[i]
        ax.text(
            0.95,
            0.05,
            fig_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # # Set up the text for the subplot
        # ax.text(-0.1, 1.1, key, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

        # Add the contourf object to the list
        cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(
        cf_list[0],
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=fig.axes,
        shrink=0.8,
    )
    cbar.set_label("correlation coefficients")

    # print("ax_labels shape", np.shape(ax_labels))
    # for i, ax in enumerate(axs):
    #     # Add the label to the bottom left corner of the subplot
    #     ax.text(0.05, 0.05, ax_labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')

    # Set up the path for saving the figure
    fig_name = f"{variable}_{region}_{forecast_range}_{experiment}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Define a new function to plot the time series for raw, lagged and NAO-matched
# data
def plot_seasonal_correlations_timeseries_methods(
    models,
    observations_path,
    obs_variable,
    obs_path,
    region,
    region_grid,
    forecast_range,
    start_year,
    end_year,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    time_series_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
):
    """
    Plots the time series for the correlations for the different models and seasons"
    """

    # Create an empty list to store the processed observations
    obs_list = []

    # Create empty lists to store the r field for the time series
    rfield_list = []
    pfield_list = []

    # List for the ensemble mean array
    ensemble_mean_array_list = []

    # List for the ensemble members count
    ensemble_members_count_list = []

    # Set up the ax labels
    ax_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    # Set up the methods
    methods = ["raw", "lagged", "nao_matched"]

    # set up the model load region
    model_load_region = "global"

    # Loop over the methods
    for method in methods:
        print("method", method)

        # Loop over the seasons
        for i, obs_season in enumerate(seasons_list_obs):
            print("obs season", obs_season)

            # Set up the model season
            model_season = seasons_list_mod[i]

            # Process the observations for the region and season
            region = "central-europe"  # hardcoded for now
            obs = process_observations_timeseries(
                obs_variable, region, forecast_range, obs_season, observations_path
            )

            # if the method is raw
            if method == "raw":
                print("method is raw")

                # Load the model data
                model_datasets = load_data(
                    dic.base_dir,
                    models,
                    obs_variable,
                    model_load_region,
                    forecast_range,
                    model_season,
                )

                # Process the model data
                model_data, _ = process_data(model_datasets, obs_variable)

                # Now use the function calculate_correlations_timeseries
                # to get the correlation time series for the seasons
                (
                    r,
                    p,
                    ensemble_mean_array,
                    observed_data_array,
                    ensemble_members_count,
                    obs_years,
                    model_years,
                ) = calculate_correlations_timeseries(
                    obs,
                    model_data,
                    models,
                    obs_variable,
                    obs_season,
                    model_season,
                    forecast_range,
                    method=method,
                )


# for the same variable, region and forecast range (e.g. psl global years 2-9)
# but with different seasons (e.g. DJFM, MAM, JJA, SON)
def plot_seasonal_correlations_timeseries(
    models,
    observations_path,
    variable,
    forecast_range,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    obs_var_name,
    north_sea_grid,
    central_europe_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
):
    """
    Plots the time series of correlations between the observed and model data for the given variable, region,
    forecast range, and seasons.

    Args:
        models (list): A list of model names to plot.
        observations_path (str): The path to the observations file.
        variable (str): The variable to plot.
        region (str): The region to plot.
        region_grid (list): The gridboxes that define the region.
        forecast_range (list): The forecast range to plot.
        seasons_list_obs (list): The seasons to plot for the observed data.
        seasons_list_mod (list): The seasons to plot for the model data.
        plots_dir (str): The directory to save the plots in.
        obs_var_name (str): The name of the variable in the observations file.
        north_sea_grid (list): The gridboxes that define the North Sea region.
        central_europe_grid (list): The gridboxes that define the Central Europe region.
        p_sig (float): The significance level for the correlation coefficient.
        experiment (str): The name of the experiment to plot.

    Returns:
        None.
    """

    # Create an empty list to store the processed observations
    # for each season
    obs_list = []

    # Create empty lists to store the r field
    # for each season
    r_north_sea_list = []
    r_central_europe_list = []

    # Store the p values
    p_north_sea_list = []
    p_central_europe_list = []

    # List for the ensemble mean array
    ensemble_mean_array_list = []

    # Create an empty list to store the ensemble members count
    ensemble_members_count_list = []

    # Create an empty list to store the obs years and model years
    obs_years_list = []
    model_years_list = []

    # Set up the labels for the subplots
    ax_labels = ["A", "B", "C", "D"]

    # Set up the model load region
    # will always be global
    model_load_region = "global"

    # Loop over the seasons
    for i, season in enumerate(seasons_list_obs):

        # Print the season(s) being processed
        print("obs season", season)

        # Set up the model season
        model_season = seasons_list_mod[i]
        print("model season", model_season)

        # If the season is DJFM or MAM
        # then we want to use the North Sea grid
        # If the variable is 'sfcWind'
        if variable == "sfcWind":
            print("variable is sfcWind")
            print("Selecting boxes according to the season of interest")
            if season in ["DJFM", "MAM"]:
                # Set up the region
                region = "north-sea"
            elif season in ["JJA", "SON"]:
                # Set up the region
                region = "central-europe"
            else:
                print("Error: season not found")
                sys.exit()
        else:
            print("variable is not sfcWind")
            print("Selecting a single box for all seasons")
            # Set up the region
            region = "central-europe"

        # Print the region
        print("region", region)

        # Process the observations
        # To get a 1D array of the observations
        # which is the gridbox average
        obs = process_observations_timeseries(
            variable, region, forecast_range, season, observations_path
        )

        # Print the shape of the observations
        print("obs shape", np.shape(obs))

        # Load the model data
        model_datasets = load_data(
            dic.base_dir,
            models,
            variable,
            model_load_region,
            forecast_range,
            model_season,
        )
        # Process the model data
        model_data, _ = process_data(model_datasets, variable)

        # Print the shape of the model data
        # this still has spatial dimensions
        print("model shape", np.shape(model_data))

        # now use the function calculate_correlations_timeseries
        # to get the correlation time series for the seasons
        (
            r,
            p,
            ensemble_mean_array,
            observed_data_array,
            ensemble_members_count,
            obs_years,
            model_years,
        ) = calculate_correlations_timeseries(obs, model_data, models, variable, region)

        # Verify thet the shape of the ensemble mean array is correct
        if np.shape(ensemble_mean_array) != np.shape(observed_data_array):
            print(
                "Error: ensemble mean array shape does not match observed data array shape"
            )
            sys.exit()

        if variable == "sfcWind":
            # Depending on the season, append the r to the correct list
            if season in ["DJFM", "MAM"]:
                r_north_sea_list.append(r)
                p_north_sea_list.append(p)
            elif season in ["JJA", "SON"]:
                r_central_europe_list.append(r)
                p_central_europe_list.append(p)
            else:
                print("Error: season not found")
                sys.exit()
        else:
            # Append the r to the central europe list
            r_central_europe_list.append(r)
            p_central_europe_list.append(p)

        # Append the ensemble members count to the list
        ensemble_members_count_list.append(ensemble_members_count)

        # Append the ensemble mean array to the list
        ensemble_mean_array_list.append(ensemble_mean_array)

        # Append the processed observations to the list
        obs_list.append(observed_data_array)

        # Append the obs years and model years to the lists
        obs_years_list.append(obs_years)
        model_years_list.append(model_years)

    # Set the font size for the plots
    plt.rcParams.update({"font.size": 12})

    # Set up the figure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 8),
        sharex=True,
        sharey="row",
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{variable} {region} {forecast_range} {experiment} correlation coefficients timeseries, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.95)

    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # Flatten the axs array
    axs = axs.flatten()

    # Iterate over the seasons
    for i, season in enumerate(seasons_list_obs):
        ax = axs[i]

        # Print the season being plotted
        print("plotting season", season)
        # Print the axis index
        print("axis index", i)

        # Print the values in the r and p lists
        # print("r_north_sea_list", r_north_sea_list)
        # print("p_north_sea_list", p_north_sea_list)

        # print("r_central_europe_list", r_central_europe_list)
        # print("p_central_europe_list", p_central_europe_list)

        if variable == "sfcWind":
            # Extract the r and p values
            # depending on the season
            if season in ["DJFM", "MAM"]:
                r = r_north_sea_list[i]
                p = p_north_sea_list[i]
            elif season in ["JJA", "SON"]:
                # run the index back by 2
                # so that the index matches the correct season
                i_season = i - 2
                r = r_central_europe_list[i_season]
                p = p_central_europe_list[i_season]
            else:
                print("Error: season not found")
                sys.exit()
        else:
            # Extract the r and p values
            r = r_central_europe_list[i]
            p = p_central_europe_list[i]

        # print the shape of the model years
        # print("model years shape", np.shape(model_years_list[i]))
        # print("model years", model_years_list[i])

        # # print the shape of the ensemble mean array
        # print("ensemble mean array shape", np.shape(ensemble_mean_array_list[i]))

        # # print the shape of the obs years
        # print("obs years shape", np.shape(obs_years_list[i]))
        # print("obs years", obs_years_list[i])

        # # print the shape of the obs
        # print("obs shape", np.shape(obs_list[i]))

        # if the variable is rsds
        # Divide the ERA5 monthly mean ssrd by 86400 to convert from J m^-2 to W m^-2
        if variable == "rsds":
            # Divide the obs by 86400
            obs_list[i] = obs_list[i] / 86400

        # Plot the ensemble mean
        ax.plot(
            model_years_list[i], ensemble_mean_array_list[i], color="red", label="dcppA"
        )

        # Plot the observed data
        ax.plot(obs_years_list[i], obs_list[i], color="black", label="ERA5")

        # Set up the plots
        # Add a horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        # ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
        if variable == "sfcWind":
            if i == 0 or i == 1:
                ax.set_ylim([-0.6, 0.6])
            elif i == 2 or i == 3:
                ax.set_ylim([-0.2, 0.2])
            # ax.set_xlabel("Year")
            if i == 0 or i == 2:
                ax.set_ylabel("sfcWind anomalies (m/s)")
        else:
            if i == 0 or i == 2:
                ax.set_ylabel("Irradiance anomalies (W m^-2)")

        # set the x-axis label for the bottom row
        if i == 2 or i == 3:
            ax.set_xlabel("year")

        # Set up a textbox with the season name in the top left corner
        ax.text(
            0.05,
            0.95,
            season,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Only if the variable is sfcWind
        if variable == "sfcWind":
            # Depending on the season, set up the region name
            # as a textbox in the top right corner
            if season in ["DJFM", "MAM"]:
                region_name = "North Sea"
            elif season in ["JJA", "SON"]:
                region_name = "Central Europe"
            else:
                print("Error: season not found")
                sys.exit()
        else:
            # Set up the region name as a textbox in the top right corner
            region_name = "Central Europe"

        # Add a textbox with the region name
        ax.text(
            0.95,
            0.95,
            region_name,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add a textbox in the bottom right with the figure letter
        # extract the figure letter from the ax_labels list
        fig_letter = ax_labels[i]
        ax.text(
            0.95,
            0.05,
            fig_letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Set up the p values
        # If less that 0.05, then set as text '< 0.05'
        # If less than 0.01, then set as text '< 0.01'
        if p < 0.01:
            p_text = "< 0.01"
        elif p < 0.05:
            p_text = "< 0.05"
        else:
            p_text = f"= {p:.2f}"

        # Extract the ensemble members count
        ensemble_members_count = ensemble_members_count_list[i]
        # Take the sum of the ensemble members count
        no_ensemble_members = sum(ensemble_members_count.values())

        # Set up the title for the subplot
        ax.set_title(
            f"ACC = {r:.2f}, p {p_text}, n = {no_ensemble_members}", fontsize=10
        )

    # Adjust the layout
    # plt.tight_layout()

    # Set up the path for saving the figure
    fig_name = f"{variable}_{region}_{forecast_range}_{experiment}_sig-{p_sig}_correlation_coefficients_timeseries_subplots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Define a new function to plot the NAO anomalies time series
# for the different seasons: DJFM, MAM, JJA, SON
# But using the pointwise definition of the summertime NAO index from Wang and Ting (2022)


def plot_seasonal_nao_anomalies_timeseries(
    models,
    observations_path,
    forecast_range,
    seasons_list_obs,
    seasons_list_mod,
    plots_dir,
    azores_grid,
    iceland_grid,
    snao_south_grid,
    snao_north_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
    variable="psl",
):
    """
    Plot the NAO anomalies time series for the different seasons: DJFM, MAM, JJA, SON,
    using the pointwise definition of the summertime NAO index from Wang and Ting (2022).

    Parameters
    ----------
    models : list of str
        List of model names to plot.
    observations_path : str
        Path to the observations file.
    forecast_range : str
        Forecast range to plot, in the format 'YYYY-MM'.
    seasons_list_obs : list of str
        List of seasons to plot for the observations.
    seasons_list_mod : list of str
        List of seasons to plot for the models.
    plots_dir : str
        Directory where the plots will be saved.
    azores_grid : dict
        Latitude and longitude coordinates of the Azores grid point.
    iceland_grid : dict
        Latitude and longitude coordinates of the Iceland grid point.
    snao_south_grid : dict
        Latitude and longitude coordinates of the southern SNAO grid point.
    snao_north_grid : dict
        Latitude and longitude coordinates of the northern SNAO grid point.
    p_sig : float, optional
        Significance level for the correlation coefficient, by default 0.05.
    experiment : str, optional
        Name of the experiment, by default 'dcppA-hindcast'.
    variable : str, optional
        Variable to plot, by default 'psl'.

    Returns
    -------
    None
    """

    # Create an empty list to store the processed obs NAO
    obs_nao_anoms_list = []

    # Create empty lists to store the r field and p field for the NAO
    # anomaly correlations
    r_list = []
    p_list = []

    # Create empty lists to store the ensemble mean array
    model_nao_anoms_list = []

    # Create an empty list to store the ensemble members count
    ensemble_members_count_list = []

    # Create an empty list to store the obs years and model years
    obs_years_list = []
    model_years_list = []

    # Set up the labels for the subplots
    ax_labels = ["A", "B", "C", "D"]

    # Set up the labels for the subplots
    ax_labels = ["A", "B", "C", "D"]

    # Set up the model load region
    # will always be global
    model_load_region = "global"

    # Loop over the seasons
    for i, season in enumerate(seasons_list_obs):

        # Print the season(s) being processed
        print("obs season", season)

        # Set up the model season
        model_season = seasons_list_mod[i]
        print("model season", model_season)

        # Process the observations
        # To get a 1D array of the NAO anomalies (azores - iceland)
        # Using the function process_obs_nao_anoms
        # the function call depends on the season
        if season in ["DJFM", "MAM", "SON"]:
            # Process the obs NAO anomalies
            obs_nao_anoms = process_obs_nao_index(
                forecast_range,
                season,
                observations_path,
                variable=variable,
                nao_type="default",
            )
        elif season in ["JJA"]:
            # Process the obs SNAO anomalies
            obs_nao_anoms = process_obs_nao_index(
                forecast_range,
                season,
                observations_path,
                variable=variable,
                nao_type="snao",
            )
        else:
            print("Error: season not found")
            sys.exit()

        # Print the shape of the observations
        print("obs shape", np.shape(obs_nao_anoms))

        # Load the model data
        model_datasets = load_data(
            dic.base_dir,
            models,
            variable,
            model_load_region,
            forecast_range,
            model_season,
        )
        # Process the model data
        model_data, _ = process_data(model_datasets, variable)

        # Print the shape of the model data
        # this still has spatial dimensions
        print("model shape", np.shape(model_data))

        # Now calculate the NAO anomalies for the model data
        # Using the function calculate_model_nao_anoms
        # the function call depends on the season
        if season in ["DJFM", "MAM", "SON"]:
            # Calculate the model NAO anomalies
            (
                ensemble_mean_nao_anoms,
                ensemble_members_nao_anoms,
                model_years,
                ensemble_members_count,
            ) = calculate_model_nao_anoms(
                model_data,
                models,
                azores_grid,
                iceland_grid,
                snao_south_grid,
                snao_north_grid,
                nao_type="default",
            )
        elif season in ["JJA"]:
            # Calculate the model SNAO anomalies
            (
                ensemble_mean_nao_anoms,
                ensemble_members_nao_anoms,
                model_years,
                ensemble_members_count,
            ) = calculate_model_nao_anoms(
                model_data,
                models,
                azores_grid,
                iceland_grid,
                snao_south_grid,
                snao_north_grid,
                nao_type="snao",
            )
        else:
            print("Error: season not found")
            sys.exit()

        # Now use the function calculate_nao_correlations
        # to get the correlations and p values for the NAO anomalies
        # for the different seasons
        r, p, ensemble_mean_nao_array, observed_nao_array, model_years, obs_years = (
            calculate_nao_correlations(obs_nao_anoms, ensemble_mean_nao_anoms, variable)
        )

        # Verify thet the shape of the ensemble mean array is correct
        if np.shape(ensemble_mean_nao_array) != np.shape(observed_nao_array):
            print(
                "Error: ensemble mean array shape does not match observed data array shape"
            )
            sys.exit()

        # Append the ensemble members count to the list
        ensemble_members_count_list.append(ensemble_members_count)

        # Append the ensemble mean array to the list
        model_nao_anoms_list.append(ensemble_mean_nao_array)

        # Append the processed observations to the list
        obs_nao_anoms_list.append(observed_nao_array)

        # Append the r and p values to the lists
        r_list.append(r)
        p_list.append(p)

        # Append the obs years and model years to the lists
        obs_years_list.append(obs_years)
        model_years_list.append(model_years)

    # Set the font size for the plots
    plt.rcParams.update({"font.size": 12})

    # Set up the figure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 8),
        sharex=True,
        sharey="row",
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{variable} {forecast_range} {experiment} NAO anomalies timeseries, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.95)

    # Set up the significance threshold
    # e.g. 0.05 for 95% significance
    sig_threshold = int((1 - p_sig) * 100)

    # Flatten the axs array
    axs = axs.flatten()

    # Iterate over the seasons
    for i, season in enumerate(seasons_list_obs):
        ax = axs[i]

        # Print the season being plotted
        print("plotting season", season)
        # Print the axis index
        print("axis index", i)

        # Print the values in the r and p lists
        print("r_list", r_list)
        print("p_list", p_list)

        # Extract the r and p values
        # depending on the season
        r = r_list[i]
        p = p_list[i]

        # print the shape of the model years
        # print("model years shape", np.shape(model_years_list[i]))
        # print("model years", model_years_list[i])

        # # print the shape of the ensemble mean array
        # print("ensemble mean array shape", np.shape(ensemble_mean_array_list[i]))

        # # print the shape of the obs years
        # print("obs years shape", np.shape(obs_years_list[i]))
        # print("obs years", obs_years_list[i])

        # # print the shape of the obs
        # print("obs shape", np.shape(obs_list[i]))

        # process the nao data
        model_nao_anoms = model_nao_anoms_list[i] / 100

        # Plot the ensemble mean
        ax.plot(model_years_list[i], model_nao_anoms, color="red", label="dcppA")

        # Plot the observed data
        obs_nao_anoms = obs_nao_anoms_list[i] / 100

        # Plot the observed data
        ax.plot(obs_years_list[i], obs_nao_anoms, color="black", label="ERA5")

        # Set up the plots
        # Add a horizontal line at zero
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        # ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
        ax.set_ylim([-10, 10])
        # ax.set_xlabel("Year")
        if i == 0 or i == 2:
            ax.set_ylabel("NAO anomalies (hPa)")

        # set the x-axis label for the bottom row
        if i == 2 or i == 3:
            ax.set_xlabel("year")

        # Set up a textbox with the season name in the top left corner
        ax.text(
            0.05,
            0.95,
            season,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add a textbox in the bottom right with the figure letter
        # extract the figure letter from the ax_labels list
        fig_letter = ax_labels[i]
        ax.text(
            0.95,
            0.05,
            fig_letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Depending on the season, set up the NAO name
        if season == "JJA":
            nao_name = "SNAO"
            # set this up in a textbox in the top right corner
            ax.text(
                0.95,
                0.95,
                nao_name,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
            )

        # Set up the p values
        # If less that 0.05, then set as text '< 0.05'
        # If less than 0.01, then set as text '< 0.01'
        if p < 0.01:
            p_text = "< 0.01"
        elif p < 0.05:
            p_text = "< 0.05"
        else:
            p_text = f"= {p:.2f}"

        # Extract the ensemble members count
        ensemble_members_count = ensemble_members_count_list[i]
        # Take the sum of the ensemble members count
        no_ensemble_members = sum(ensemble_members_count.values())

        # Set up the title for the subplot
        ax.set_title(
            f"ACC = {r:.2f}, p {p_text}, n = {no_ensemble_members}", fontsize=10
        )

    # Adjust the layout
    # plt.tight_layout()

    # Set up the path for saving the figure
    fig_name = f"{variable}_{forecast_range}_{experiment}_sig-{p_sig}_nao_anomalies_timeseries_subplots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# Now we want to write another function for creating subplots
# This one will plot for the same season, region, forecast range
# but for different variables (e.g. psl, tas, sfcWind, rsds)
def plot_variable_correlations(
    models_list,
    observations_path,
    variables_list,
    region,
    region_grid,
    forecast_range,
    season,
    plots_dir,
    obs_var_names,
    azores_grid,
    iceland_grid,
    p_sig=0.05,
    experiment="dcppA-hindcast",
):
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

    # Create empty lists to store the obs_lons_converted and lons_converted
    obs_lons_converted_list = []
    lons_converted_list = []

    # Add labels A, B, C, D to the subplots
    ax_labels = ["A", "B", "C", "D"]

    # Set up the list of model variables
    model_ws_variables = ["ua", "va"]

    # Set up the list of obs variables
    obs_ws_variables = ["var131", "var132"]

    # Loop over the variables
    for i in range(len(variables_list)):

        # Print the variable being processed
        print("processing variable", variables_list[i])

        # Extract the models for the variable
        models = models_list[i]

        # Set up the model season
        if season == "JJA":
            model_season = "ULG"
        elif season == "MAM":
            model_season = "MAY"
        else:
            model_season = season

        # If the variable is ua or va, then set up a different path for the observations
        if variables_list[i] in ["ua", "va"]:
            # Set up the observations path
            observations_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1694423850.2771118-29739-1-db661393-5c44-4603-87a8-2d7abee184d8.nc"
        elif variables_list[i] == "wind":
            # Print that the variable is Wind
            print("variable is Wind")
            print("Processing the 850 level wind speeds")

            # Hard code the observations path lol
            observations_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/adaptor.mars.internal-1694423850.2771118-29739-1-db661393-5c44-4603-87a8-2d7abee184d8.nc"

            # TODO: Set up processing of the obs and model data for the 850 level wind speeds here
            # Calculate the U and V components of the observed wind
            obs_u = process_observations(
                model_ws_variables[0],
                region,
                region_grid,
                forecast_range,
                season,
                observations_path,
                obs_ws_variables[0],
            )
            obs_v = process_observations(
                model_ws_variables[1],
                region,
                region_grid,
                forecast_range,
                season,
                observations_path,
                obs_ws_variables[1],
            )

            # Calculate the observed wind speed
            obs = np.sqrt(np.square(obs_u) + np.square(obs_v))

            # Append the processed observations to the list
            obs_list.append(obs)

            # Load and process the model data for both the U and V components
            model_datasets_u = load_data(
                dic.base_dir,
                models,
                model_ws_variables[0],
                region,
                forecast_range,
                model_season,
            )
            model_data_u, model_time_u = process_data(
                model_datasets_u, model_ws_variables[0]
            )

            # For the v component
            model_datasets_v = load_data(
                dic.base_dir,
                models,
                model_ws_variables[1],
                region,
                forecast_range,
                model_season,
            )
            model_data_v, model_time_v = process_data(
                model_datasets_v, model_ws_variables[1]
            )

            # Create a dictionary to store the model data
            model_data_ws = {}

            # Loop over the models
            for model in models:
                # Extract the U and V components
                model_data_u_model = model_data_u[model]
                model_data_v_model = model_data_v[model]

                # Create a list to store the ensemble members
                model_data_ws[model] = []

                # Set up the no. of ensemble members
                no_members_model = len(model_data_u_model)

                # Loop over the ensemble members
                for n in range(no_members_model):
                    # Extract the u and v components
                    u = model_data_u_model[n]
                    v = model_data_v_model[n]

                    # Calculate the wind speed
                    ws = np.sqrt(np.square(u) + np.square(v))

                    # Append the wind speed to the list
                    model_data_ws[model].append(ws)

            # Set up the model data
            model_data = model_data_ws
        else:
            print("variable is not wind or wind components")
            # Process the observations
            obs = process_observations(
                variables_list[i],
                region,
                region_grid,
                forecast_range,
                season,
                observations_path,
                obs_var_names[i],
            )

            # Load and process the model data
            model_datasets = load_data(
                dic.base_dir,
                models,
                variables_list[i],
                region,
                forecast_range,
                model_season,
            )
            # Process the model data
            model_data, model_time = process_data(model_datasets, variables_list[i])

        # Calculate the spatial correlations for the model
        rfield, pfield, obs_lons_converted, lons_converted, ensemble_members_count = (
            calculate_field_stats(obs, model_data, models, variables_list[i])
        )

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
    plt.rcParams.update({"font.size": 12})

    # Set the projection
    proj = ccrs.PlateCarree()

    # Set up the lats and lons for the azores grid
    azores_lon1, azores_lon2 = azores_grid["lon1"], azores_grid["lon2"]
    azores_lat1, azores_lat2 = azores_grid["lat1"], azores_grid["lat2"]

    # Set up the lats and lons for the iceland grid
    iceland_lon1, iceland_lon2 = iceland_grid["lon1"], iceland_grid["lon2"]
    iceland_lat1, iceland_lat2 = iceland_grid["lat1"], iceland_grid["lat2"]

    # subtract 180 from all of the azores and iceland lons
    azores_lon1, azores_lon2 = azores_lon1 - 180, azores_lon2 - 180
    iceland_lon1, iceland_lon2 = iceland_lon1 - 180, iceland_lon2 - 180

    # Set up the fgure size and subplot parameters
    # for a 2x2 grid of subplots
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        subplot_kw={"projection": proj},
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    # Set up the title for the figure
    title = f"{region} {forecast_range} {season} {experiment} correlation coefficients, p < {p_sig} ({int((1 - p_sig) * 100)}%)"

    # Set up the supertitle for the figure
    fig.suptitle(title, fontsize=12, y=0.90)

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

        # Extract the obs_lons_converted and lons_converted
        obs_lons_converted, lons_converted = (
            obs_lons_converted_list[i],
            lons_converted_list[i],
        )

        # Ensemble members count
        ensemble_members_count = ensemble_members_count_list[i]

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
        ax.plot(
            [azores_lon1, azores_lon2, azores_lon2, azores_lon1, azores_lon1],
            [azores_lat1, azores_lat1, azores_lat2, azores_lat2, azores_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )
        ax.plot(
            [iceland_lon1, iceland_lon2, iceland_lon2, iceland_lon1, iceland_lon1],
            [iceland_lat1, iceland_lat1, iceland_lat2, iceland_lat2, iceland_lat1],
            color="green",
            linewidth=2,
            transform=proj,
        )

        # Add filled contours
        # Contour levels
        clevs = np.arange(-1, 1.1, 0.1)
        # Contour levels for p-values
        clevs_p = np.arange(0, 1.1, 0.1)
        # Plot the filled contours
        cf = ax.contourf(lons, lats, rfield, clevs, cmap="RdBu_r", transform=proj)

        # If the variables is 'tas'
        # then we want to invert the stippling
        # so that stippling is plotted where there is no significant correlation
        if variable in ["tas", "tos"]:
            # replace values in pfield that are less than 0.05 with nan
            pfield[pfield < p_sig] = np.nan

            # Add stippling where rfield is significantly different from zero
            ax.contourf(lons, lats, pfield, hatches=["xxxx"], alpha=0, transform=proj)
        else:
            # replace values in pfield that are greater than 0.05 with nan
            pfield[pfield > p_sig] = np.nan

            # Add stippling where rfield is significantly different from zero
            ax.contourf(lons, lats, pfield, hatches=["...."], alpha=0, transform=proj)

        # Add a textbox with the variable name
        ax.text(
            0.05,
            0.95,
            variable,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Get the number of ensemble members
        # as the sum of the ensemble_members_count_list
        ensemble_members_count = sum(ensemble_members_count.values())

        # Add a textbox with the number of ensemble members in the bottom left corner
        ax.text(
            0.05,
            0.05,
            f"N = {ensemble_members_count}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add a textbox in the bottom right with the figure letter
        fig_letter = ax_labels[i]
        ax.text(
            0.95,
            0.05,
            fig_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add the contourf object to the list
        cf_list.append(cf)

    # Create a single colorbar for all of the subplots
    cbar = plt.colorbar(
        cf_list[0],
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=fig.axes,
        shrink=0.8,
    )
    cbar.set_label("correlation coefficients")

    # Set up the path for saving the figure
    fig_name = f"{region}_{forecast_range}_{season}_{experiment}_sig-{p_sig}_correlation_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()


# define a main function


def main():
    """Main function for the program.

    This function parses the arguments from the command line
    and then calls the functions to load and process the data.
    """

    # Create a usage statement for the script.
    USAGE_STATEMENT = (
        """python functions.py <variable> <model> <region> <forecast_range> <season>"""
    )

    # Check if the number of arguments is correct.
    if len(sys.argv) != 6:
        # print(f"Expected 6 arguments, but got {len(sys.argv)}")
        # print(USAGE_STATEMENT)
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
    # print("variable = ", args.variable)
    # print("model = ", args.model)
    # print("region = ", args.region)
    # print("forecast range = ", args.forecast_range)
    # print("season = ", args.season)

    # If the model specified == "all", then run the script for all models.
    if args.model == "all":
        args.model = dic.models

    # If the type of the model argument is a string, then convert it to a list.
    if type(args.model) == str:
        args.model = [args.model]

    # Load the data.
    datasets = load_data(
        dic.base_dir,
        args.model,
        args.variable,
        args.region,
        args.forecast_range,
        args.season,
    )

    # Process the model data.
    variable_data, model_time = process_data(datasets, args.variable)

    # Choose the obs path based on the variable
    obs_path = choose_obs_path(args)

    # choose the obs var name based on the variable
    obs_var_name = choose_obs_var_name(args)

    # Process the observations.
    obs = process_observations(
        args.variable,
        args.region,
        dic.north_atlantic_grid,
        args.forecast_range,
        args.season,
        obs_path,
        obs_var_name,
    )

    # Call the function to calculate the ACC
    rfield, pfield, obs_lons_converted, lons_converted = calculate_field_stats(
        obs, variable_data, args.model
    )

    # Call the function to plot the ACC
    plot_correlations(
        args.model,
        rfield,
        pfield,
        obs,
        args.variable,
        args.region,
        args.season,
        args.forecast_range,
        dic.plots_dir,
        obs_lons_converted,
        lons_converted,
        dic.azores_grid,
        dic.iceland_grid,
    )


# Call the main function.
if __name__ == "__main__":
    main()
