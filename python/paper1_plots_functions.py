"""
Functions for use in paper1_plots.ipynb notesbook.
"""

# Local Imports
import os
import sys

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from scipy import signal
from datetime import datetime
import pandas as pd
import random

# import tqdm
from tqdm import tqdm

# Import typing
from typing import Tuple

# Import xarray
import xarray as xr

# Local imports
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts

# # Import functions from skill-maps
sys.path.append("/home/users/benhutch/skill-maps/python")
# import functions as fnc
import plotting_functions as plt_fnc
import nao_alt_lag_functions as nal_fnc

# Import nao skill functions
import nao_skill_functions as nao_fnc

# Import functions
import functions as fnc

# Import functions from plot_init_benefit
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
import plot_init_benefit as pib_fnc

# Import the nao_matching_seasons functions
import nao_matching_seasons as nao_match_fnc

# Import the historical functions
sys.path.append("/home/users/benhutch/skill-maps-differences")
import functions_diff as hist_fnc


# Define a function to convert the season
def convert_season(season, dic):
    """
    Convert a season from a number to a string using a dictionary.

    Args:
        season (str): The season to convert.
        dic (dict): A dictionary mapping season numbers to season names.
                    Must contain 'season_map' as a key.

    Returns:
        str: The converted season name.

    Example:
        >>> season = "1"
        >>> dic = {"1": "DJF", "2": "MAM", "3": "JJA", "4": "SON"}
        >>> convert_season(season, dic)
        'DJF'
    """
    # If season contains a number, convert it to the string
    if season in ["1", "2", "3", "4"]:
        season = dic.season_map[season]

    return season


# Define a function to extract historical models based on the variable
def extract_hist_models(variable, dic):
    """
    For a given variable, extract the historical models.

    Args:
        variable (str): The variable to extract the historical models for.
        dic (dict): A dictionary containing the historical models
                    for each variable.
                    Must contain 'historical_models_map' as a key.

    Returns:
        list: A list of the historical models for the given variable.
    """

    # Extract the historical models for the given variable
    hist_models = dic.historical_models_map[variable]

    # Return the historical models
    return hist_models


# Define a new function to load and process the historical data
def load_and_process_hist_data(
    base_dir, hist_models, variable, region, forecast_range, season
):
    """
    Load and process the historical data for a given variable, region,
    forecast range and season. Assumes surface data.

    Args:
        base_dir (str): The base directory containing the historical data.
        hist_models (list): A list of the historical models to load the data for.
        variable (str): The variable to load the data for.
        region (str): The region to load the data for.
        forecast_range (str): The forecast range to load the data for.
        season (str): The season to load the data for.

    Returns:
        hist_data (dict): The processed historical data.
                          As a dictionary containing the model names as keys.
    """

    hist_datasets = hist_fnc.load_processed_historical_data(
        base_dir, hist_models, variable, region, forecast_range, season
    )

    hist_data, _ = hist_fnc.process_data(hist_datasets, variable)

    return hist_data


# Define a new function to load and process the model data
def load_and_process_dcpp_data(
    base_dir, dcpp_models, variable, region, forecast_range, season
):
    """
    Load and process the model data for a given variable, region,
    forecast range and season. Assumes surface data.

    Args:
        base_dir (str): The base directory containing the dcpp data.
        dcpp_models (list): A list of the dcpp models to load the data for.
        variable (str): The variable to load the data for.
        region (str): The region to load the data for.
        forecast_range (str): The forecast range to load the data for.
        season (str): The season to load the data for.

    Returns:
        dcpp_data (dict): The processed dcpp data.
                          As a dictionary containing the model names as keys.
    """

    dcpp_datasets = fnc.load_data(
        base_dir, dcpp_models, variable, region, forecast_range, season
    )

    dcpp_data, _ = fnc.process_data(dcpp_datasets, variable)

    return dcpp_data


# Define a new function to align the time periods and convert to array
def align_and_convert_to_array(hist_data, dcpp_data, hist_models, dcpp_models, obs):
    """
    Align the time periods and convert the data to an array.

    Args:
        hist_data (dict): The processed historical data.
                          As a dictionary containing the model names as keys.
        dcpp_data (dict): The processed dcpp data.
                          As a dictionary containing the model names as keys.
        hist_models (list): A list of the historical models to load data for.
        dcpp_models (list): A list of the dcpp models to load the data for.
        variable (str): The variable to load the data for.
        obs (array): The processed observations.

    Returns:
        fcst1 (array): The processed dcpp data as an array.
        fcst2 (array): The processed historical data as an array.
        obs (array): The processed observations as an array.
        common_years (array): The common years between all three datasets.
    """

    if hist_data is not None:
        # Use constrain_years to make sure that the models have the same time axis
        constrained_hist_data = fnc.constrain_years(hist_data, hist_models)

        constrained_dcpp_data = fnc.constrain_years(dcpp_data, dcpp_models)

        # Align the forecasts and observations
        fcst1, fcst2, obs, common_years = fnc.align_forecast1_forecast2_obs(
            constrained_dcpp_data, dcpp_models, constrained_hist_data, hist_models, obs
        )
    else:
        # Constraint the dcpp data
        constrained_dcpp_data = fnc.constrain_years(dcpp_data, dcpp_models)

        # Align the forecasts and observations
        fcst1, fcst2, obs, common_years = fnc.align_forecast1_forecast2_obs(
            constrained_dcpp_data, dcpp_models, None, None, obs
        )

    # Return the aligned data
    return fcst1, fcst2, obs, common_years


# Define a new function to load the NAO matched data
def load_nao_matched_data(
    base_dir: str,
    variable: str,
    region: str,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load the NAO matched members and mean from a directory.

    Parameters
    ----------
    base_dir : str
        The base directory containing the NAO matched data.
    variable : str
        The variable to load (e.g. "t2m").
    region : str
        The region to load (e.g. "uk").
    season : str
        The season to load (e.g. "djf").
    forecast_range : str
        The forecast range to load (e.g. "day10").
    start_year : int
        The start year of the data to load.
    end_year : int
        The end year of the data to load.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        A tuple containing the NAO matched members and mean as xarray Datasets.
    """

    # Convert the season to the model season
    if season == "MAM":
        season = "MAY"
    elif season == "JJA":
        season = "ULG"

    # Set up the path to the data
    path_nao_match_dir = f"{base_dir}/{variable}/{region}/{season}/{forecast_range}/{start_year}-{end_year}/"

    # Check if there are files in the directory
    if len(os.listdir(path_nao_match_dir)) == 0:
        raise ValueError("There are no files in the directory")

    # Extract the files in the directory
    files = os.listdir(path_nao_match_dir)

    # Extract the file containing the lagged ensemble NAO matched mean
    nao_matched_mean_file = [file for file in files if "mean_lagged" in file][0]

    # Extract the file containing the lagged ensemble NAO matched members
    nao_matched_members_file = [file for file in files if "members_lagged" in file][0]

    # Open the datasets using xarray
    nao_matched_members = xr.open_dataset(path_nao_match_dir + nao_matched_members_file)
    nao_matched_mean = xr.open_dataset(path_nao_match_dir + nao_matched_mean_file)

    return (nao_matched_members, nao_matched_mean)


# Define a function to align the NAO matched data with the constrained hist
# data and the obs
def align_nao_matched_members(
    obs: xr.DataArray,
    nao_matched_members: xr.Dataset,
    constrained_hist_data: dict,
    hist_models: list,
) -> tuple:
    """
    Aligns the NAO matched members, observations, and constrained historical data
    to have the same years.

    Args:
    obs (xr.DataArray): The observations as an xarray DataArray.
    nao_matched_members (xr.Dataset): The NAO matched members as an xarray Dataset.
    constrained_hist_data (dict): The constrained historical data as a dictionary
                                    of xarray Datasets.
    hist_models (list): A list of the historical models.

    Returns:
    tuple: A tuple containing the aligned NAO matched members, forecast2,
            observations, and common years.
            Contains the following:
            fcst1_nm (array): The aligned NAO matched members.
            fcst2 (array): The aligned constrained historical data.
            obs (array): The aligned observations.
            common_years (array): The common years between all three datasets.
    """

    # First extract the years for the observations
    obs_years = obs.time.dt.year.values

    # Loop over the years
    for year in obs_years:
        # If there are any NaN values in the observations
        obs_year = obs.sel(time=f"{year}")
        if np.isnan(obs_year.values).any():
            print(f"there are NaN values in the observations for {year}")
            if np.isnan(obs_year.values).all():
                print(f"all values are NaN for {year}")
                # Delete the year from the observations
                obs = obs.sel(time=obs.time.dt.year != year)
            else:
                print(f"not all values are NaN for {year}")
        else:
            print(f"there are no NaN values in the observations for {year}")

    # Extract the constrained obs_years
    obs_years = obs.time.dt.year.values

    # Extract the years for the NAO matched members
    nao_matched_members_years = nao_matched_members.time.values

    # Check that there are no duplicate years in the NAO matched members
    if len(nao_matched_members_years) != len(np.unique(nao_matched_members_years)):
        raise ValueError("there are duplicate years in the NAO matched members")

    # Loop over the years for the NAO matched members
    # and check that there are no NaN values
    for year in nao_matched_members_years:
        # If there are any NaN values in the observations
        nao_matched_year = nao_matched_members["__xarray_dataarray_variable__"].sel(
            time=year
        )
        if np.isnan(nao_matched_year.values).any():
            print(f"there are NaN values in the NAO matched members for {year}")
            if np.isnan(nao_matched_year.values).all():
                print(f"all values are NaN for {year}")
                # Delete the year from the observations
                nao_matched_members = nao_matched_members.sel(
                    time=nao_matched_members.time.values != year
                )
            else:
                print(f"not all values are NaN for {year}")
                # print("deleting the year containing some NaNs from the NAO matched members")
                # # Delete the year from the NAo matched members
                # nao_matched_members = nao_matched_members.sel(time=nao_matched_members.time.values != year)
        else:
            print(f"there are no NaN values in the NAO matched members for {year}")

    # Extract the constrained nao_matched_members_years
    nao_matched_members_years = nao_matched_members.time.values

    # Extract the years for the constrained historical data
    constrained_hist_data_years = constrained_hist_data[hist_models[0]][
        0
    ].time.dt.year.values

    # If the years for the NAO matched members are not the same as the constrained historical data
    if not np.array_equal(nao_matched_members_years, constrained_hist_data_years):
        print(
            "years for NAO matched members and constrained historical data are not the same"
        )

        # Find the common years
        common_years = np.intersect1d(
            nao_matched_members_years, constrained_hist_data_years
        )

        # Extract the common years from the NAO matched members
        common_years_mask = np.in1d(nao_matched_members.time.values, common_years)
        fcst1_nm = nao_matched_members.isel(time=common_years_mask)

        # Extract the common years from the constrained historical data
        constrained_hist_data_nmatch = {}
        for model in constrained_hist_data:
            model_data = constrained_hist_data[model]
            for member in model_data:
                # Extract the years for the member
                member_years = member.time.dt.year.values

                # If the years for the member are not the same as the common years
                if not np.array_equal(member_years, common_years):
                    # print(f"years for {model} member {member} are not the same as the common years")
                    # Extract the common years for the member
                    member = member.sel(time=member.time.dt.year.isin(common_years))

                # Append to the list
                if model not in constrained_hist_data_nmatch:
                    constrained_hist_data_nmatch[model] = []

                constrained_hist_data_nmatch[model].append(member)

        # FIXME: Check the aligning of years here
        fcst1_nm_years = fcst1_nm.time.values

        # Check the new years
        fcst2_years = constrained_hist_data_nmatch[hist_models[0]][
            0
        ].time.dt.year.values

        # Assert that the NAO matched members and constrained historical data
        # have the same years
        assert np.array_equal(fcst2_years, fcst1_nm_years), "the years are not the same"

        # NAO matched members years
        nao_matched_members_years = fcst1_nm_years

    # Forecast 1 NM and forecast 2 should now be the same
    common_model_years = nao_matched_members_years

    # If the obs years are not the same as the common model years
    if not np.array_equal(obs_years, common_model_years):
        print("years for observations and common model years are not the same")

        # Find the common years
        common_years = np.intersect1d(obs_years, common_model_years)

        # Select the common years from the observations
        obs = obs.sel(time=obs.time.dt.year.isin(common_years))

        # Use a boolean mask to select the common years from the NAO matched members
        common_years_mask = np.in1d(nao_matched_members.time.values, common_years)

        # Extract the common years from the NAO matched members
        fcst1_nm = nao_matched_members.isel(time=common_years_mask)

        # Extract the common years from the constrained historical data
        constrained_hist_data_nmatch_obs = {}

        # Extract the common obs years from the constrained historical data
        for model in constrained_hist_data:
            model_data = constrained_hist_data[model]
            for member in model_data:
                # Extract the years for the member
                member_years = member.time.dt.year.values

                # If the years for the member are not the same as the common years
                if not np.array_equal(member_years, common_years):
                    # print(f"years for {model} member {member} are not the same as the common years")
                    # Extract the common years for the member
                    member = member.sel(time=member.time.dt.year.isin(common_years))

                # Append to the list
                if model not in constrained_hist_data_nmatch_obs:
                    constrained_hist_data_nmatch_obs[model] = []

                constrained_hist_data_nmatch_obs[model].append(member)

        # Check the new years
        obs_years = obs.time.dt.year.values

        # Extract the years for the constrained historical data
        constrained_hist_data_years = constrained_hist_data_nmatch_obs[hist_models[0]][
            0
        ].time.dt.year.values

        # Assert that the arrays are the same
        assert np.array_equal(
            obs_years, constrained_hist_data_years
        ), "the years are not the same"

        # Extract the nao matched members years again
        nao_matched_members_years = fcst1_nm.time.values

        # Assert that the arrays are the same
        assert np.array_equal(
            obs_years, nao_matched_members_years
        ), "the years are not the same"

    # Extract the arrays from the datasets
    fcst1_nm = fcst1_nm["__xarray_dataarray_variable__"].values

    # Extract the obs
    obs = obs.values

    # Extract the no. ensemble members for f2
    n_members_hist = np.sum(
        [
            len(constrained_hist_data_nmatch[model])
            for model in constrained_hist_data_nmatch
        ]
    )

    # Set up the fcst2 array
    fcst2 = np.zeros(
        [n_members_hist, len(obs_years), fcst1_nm.shape[2], fcst1_nm.shape[3]]
    )

    # Initialize the member index counter
    member_index = 0

    # Loop over the models
    for model in constrained_hist_data_nmatch:
        # Extract the model data
        model_data = constrained_hist_data_nmatch[model]

        # Loop over the members
        for member in model_data:
            # Extract the member data
            member_data = member.values

            # Set up the member data
            fcst2[member_index, :, :, :] = member_data

            # Increment the member index
            member_index += 1

    # If the time axis is not the second axis of the nao matched members
    if fcst1_nm.shape[1] != len(obs_years):
        print("time axis is not the second axis of the nao matched members")

        # Swap the first and second axes
        fcst1_nm = np.swapaxes(fcst1_nm, 0, 1)

    # Assert that the array shapes are the same
    assert (
        fcst1_nm[0].shape == fcst2[0].shape
    ), "the forecast array shapes are not the same"

    # Assert that the array shapes are the same
    assert (
        fcst1_nm[0].shape == obs.shape
    ), "the forecast and obs array shapes are not the same"

    common_years = obs_years

    # Print the shapes
    print("fcst1_nm shape:", fcst1_nm.shape)
    print("fcst2 shape:", fcst2.shape)
    print("obs shape:", obs.shape)

    return (fcst1_nm, fcst2, obs, common_years)


# Create a function to process the raw data for the full forecast period
# TODO: may also need to do this for lagged at some point as well
def forecast_stats_var(
    variables: list,
    season: str,
    forecast_range: str,
    region: str = "global",
    start_year: int = 1961,
    end_year: int = 2023,
    method: str = "raw",
    no_bootstraps: int = 1,
    base_dir: str = "/home/users/benhutch/skill-maps-processed-data",
):
    """
    Wrapper function which processes and creates a forecast_stats dictionary
    object for each variable in the variables list.

    Inputs:
    -------

    variables: list
        List of variables to process.
        e.g. ["tas", "pr", "psl"]

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    region: str
        Region to process.
        e.g. default is "global"

    start_year: int
        Start year to process.
        e.g. default is 1961

    end_year: int
        End year to process.
        e.g. default is 2023

    method: str
        Method to process.
        e.g. default is "raw"

    no_bootstraps: int
        Number of bootstraps to process.
        e.g. default is 1

    base_dir: str
        Base directory to process.
        e.g. default is "/home/users/benhutch/skill-maps-processed-data"

    Outputs:
    --------
    forecast_stats_var: dict
        Dictionary containing forecast statistics for each variable.
    """

    # Create empty dictionary containing a key for each variable
    forecast_stats_var = {}

    # Translate the seasons
    if season == "DJF":
        model_season = "DJF"
    if season == "DJFM":
        model_season = "DJFM"
    elif season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    elif season == "SON":
        model_season = "SON"
    else:
        raise ValueError(f"season {season} not recognised!")

    # Loop over each variable in the variables list
    for variable in variables:
        # Do some logging
        print(f"Processing {variable}...")

        # Assign the obs variable name
        obs_var_name = dicts.var_name_map[variable]

        # Print the obs variable name
        print(f"obs_var_name = {obs_var_name}")

        # Set up the dcpp models for the variable
        dcpp_models = nao_match_fnc.match_variable_models(match_var=variable)

        # Set up the obs path for the variable
        obs_path = nao_match_fnc.find_obs_path(match_var=variable)

        # Process the observations for this variable
        # Prrocess the observations
        obs = fnc.process_observations(
            variable=variable,
            region=region,
            region_grid=dicts.gridspec_global,
            forecast_range=forecast_range,
            season=season,
            observations_path=obs_path,
            obs_var_name=variable,
        )

        # Load and process the dcpp model data
        dcpp_data = load_and_process_dcpp_data(
            base_dir=base_dir,
            dcpp_models=dcpp_models,
            variable=variable,
            region=region,
            forecast_range=forecast_range,
            season=model_season,
        )

        # Make sure that the individual models have the same valid years
        dcpp_data = fnc.constrain_years(model_data=dcpp_data, models=dcpp_models)

        # Align the obs and dcpp data
        obs, dcpp_data, _ = fnc.remove_years_with_nans_nao(
            observed_data=obs, model_data=dcpp_data, models=dcpp_models
        )

        # Extract the years
        years = obs.time.dt.year.values

        # print the first and last years from the obs
        print(f"obs start year = {years[0]}")
        print(f"obs end year = {years[-1]}")

        # Extract an array from the obs
        obs_array = obs.values

        # Extract the dims from the obs data
        nyears = obs_array.shape[0]
        lats = obs_array.shape[1]
        lons = obs_array.shape[2]

        # Calculate the number of dcpp ensemble members
        dcpp_ensemble_members = np.sum([len(dcpp_data[model]) for model in dcpp_models])

        # Create an empty array to store the dcpp data
        dcpp_array = np.zeros([dcpp_ensemble_members, nyears, lats, lons])

        # Set up the member counter
        member_index = 0

        # Loop over the models
        for model in dcpp_models:
            dcpp_model_data = dcpp_data[model]

            # Loop over the ensemble members
            for member in dcpp_model_data:
                # Increment the member index
                member_index += 1

                # Extract the data
                data = member.values

                # If the data has four dimensions
                if len(data.shape) == 4:
                    # Squeeze the data
                    data = np.squeeze(data)

                # Assign the data to the forecast1 array
                dcpp_array[member_index - 1, :, :, :] = data

        # Assert that obs and dcpp_array have the same shape
        assert (
            obs_array.shape == dcpp_array[0, :, :, :].shape
        ), "obs and dcpp_array have different shapes!"

        # Create an empty dictionary to store the forecast stats
        forecast_stats_var[variable] = {}

        # Calculate the forecast stats for the variable
        forecast_stats_var[variable] = fnc.forecast_stats(
            obs=obs_array,
            forecast1=dcpp_array,
            forecast2=dcpp_array,  # use the same here as a placeholder for historical
            no_boot=no_bootstraps,
        )

        # Do some logging
        print(f"Finished processing {variable}!")

    # If psl is in the variables list
    if "psl" in variables:
        print("Calculating the NAO index...")

        # Set up the dcpp models for the variable
        dcpp_models = nao_match_fnc.match_variable_models(match_var="psl")

        # Set up the obs path for the variable
        obs_path = nao_match_fnc.find_obs_path(match_var="psl")

        # Process the observations for the NAO index
        obs_psl_anom = fnc.read_obs(
            variable="psl",
            region=region,
            forecast_range=forecast_range,
            season=season,
            observations_path=obs_path,
            start_year=1960,
            end_year=2023,
        )

        # Load adn process the dcpp model data
        dcpp_data = load_and_process_dcpp_data(
            base_dir=base_dir,
            dcpp_models=dcpp_models,
            variable="psl",
            region=region,
            forecast_range=forecast_range,
            season=model_season,
        )

        # Remove the years with NaNs from the obs and dcpp data
        obs_psl_anom, dcpp_data, _ = fnc.remove_years_with_nans_nao(
            observed_data=obs_psl_anom,
            model_data=dcpp_data,
            models=dcpp_models,
            NAO_matched=False,
        )

        # Extract the nao stats
        nao_stats_dict = nao_fnc.nao_stats(
            obs_psl=obs_psl_anom,
            hindcast_psl=dcpp_data,
            models_list=dcpp_models,
            lag=4,
            short_period=(1965, 2010),
            season=season,
        )

    else:
        print("Not calculating the NAO index...")
        nao_stats_dict = None

    # Return the forecast_stats_var dictionary
    return forecast_stats_var, nao_stats_dict


# Define a plotting function for this data
def plot_forecast_stats_var(
    forecast_stats_var_dic: dict,
    nao_stats_dict: dict,
    psl_models: list,
    season: str,
    forecast_range: str,
    figsize_x: int = 10,
    figsize_y: int = 12,
    gridbox_corr: dict = None,
    gridbox_plot: dict = None,
    sig_threshold: float = 0.05,
):
    """
    Plots the correlation fields for each variable in the forecast_stats_var.

    Inputs:
    -------

    forecast_stats_var_dic: dict
        Dictionary containing forecast statistics for each variable.
        Dictionary keys are the variable names.
        e.g. forecast_stats_var["tas"] = {"corr": corr, "rmse": rmse, "bias": bias}

    nao_stats_dict: dict
        Dictionary containing the NAO stats.
        e.g. nao_stats_dict = {"corr": corr, "rmse": rmse, "bias": bias}

    psl_models: list
        List of models which are used to calculate the NAO index.

    season: str
        Season to process.

    forecast_range: str
        Forecast range to process.

    figsize_x: int
        Figure size in x direction.
        e.g. default is 10

    figsize_y: int
        Figure size in y direction.
        e.g. default is 12

    gridbox_corr: dict
        Dictionary containing the gridbox which is used to calculate the correlation.
        e.g. gridbox_corr = {"lat": lat, "lon": lon, "corr": corr}

    gridbox_plot: dict
        Dictionary containing the gridbox which is used to constrain
        the domain of the plot.
        e.g. gridbox_plot = {"lat": lat, "lon": lon, "corr": corr}

    sig_threshold: float
        Significance threshold for the correlation.
        e.g. default is 0.05

    Outputs:
    --------

    None

    """

    # First do the processing for the NAO index
    if forecast_range == "2-9" and season == "DJFM":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1966, 2019 + 1))
        nyears_long_lag = len(np.arange(1969, 2019 + 1))
    elif forecast_range == "2-9" and season == "JJA":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1967, 2019 + 1))
        nyears_long_lag = len(np.arange(1970, 2019 + 1))
    elif forecast_range == "2-5":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1964, 2017 + 1))
        nyears_long_lag = len(np.arange(1967, 2017 + 1))
    elif forecast_range == "2-3":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1963, 2016 + 1))
        nyears_long_lag = len(np.arange(1966, 2016 + 1))
    else:
        raise ValueError("forecast_range must be either 2-9 or 2-5 or 2-3")

    # Set up the forecast ylims
    if season == "DJFM":
        ylims = [-10, 10]
    elif season == "JJA":
        ylims = [-3, 3]
    else:
        raise ValueError(f"season {season} not recognised!")

    # Set up the arrays for plotting the NAO index
    total_nens = 0
    total_lagged_nens = 0

    # Loop over the models
    for model in psl_models:
        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Add the number of ensemble members to the total
        total_nens += nao_stats["nens"]

        # And for the lagged ensemble
        total_lagged_nens += nao_stats["nens_lag"]

    # Set up the arrays for the NAO index
    nao_members = np.zeros([total_nens, nyears_long])

    # Set up the lagged arrays for the NAO index
    nao_members_lag = np.zeros([total_lagged_nens, nyears_long_lag])

    # Set up the counter for the current index
    current_index = 0
    current_index_lag = 0

    # Iterate over the models
    for i, model in enumerate(psl_models):
        print(f"Extracting ensemble members for model {model}...")

        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Loop over the ensemble members
        for j in range(nao_stats["nens"]):
            print(f"Extracting member {j} from model {model}...")

            # Extract the nao member
            nao_member = nao_stats["model_nao_ts_members"][j, :]

            # Set up the lnegth of the correct time series
            nyears_bcc = len(nao_stats_dict["BCC-CSM2-MR"]["years"])

            # If the model is not BCC-CSM2-MR
            if model != "BCC-CSM2-MR":
                # Extract the len of the time series
                nyears = len(nao_stats["years"][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats["years"])

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print(
                    "The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                        model
                    )
                )

                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears

                # Assert that the new len is correct
                assert (
                    len(nao_member[1:skip_years]) == nyears_bcc
                ), "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members[current_index, :] = nao_member[1:skip_years]
                else:
                    nao_members[current_index, :] = nao_member[1:]
            else:
                # Append this member to the array
                nao_members[current_index, :] = nao_member

            # Increment the counter
            current_index += 1

        # Loop over the lagged ensemble members
        for j in range(nao_stats["nens_lag"]):

            # Extract the NAO index for this member
            nao_member = nao_stats["model_nao_ts_lag_members"][j, :]
            print("NAO index extracted for member {}".format(j))

            # Set up the length of the correct time series
            nyears_bcc = len(nao_stats_dict["BCC-CSM2-MR"]["years_lag"])

            if model != "BCC-CSM2-MR":
                # Extract the length of the time series for this model
                nyears = len(nao_stats["years_lag"][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats["years_lag"])

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print(
                    "The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                        model
                    )
                )

                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears

                # Assert that the new len is correct
                assert (
                    len(nao_member[1:skip_years]) == nyears_bcc
                ), "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members_lag[current_index_lag, :] = nao_member[1:skip_years]
                else:
                    nao_members_lag[current_index_lag, :] = nao_member[1:]
            else:
                # Append this member to the array
                nao_members_lag[current_index_lag, :] = nao_member

            # Increment the counter
            current_index_lag += 1

    # Set up the initialisation offset
    if forecast_range == "2-9":
        init_offset = 5
    elif forecast_range == "2-5":
        init_offset = 2
    elif forecast_range == "2-3":
        init_offset = 1
    else:
        raise ValueError("forecast_range must be either 2-9 or 2-5")

    # Set up the axis labels
    axis_labels = ["a", "b", "c", "d", "e", "f"]

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Count the number of keys in the forecast_stats_var_dic
    no_keys = len(forecast_stats_var_dic.keys())

    # Set up the nrows depending on whether the number of keys is even or odd
    if no_keys % 2 == 0:
        nrows = int(no_keys / 2) + 1  # Extra row for the NAO index
    else:
        nrows = int((no_keys + 1) / 2) + 1

    # Set up the figure
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(figsize_x, figsize_y))

    # Update the params for mathtext default rcParams
    plt.rcParams.update({"mathtext.default": "regular"})

    # Set up the sup title
    sup_title = "Total correlation skill (r) for each variable"

    # Set up the sup title
    fig.suptitle(sup_title, fontsize=6, y=0.93)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # If the gridbox_corr is not None
    if gridbox_corr is not None:
        # Extract the lats and lons from the gridbox_corr
        lon1_corr, lon2_corr = gridbox_corr["lon1"], gridbox_corr["lon2"]
        lat1_corr, lat2_corr = gridbox_corr["lat1"], gridbox_corr["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
        lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
        lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))

        # Constrain the lats and lons
        # lats_corr = lats[lat1_idx_corr:lat2_idx_corr]
        # lons_corr = lons[lon1_idx_corr:lon2_idx_corr]

    # Set up the extent
    # Using the gridbox plot here as this is the extent of the plot
    if gridbox_plot is not None:
        lon1_gb, lon2_gb = gridbox_plot["lon1"], gridbox_plot["lon2"]
        lat1_gb, lat2_gb = gridbox_plot["lat1"], gridbox_plot["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_gb = np.argmin(np.abs(lats - lat1_gb))
        lat2_idx_gb = np.argmin(np.abs(lats - lat2_gb))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_gb = np.argmin(np.abs(lons - lon1_gb))
        lon2_idx_gb = np.argmin(np.abs(lons - lon2_gb))

        # Constrain the lats and lons
        lats = lats[lat1_idx_gb:lat2_idx_gb]
        lons = lons[lon1_idx_gb:lon2_idx_gb]

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up a list to store the contourf objects
    cf_list = []

    # Set up the axes
    axes = []

    # Loop over the keys and forecast_stats_var_dic
    for i, (key, forecast_stats) in enumerate(forecast_stats_var_dic.items()):
        # Logging
        print(f"Plotting variable {key}...")
        print(f"Plotting index {i}...")

        # print the keys for forecast stats
        print(f"forecast_stats.keys() = {forecast_stats.keys()}")
        sys.exit()

        # Extract the correlation arrays from the forecast_stats dictionary
        corr = forecast_stats["corr1"]
        corr1_p = forecast_stats["corr1_p"]

        # Extract the time series
        fcst1_ts = forecast_stats["f1_ts"]
        obs_ts = forecast_stats["o_ts"]

        # Extract the values
        nens1 = forecast_stats["nens1"]
        start_year = 1966 - 5
        end_year = 2019 - 5

        # Print the start and end years
        print(f"start_year = {start_year}")
        print(f"end_year = {end_year}")
        print(f"nens1 = {nens1}")
        print(f"for variable {key}")

        # If grid_box_plot is not None
        if gridbox_plot is not None:
            # Print
            print("Constraining data to gridbox_plot...")
            print("As defined by gridbox_plot = ", gridbox_plot)

            # Constrain the data to the gridbox_plot
            corr = corr[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]
            corr1_p = corr1_p[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]

        # Set up the axes
        ax = plt.subplot(nrows, 2, i + 1, projection=proj)

        # Include coastlines
        ax.coastlines()

        # # Add borders (?)
        # ax.add_feature(cfeature.BORDERS)

        # Set up the cf object
        cf = ax.contourf(lons, lats, corr, clevs, transform=proj, cmap="RdBu_r")

        # If gridbox_corr is not None
        if gridbox_corr is not None:
            # Loggging
            print("Calculating the correlations with a specific gridbox...")
            print("As defined by gridbox_corr = ", gridbox_corr)

            # Constrain the ts to the gridbox_corr
            fcst1_ts = fcst1_ts[
                :, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr
            ]
            obs_ts = obs_ts[:, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]

            # Calculate the mean of both time series
            fcst1_ts_mean = np.mean(fcst1_ts, axis=(1, 2))
            obs_ts_mean = np.mean(obs_ts, axis=(1, 2))

            # Calculate the correlation between the two time series
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Show these values on the plot
            ax.text(
                0.05,
                0.05,
                f"r = {r:.2f}, p = {p:.2f}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8,
            )

            # Add the gridbox to the plot
            ax.plot(
                [lon1_corr, lon2_corr, lon2_corr, lon1_corr, lon1_corr],
                [lat1_corr, lat1_corr, lat2_corr, lat2_corr, lat1_corr],
                color="green",
                linewidth=2,
                transform=proj,
            )

        # If any of the corr1 values are NaNs
        # then set the p values to NaNs at the same locations
        corr1_p[np.isnan(corr)] = np.nan

        # If any of the corr1_p values are greater than the sig_threshold
        # then set the corr1 values to NaNs at the same locations
        # WHat if the sig threshold is two sided?
        # We wan't to set the corr1_p values to NaNs
        # Where corr1_p<sig_threshold and corr1_p>1-sig_threshold
        corr1_p[(corr1_p > sig_threshold) & (corr1_p < 1 - sig_threshold)] = np.nan

        # plot the p-values
        ax.contourf(lons, lats, corr1_p, hatches=[".."], alpha=0.0, transform=proj)

        # Add a text box with the axis label
        ax.text(
            0.95,
            0.05,
            f"{axis_labels[i]}",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8,
        )

        # Add a textboc with the variable name in the top left
        ax.text(
            0.05,
            0.95,
            f"{key}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8,
        )

        # Include the number of ensemble members in the top right of the figure
        ax.text(
            0.95,
            0.95,
            f"n = {nens1}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8,
        )

        # Add a text box with the season in the top right
        # ax.text(0.95, 0.95, f"{season}", transform=ax.transAxes,
        #         va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
        #         fontsize=6)

        # Add the contourf object to the list
        cf_list.append(cf)

        # Add the axes to the list
        axes.append(ax)

    # Add a colorbar
    cbar = fig.colorbar(
        cf_list[0], ax=axes, orientation="horizontal", pad=0.05, shrink=0.8
    )
    cbar.set_label("correlation coefficient", fontsize=10)

    # Now plot the NAO index
    print("Plotting the raw NAO index...")

    # Extract the total_nesn
    total_nens = nao_members.shape[0]

    # Calculate the mean of the nao_members
    nao_members_mean = np.mean(nao_members, axis=0)

    # Calculate the correlation between the nao_members_mean and the obs_ts
    corr1, p1 = pearsonr(nao_members_mean, nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts"])

    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1 = (corr1) / (np.std(nao_members_mean) / np.std(nao_members))

    # Calculate the 5th and 95th percentiles of the nao_members
    nao_members_mean_min = np.percentile(nao_members, 5, axis=0)
    nao_members_mean_max = np.percentile(nao_members, 95, axis=0)

    # Plot the ensemble mean
    ax1 = axs[2, 0]

    # Plot the ensemble mean
    ax1.plot(
        nao_stats_dict["BCC-CSM2-MR"]["years"] - init_offset,
        nao_members_mean / 100,
        color="red",
        label="dcppA",
    )

    # Plot the obs
    ax1.plot(
        nao_stats_dict["BCC-CSM2-MR"]["years"] - init_offset,
        nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts"] / 100,
        color="black",
        label="ERA5",
    )

    # Plot the 5th and 95th percentiles
    ax1.fill_between(
        nao_stats_dict["BCC-CSM2-MR"]["years"] - init_offset,
        nao_members_mean_min / 100,
        nao_members_mean_max / 100,
        color="red",
        alpha=0.2,
    )

    # Add a text box with the axis label
    ax1.text(
        0.95,
        0.05,
        f"{axis_labels[-2]}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=8,
    )

    # Include a textbox containing the total nens in the top right
    ax1.text(
        0.95,
        0.95,
        f"n = {total_nens}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=8,
    )

    # Include a title which contains the correlations
    # p values and the rpccorr1_pbbbbeuibweiub
    ax1.set_title(
        f"ACC = {corr1:.2f} (p = {p1:.2f}), " f"RPC = {rpc1:.2f}, " f"N = {total_nens}",
        fontsize=10,
    )

    # Include a legend
    ax1.legend(loc="upper left", fontsize=8)

    # Set a horizontal line at zero
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Set the y limits
    ax1.set_ylim(ylims)

    # Set the y label
    ax1.set_ylabel("NAO (hPa)")

    # Set the x label
    ax1.set_xlabel("Initialisation year")

    # Print that we are plotting the lag and variance adjusted NAO index
    print("Plotting the lag and variance adjusted NAO index...")

    # Extract the total_nens
    total_nens_lag = nao_members_lag.shape[0]

    # Calculate the mean of the nao_members
    nao_members_mean_lag = np.mean(nao_members_lag, axis=0)

    # Calculate the correlation between the nao_members_mean and the obs_ts
    corr1_lag, p1_lag = pearsonr(
        nao_members_mean_lag, nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts_lag"]
    )

    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1_lag = (corr1_lag) / (np.std(nao_members_mean_lag) / np.std(nao_members_lag))

    # Calculate the rps between the model nao index and the obs nao index
    rps1 = rpc1_lag * (
        np.std(nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts_lag"])
        / np.std(nao_members_lag)
    )

    # Var adjust the NAO
    nao_var_adjust = nao_members_mean_lag * rps1

    # Calculate the RMSE between the ensemble mean and observations
    rmse = np.sqrt(
        np.mean((nao_var_adjust - nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts_lag"]) ** 2)
    )

    # Calculate the upper and lower bounds
    ci_lower = nao_var_adjust - rmse
    ci_upper = nao_var_adjust + rmse

    # Set up the axes
    ax2 = axs[2, 1]

    # Plot the ensemble mean
    ax2.plot(
        nao_stats_dict["BCC-CSM2-MR"]["years_lag"] - init_offset,
        nao_var_adjust / 100,
        color="red",
        label="dcppA",
    )

    # Plot the obs
    ax2.plot(
        nao_stats_dict["BCC-CSM2-MR"]["years_lag"] - init_offset,
        nao_stats_dict["BCC-CSM2-MR"]["obs_nao_ts_lag"] / 100,
        color="black",
        label="ERA5",
    )

    # Plot the 5th and 95th percentiles
    ax2.fill_between(
        nao_stats_dict["BCC-CSM2-MR"]["years_lag"] - init_offset,
        ci_lower / 100,
        ci_upper / 100,
        color="red",
        alpha=0.2,
    )

    # Add a text box with the axis label
    ax2.text(
        0.95,
        0.05,
        f"{axis_labels[-1]}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=8,
    )

    # Include a textbox containing the total nens in the top right
    ax2.text(
        0.95,
        0.95,
        f"n = {total_nens_lag}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=8,
    )

    # Include a title which contains the correlations
    # p values and rpc values
    ax2.set_title(
        f"ACC = {corr1_lag:.2f} (p = {p1_lag:.2f}), "
        f"RPC = {rpc1_lag:.2f}, "
        f"N = {total_nens_lag}",
        fontsize=10,
    )

    # Set a horizontal line at zero
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # # Share the y axis with ax1
    # ax2.set_ylim([-10, 10])

    # Share the y-axis with ax1
    ax2.sharey(ax1)

    # Set up the pathname for saving the figure
    fig_name = f"different_variables_corr_{start_year}_{end_year}"

    # Set up the plots directory
    plots_dir = "/gws/nopw/j04/canari/users/benhutch/plots"

    # Set up the path to the figure
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # # Specify a tight layout
    # plt.tight_layout()

    # Show the figure
    plt.show()


# Define a function for checking whether bootstrapped files exist for list of variables
def check_bootstraps_exist(
    variables: list,
    no_bootstraps: list,
    season: str,
    forecast_range: str,
    region: str = "global",
    method: list = ["raw", "alt_lag"],
    base_dir: str = "/gws/nopw/j04/canari/users/benhutch/bootstrapping",
):
    """
    Function which checks whether bootstrapped files exist for a list of variables.

    Inputs:
    -------

    variables: list
        List of variables to process.
        e.g. ["tas", "pr", "psl"]

    no_bootstraps: list
        List of the no bootstraps for each variable.
        e.g. [100, 100, 100]

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    region: str
        Region to process.
        e.g. default is "global"

    method: list
        List of methods to process.
        e.g. ["raw", "alt_lag"]

    base_dir: str
        Base directory to process.
        e.g. default is "/gws/nopw/j04/canari/users/benhutch/bootstrapping"

    Outputs:
    --------
    bootstraps_exist: bool
        Boolean value indicating whether bootstraps exist for the variables.
    """

    # Create a dictionary to store the list of files available
    # for each variable, method and no_bootstraps combination
    files_available = {}

    # Loop over the methods
    for meth in method:
        print(f"Checking whether bootstraps exist for method {meth}...")

        # Loop over the variables and no_bootstraps at the same time
        for var, nboot in zip(variables, no_bootstraps):
            print(f"Checking whether bootstraps exist for variable {var}...")
            print(f"Checking whether bootstraps exist for {nboot} bootstraps...")

            # Set up the base path
            base_path = os.path.join(
                base_dir,
                var,
                region,
                season,
                forecast_range,
                meth,
                f"no_bootstraps_{nboot}",
            )

            # # Assert that this directory exists and is not empty
            # assert os.path.isdir(base_path), f"Directory {base_path} does not exist!"

            # If the directory does not exist
            if not os.path.isdir(base_path):
                print(f"Directory {base_path} does not exist!")

                # continue to the next iteration
                continue

            # Set up the key for the files_available dictionary
            key = (meth, var, f"nboot_{nboot}")

            # List the files in the directory
            files = os.listdir(base_path)

            # Add the files to the dictionary
            files_available[key] = files

    # Return the files_available dictionary
    return files_available


# Create another function to form the bs_skill_maps dictionary
def create_bs_dict(
    variables: list,
    no_bootstraps: list,
    season: str,
    forecast_range: str,
    methods: list,
    region: str = "global",
    base_dir: str = "/gws/nopw/j04/canari/users/benhutch/bootstrapping",
    model_season: str = "ONDJFM",
    load_hist: bool = False,
):
    """
    Function which creates a dictionary of bootstrapped skill maps for a list of variables.

    Inputs:
    -------

    variables: list
        List of variables to process.
        e.g. ["tas", "pr", "psl"]

    no_bootstraps: list
        List of the no bootstraps for each variable.
        e.g. [100, 100, 100]

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    region: str
        Region to process.
        e.g. default is "global"

    method: list
        List of methods to process.
        e.g. ["raw", "alt_lag", "alt_lag2"]

    base_dir: str
        Base directory to process.
        e.g. default is "/gws/nopw/j04/canari/users/benhutch/bootstrapping"

    model_season: str
        Model season to process.

    load_hist: bool
        Boolean value indicating whether to load the historical data.

    Outputs:
    --------
    bs_skill_maps: dict
        Dictionary containing the bootstrapped skill maps for each variable.
        Dictionary keys are the variable names.
        e.g. bs_skill_maps["tas"] = {"raw": {"corr": corr, "rmse": rmse, "bias": bias},
                                        "alt_lag": {"corr": corr, "rmse": rmse, "bias": bias}}
    """

    # Create an empty dictionary to store the bootstrapped skill maps
    bs_skill_maps = {}

    # Set up the mdi
    mdi = -9999.0

    # FIXME: fix tyhis - add f2
    # Create a dictionary containing the reauired stars for each variable
    stats_dict = {
        "corr1": [],
        "corr1_short": [],
        "corr1_p": [],
        "corr1_p_short": [],
        "corr2": [],
        "corr2_p": [],
        "corr12": [],
        "corr12_p": [],
        "corr_diff": [],
        "corr_diff_p": [],
        "partialr": [],
        "partialr_p": [],
        "f1_ts": [],
        "f2_ts": [],
        "f1_ts_short": [],
        "f1_em_resid": [],
        "obs_resid": [],
        "o_ts": [],
        "o_ts_short": [],
        "nens1": mdi,
        "nens2": mdi,
        "start_year": mdi,
        "end_year": mdi,
        "end_year_short": mdi,
    }

    # Loop over the variables and nboot
    for var, nboot, method in zip(variables, no_bootstraps, methods):
        print(f"Processing variable {var}...")
        print(f"Processing {nboot} bootstraps...")

        # Set up the base path
        base_path = os.path.join(
            base_dir,
            var,
            region,
            season,
            forecast_range,
            method,
            f"no_bootstraps_{nboot}",
        )

        # Print the base path
        print(f"base_path = {base_path}")

        # Assert that this directory exists and is not empty
        assert os.path.isdir(base_path), f"Directory {base_path} does not exist!"

        # Set up the dictionary key for this variable
        key = (var, f"nboot_{nboot}", method)

        # Create an empty dictionary to store the skill maps for this variable
        skill_maps = {}

        # Find the file containing "corr1_{variable}" in the base_path
        corr1_file = [
            file
            for file in os.listdir(base_path)
            if f"corr1_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # Corr1 file short
        corr1_short_file = [
            file
            for file in os.listdir(base_path)
            if f"corr1_{var}_{region}_{model_season}_{forecast_range}_short" in file
        ]

        # Find the file containing "corr1_p_{variable}" in the base_path
        corr1_p_file = [
            file
            for file in os.listdir(base_path)
            if f"corr1_p_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # fiel containing corr1_p_short
        corr1_p_short_file = [
            file
            for file in os.listdir(base_path)
            if f"corr1_p_{var}_{region}_{model_season}_{forecast_range}_short" in file
        ]

        # file containing corr2
        corr2_file = [
            file
            for file in os.listdir(base_path)
            if f"corr2_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing corr2_p
        corr2_p_file = [
            file
            for file in os.listdir(base_path)
            if f"corr2_p_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing corr12
        corr12_file = [
            file
            for file in os.listdir(base_path)
            if f"corr12_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing corr12_p
        corr12_p_file = [
            file
            for file in os.listdir(base_path)
            if f"corr12_p_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing corr_diff
        corr_diff_file = [
            file
            for file in os.listdir(base_path)
            if f"corr_diff_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing corr_diff_p
        corr_diff_p_file = [
            file
            for file in os.listdir(base_path)
            if f"corr_diff_p_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing partialr
        partialr_file = [
            file
            for file in os.listdir(base_path)
            if f"partial_r_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing partialr_p
        partialr_p_file = [
            file
            for file in os.listdir(base_path)
            if f"partial_r_p_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # file containing f1_em_resid
        f1_em_resid_file = [
            file
            for file in os.listdir(base_path)
            if f"fcst1_em_resid_{var}_{region}_{model_season}_{forecast_range}.npy"
            in file
        ]

        # file containing obs_resid
        obs_resid_file = [
            file
            for file in os.listdir(base_path)
            if f"obs_resid_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # Find the file containing "fcst1_ts_{variable}" in the base_path
        fcst1_ts_file = [
            file
            for file in os.listdir(base_path)
            if f"fcst1_ts_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # Find the file containing "fcst2_ts_{variable}" in the base_path
        fcst2_ts_file = [
            file
            for file in os.listdir(base_path)
            if f"fcst2_ts_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # Short fcst1_ts file
        fcst1_ts_file_short = [
            file
            for file in os.listdir(base_path)
            if f"fcst1_ts_{var}_{region}_{model_season}_{forecast_range}_short" in file
        ]

        # Find the file containing "obs_ts_{variable} short
        obs_ts_file_short = [
            file
            for file in os.listdir(base_path)
            if f"obs_ts_{var}_{region}_{model_season}_{forecast_range}_short" in file
        ]

        # Find the file containing "obs_ts_{variable}" in the base_path
        obs_ts_file = [
            file
            for file in os.listdir(base_path)
            if f"obs_ts_{var}_{region}_{model_season}_{forecast_range}.npy" in file
        ]

        # Find the file containing "nens1_{variable}" in the base_path
        nens1_file = [file for file in os.listdir(base_path) if f"nens1_{var}" in file]

        # File containing nens2
        nens2_file = [file for file in os.listdir(base_path) if f"nens2_{var}" in file]

        # Find the file containing "start_end_years_{variable}" in the base_path
        start_end_years_file = [
            file for file in os.listdir(base_path) if f"start_end_years_{var}" in file
        ]

        # if start_end_years_file is empty
        if not start_end_years_file:
            # Find the file ending in "common_years.npy"
            common_years_file = [
                file for file in os.listdir(base_path) if "common_years.npy" in file
            ]

            # Asserrt that the length of each of the lists is equal to 1
            for file in [
                corr1_file,
                corr1_p_file,
                fcst1_ts_file,
                obs_ts_file,
                nens1_file,
            ]:
                assert (
                    len(file) == 1
                ), f"Length of file list is not equal to 1 for {file}"
        else:
            # Assert that the length of each of these lists is equal to 1
            for file in [
                corr1_file,
                corr1_p_file,
                fcst1_ts_file,
                obs_ts_file,
                nens1_file,
                start_end_years_file,
            ]:
                assert (
                    len(file) == 1
                ), f"Length of file list is not equal to 1 for {file}"

        # Load the files
        corr1 = np.load(os.path.join(base_path, corr1_file[0]))

        # Load the files
        corr1_p = np.load(os.path.join(base_path, corr1_p_file[0]))

        if load_hist:
            # load the corr2 file
            corr2 = np.load(os.path.join(base_path, corr2_file[0]))

            # load the corr2_p file
            corr2_p = np.load(os.path.join(base_path, corr2_p_file[0]))

            # load the corr12 file
            corr12 = np.load(os.path.join(base_path, corr12_file[0]))

            # load the corr12_p file
            corr12_p = np.load(os.path.join(base_path, corr12_p_file[0]))

            # load the corr_diff file
            corr_diff = np.load(os.path.join(base_path, corr_diff_file[0]))

            # load the corr_diff_p file
            corr_diff_p = np.load(os.path.join(base_path, corr_diff_p_file[0]))

            # load the partialr file
            partialr = np.load(os.path.join(base_path, partialr_file[0]))

            # load the partialr_p file
            partialr_p = np.load(os.path.join(base_path, partialr_p_file[0]))

            # load the f1_em_resid file
            f1_em_resid = np.load(os.path.join(base_path, f1_em_resid_file[0]))

            # load the obs_resid file
            obs_resid = np.load(os.path.join(base_path, obs_resid_file[0]))

            # Load the files
            fcst2_ts = np.load(os.path.join(base_path, fcst2_ts_file[0]))

            # Load the nens2 file
            nens2 = np.loadtxt(os.path.join(base_path, nens2_file[0])).astype(int)

        # Load the files
        fcst1_ts = np.load(os.path.join(base_path, fcst1_ts_file[0]))

        # Load the files
        obs_ts = np.load(os.path.join(base_path, obs_ts_file[0]))

        # Load the files
        nens1 = np.loadtxt(os.path.join(base_path, nens1_file[0])).astype(int)

        # If all of the files exist
        if (
            corr1_short_file
            and corr1_p_short_file
            and fcst1_ts_file_short
            and obs_ts_file_short
        ):
            # Load the files
            corr1_short = np.load(os.path.join(base_path, corr1_short_file[0]))

            # Load the files
            corr1_p_short = np.load(os.path.join(base_path, corr1_p_short_file[0]))

            # Load the files
            fcst1_ts_short = np.load(os.path.join(base_path, fcst1_ts_file_short[0]))

            # Load the files
            obs_ts_short = np.load(os.path.join(base_path, obs_ts_file_short[0]))

            # # Load the files
            # common_years = np.load(os.path.join(base_path, common_years_file[0]))

            # # Extract the start and end years
            # start_year = common_years[0].astype(int)
            # end_year = common_years[1].astype(int)

            # end year short
            end_year_short = 2014

            # Add the values to the skill_maps dictionary
            skill_maps["corr1_short"] = corr1_short

            # Add the values to the skill_maps dictionary
            skill_maps["corr1_p_short"] = corr1_p_short

            # Add the values to the skill_maps dictionary
            skill_maps["f1_ts_short"] = fcst1_ts_short

            # Add the values to the skill_maps dictionary
            skill_maps["o_ts_short"] = obs_ts_short

            # Add the values to the skill_maps dictionary
            skill_maps["end_year_short"] = end_year_short

        if not start_end_years_file:
            print("start end years file does not exist")
            # load the common years file
            # common_years = np.load(os.path.join(base_path, common_years_file[0]))

            # # Extract the start and end years
            # start_year = common_years[0].astype(int)
            # end_year = common_years[1].astype(int)
        else:
            # Load the files
            start_year = np.loadtxt(
                os.path.join(base_path, start_end_years_file[0])
            ).astype(int)

            # Load the files
            end_year = np.loadtxt(
                os.path.join(base_path, start_end_years_file[0])
            ).astype(int)

        # Add the values to the skill_maps dictionary
        skill_maps["corr1"] = corr1

        # Add the values to the skill_maps dictionary
        skill_maps["corr1_p"] = corr1_p

        # Add the values to the skill_maps dictionary
        skill_maps["f1_ts"] = fcst1_ts

        # Add the values to the skill_maps dictionary
        skill_maps["o_ts"] = obs_ts

        # Add the values to the skill_maps dictionary
        skill_maps["nens1"] = nens1

        # Add the values to the skill_maps dictionary
        skill_maps["start_year"] = start_year

        # Add the values to the skill_maps dictionary
        skill_maps["end_year"] = end_year

        if load_hist:
            # Add the obs residual values
            skill_maps["obs_resid"] = obs_resid

            # add the values to the skill_maps dictionary
            skill_maps["nens2"] = nens2

            # Add the values for f1_em_resid
            skill_maps["f1_em_resid"] = f1_em_resid

            # add the values to the skill_maps dictionary
            skill_maps["f2_ts"] = fcst2_ts

            # Add the values to the skill_maps dictionary
            skill_maps["corr2"] = corr2

            # Add the values to the skill_maps dictionary
            skill_maps["corr2_p"] = corr2_p

            # Add the values to the skill_maps dictionary
            skill_maps["corr12"] = corr12

            # Add the values to the skill_maps dictionary
            skill_maps["corr12_p"] = corr12_p

            # Add the values to the skill_maps dictionary
            skill_maps["corr_diff"] = corr_diff

            # Add the values to the skill_maps dictionary
            skill_maps["corr_diff_p"] = corr_diff_p

            # Add the values to the skill_maps dictionary
            skill_maps["partialr"] = partialr

            # Add the values to the skill_maps dictionary
            skill_maps["partialr_p"] = partialr_p

        # Add the skill_maps dictionary to the bs_skill_maps dictionary
        bs_skill_maps[key] = skill_maps

    # if there are not four keys

    # Return the bs_skill_maps dictionary
    return bs_skill_maps


# Now we want to write a function to plot the skill maps
def plot_diff_variables(
    bs_skill_maps: dict,
    season: str,
    forecast_range: str,
    method_load: str,
    methods: list = None,
    figsize_x: int = 10,
    figsize_y: int = 12,
    gridbox_corr: dict = None,
    gridbox_plot: dict = None,
    sig_threshold: float = 0.05,
    winter_n_gridbox_corr: dict = dicts.iceland_grid_corrected,
    winter_s_gridbox_corr: dict = dicts.azores_grid_corrected,
    summer_n_gridbox_corr: dict = dicts.snao_north_grid,
    summer_s_gridbox_corr: dict = dicts.snao_south_grid,
    short_period: bool = False,
    plot_corr_diff: bool = False,
    plot_long_short_diff: bool = False,
    second_bs_skill_maps: dict = None,
    methods_diff: str = None,
    plot_winter_nodes: bool = False,
    corr_list: list = None,
    ts_list: list = None,
    fontsize: int = 14,
):
    """
    Plot the skill maps for different variables as a 2x2 grid.

    Inputs:
    -------

    bs_skill_maps: dict
        Dictionary containing the bootstrapped skill maps for each variable.
        Dictionary keys are the variable names.
        e.g. bs_skill_maps["tas"] = {"raw": {"corr": corr, "rmse": rmse, "bias": bias},
                                        "alt_lag": {"corr": corr, "rmse": rmse, "bias": bias}}

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    methods: list
        List of methods to process.
        Default is None.

    figsize_x: int
        Width of the figure in inches.
        e.g. default is 10

    figsize_y: int
        Height of the figure in inches.
        e.g. default is 12

    gridbox_corr: list[dict]
        A list of dictionaries containing the gridbox(es) for which to plot the skill maps.
        e.g. default is None

    gridbox_plot: 
        Dictionary with the dimensions of the area to plot.
        e.g. default is None

    sig_threshold: float
        Significance threshold for the correlation coefficients.
        e.g. default is 0.05

    winter_n_gridbox_corr: dict
        Dictionary containing the gridbox for which to calculate the correlation.
        e.g. default is dicts.iceland_grid_corrected

    winter_s_gridbox_corr: dict
        Dictionary containing the gridbox for which to calculate the correlation.
        e.g. default is dicts.azores_grid_corrected

    summer_n_gridbox_corr: dict
        Dictionary containing the gridbox for which to calculate the correlation.
        e.g. default is dicts.snao_north_grid

    summer_s_gridbox_corr: dict
        Dictionary containing the gridbox for which to calculate the correlation.
        e.g. default is dicts.snao_south_grid

    short_period: bool
        Boolean value indicating whether to constrain the data to the short
        period from Smith et al., 2020.
        Default is False.

    plot_corr_diff: bool
        Boolean value indicating whether to plot the difference in correlations
        between two methods.
        Default is False.

    plot_long_short_diff: bool
        Boolean value indicating whether to plot the difference in correlations
        between the long and short periods.
        Default is False.

    second_bs_skill_maps: dict
        Dictionary containing the bootstrapped skill maps for the second method.
        Default is None.

    plot_winter_nodes: bool
        Boolean value indicating whether to plot the winter nodes.

    corr_list: list
        List of correlation values to plot.
        Default is None.

    ts_list: list
        List of time series to plot.
        Default is None.

    Outputs:
    --------

    None
    """

    # Set up the axis labels
    axis_labels = ["a", "b", "c", "d", "e", "f"]

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Set the nrows depending on the number of keys in the bs_skill_maps dictionary
    # e.g. if this is 5 then nrows = 3
    nrows = int(np.ceil(len(bs_skill_maps.keys()) / 2))

    # print that we are setting up axis here
    print("Setting up the axis...")
    # Set up the figure
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    # if the len of bs_skill_maps.keys is 3
    if len(bs_skill_maps.keys()) == 3:
        print("Setting up the gridspec")
        # Set up the gridspec
        spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
    elif len(bs_skill_maps.keys()) == 5:
        # Set up the gridspec
        spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

    # # Adjust the whitespace
    # fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Extract the first variable name from bs_skill_maps
    # Extract the first element for each of the tuple keys
    variables = list(bs_skill_maps.keys())
    variables = [var[0] for var in variables]

    # print the variables
    print(f"variables = {variables}")

    # sys.exit()

    # If the gridbox_corr is not None
    if gridbox_corr is not None and "psl" in variables:
        # Initialize empty lists for the indexes
        lat1_idxs = [] ; lat2_idxs = []

        # And for the lons
        lon1_idxs = [] ; lon2_idxs = []

        # and for the corrs
        lon1_corrs = [] ; lon2_corrs = []

        # and for the lats
        lat1_corrs = [] ; lat2_corrs = []
        
        # Assert that gridbox_corr is a list
        assert isinstance(gridbox_corr, list), "gridbox_corr is not a list!"

        # Loop over the items in gridbox_corr
        for gridbox in gridbox_corr:
            # Extract the lats and lons from the gridbox_corr
            lon1_corr, lon2_corr = gridbox["lon1"], gridbox["lon2"]
            lat1_corr, lat2_corr = gridbox["lat1"], gridbox["lat2"]

            # find the indices of the lats which correspond to the gridbox
            lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
            lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
            lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))

            # Append the indices to the lists
            lat1_idxs.append(lat1_idx_corr)

            # Append the indices to the lists
            lat2_idxs.append(lat2_idx_corr)

            # Append the indices to the lists
            lon1_idxs.append(lon1_idx_corr)

            # Append the indices to the lists
            lon2_idxs.append(lon2_idx_corr)

            # Append the lats and lons to the lists
            lon1_corrs.append(lon1_corr)

            # Append the lats and lons to the lists
            lon2_corrs.append(lon2_corr)

            # Append the lats and lons to the lists
            lat1_corrs.append(lat1_corr)

            # Append the lats and lons to the lists
            lat2_corrs.append(lat2_corr)

        # Depending on the season
        if season in ["ONDJFM", "DJFM", "DJF"]:
            # Set up the the southern gridbox
            s_grid = winter_s_gridbox_corr
            n_grid = winter_n_gridbox_corr
        elif season in ["JJA", "JJAS", "JAS", "AMJJAS", "ULG"]:
            # Set up the the southern gridbox
            s_grid = summer_s_gridbox_corr
            n_grid = summer_n_gridbox_corr
        else:
            raise ValueError("Season not recognised!")

        # Extract the lats and lons from the gridbox_corr
        lon1_corr_n, lon2_corr_n = n_grid["lon1"], n_grid["lon2"]
        lat1_corr_n, lat2_corr_n = n_grid["lat1"], n_grid["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lon1_corr_s, lon2_corr_s = s_grid["lon1"], s_grid["lon2"]
        lat1_corr_s, lat2_corr_s = s_grid["lat1"], s_grid["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr_n = np.argmin(np.abs(lats - lat1_corr_n))
        lat2_idx_corr_n = np.argmin(np.abs(lats - lat2_corr_n))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr_n = np.argmin(np.abs(lons - lon1_corr_n))
        lon2_idx_corr_n = np.argmin(np.abs(lons - lon2_corr_n))

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr_s = np.argmin(np.abs(lats - lat1_corr_s))
        lat2_idx_corr_s = np.argmin(np.abs(lats - lat2_corr_s))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr_s = np.argmin(np.abs(lons - lon1_corr_s))
        lon2_idx_corr_s = np.argmin(np.abs(lons - lon2_corr_s))
    elif gridbox_corr is not None and "psl" not in variables:
        # Initialize empty lists for the indexes
        lat1_idxs = [] ; lat2_idxs = []

        # And for the lons
        lon1_idxs = [] ; lon2_idxs = []

        # and for the corrs
        lon1_corrs = [] ; lon2_corrs = []

        # and for the lats
        lat1_corrs = [] ; lat2_corrs = []
        
        # Assert that gridbox_corr is a list
        assert isinstance(gridbox_corr, list), "gridbox_corr is not a list!"

        # Loop over the items in gridbox_corr
        for gridbox in gridbox_corr:
            # Extract the lats and lons from the gridbox_corr
            lon1_corr, lon2_corr = gridbox["lon1"], gridbox["lon2"]
            lat1_corr, lat2_corr = gridbox["lat1"], gridbox["lat2"]

            # find the indices of the lats which correspond to the gridbox
            lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
            lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
            lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))

            # Append the indices to the lists
            lat1_idxs.append(lat1_idx_corr)

            # Append the indices to the lists
            lat2_idxs.append(lat2_idx_corr)

            # Append the indices to the lists
            lon1_idxs.append(lon1_idx_corr)

            # Append the indices to the lists
            lon2_idxs.append(lon2_idx_corr)

            # Append the lats and lons to the lists
            lon1_corrs.append(lon1_corr)

            # Append the lats and lons to the lists
            lon2_corrs.append(lon2_corr)

            # Append the lats and lons to the lists
            lat1_corrs.append(lat1_corr)

            # Append the lats and lons to the lists
            lat2_corrs.append(lat2_corr)
    else:
        AssertionError("gridbox_corr is None and variable is not psl")

    # If gridbox_plot is not None
    if gridbox_plot is not None:
        # Extract the lats and lons from the gridbox_plot
        lon1_gb, lon2_gb = gridbox_plot["lon1"], gridbox_plot["lon2"]
        lat1_gb, lat2_gb = gridbox_plot["lat1"], gridbox_plot["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_gb = np.argmin(np.abs(lats - lat1_gb))
        lat2_idx_gb = np.argmin(np.abs(lats - lat2_gb))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_gb = np.argmin(np.abs(lons - lon1_gb))
        lon2_idx_gb = np.argmin(np.abs(lons - lon2_gb))

        # Constrain the lats and lons
        lats = lats[lat1_idx_gb:lat2_idx_gb]
        lons = lons[lon1_idx_gb:lon2_idx_gb]

    if plot_corr_diff is True:
        # Set up the contour levels
        clevs = np.arange(-0.8, 0.9, 0.1)
    elif plot_long_short_diff is True:
        # Set up the contour levels
        clevs = np.arange(-0.5, 0.6, 0.1)
    else:
        # Set up the contour levels
        # clevs = np.arange(-1.0, 1.1, 0.4)
        clevs = np.array([-1. , -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Set up a list to store the contourf objects
    cf_list = []

    # Set up the axes
    axes = []

    if plot_corr_diff is True:
        print("Plotting the difference in correlations...")

        # if the second_bs_skill_maps is None
        if second_bs_skill_maps is None:
            AssertionError("second_bs_skill_maps is None!")

        # Loop over the keys in the bs_skill_maps dictionary
        for i, (key, skill_maps) in enumerate(bs_skill_maps.items()):
            # Print the variable name
            print(f"Plotting variable {key}...")
            print(f"Plotting index {i}...")

            # print the keys for the second_bs_skill_maps dictionary
            print(
                f"keys for second_bs_skill_maps = {list(second_bs_skill_maps.keys())}"
            )

            # Extract the key for this index from the bs_skill_maps dictionary
            key2 = list(second_bs_skill_maps.keys())[i]

            # Use this key to extract the skill_maps from the second_bs_skill_maps dictionary
            skill_maps2 = second_bs_skill_maps[key2]

            # Extract the correlation arrays from the skill_maps dictionary
            corr = skill_maps["corr1"]
            corr2 = skill_maps2["corr1"]

            # Extract the values
            nens1 = skill_maps["nens1"]
            nens2 = skill_maps2["nens1"]

            # Extract the start and end years
            start_year = skill_maps["start_year"]
            end_year = skill_maps["end_year"]

            # if short_period is True and the short period correlations are available
            if short_period and skill_maps.get("corr1_short") is not None:
                print("Using the short period correlations...")

                # Set up the years
                years = np.arange(
                    skill_maps["start_year"][0], skill_maps["end_year_short"] + 1
                )

                # Set up the time series
                corr = skill_maps["corr1_short"]
                corr2 = skill_maps2["corr1_short"]

            # Print the start and end years
            print(f"start_year = {start_year}")
            print(f"end_year = {end_year}")
            print(f"nens1 = {nens1}")
            print(f"nens2 = {nens2}")
            print(f"for variable {key}")

            # If grid_box_plot is not None
            if gridbox_plot is not None:
                # Print
                print("Constraining data to gridbox_plot...")
                print("As defined by gridbox_plot = ", gridbox_plot)

                # Constrain the data to the gridbox_plot
                corr = corr[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]
                corr2 = corr2[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]

            print("Setting axes 2nd time?")
            # Set up the axes
            ax = plt.subplot(nrows, 2, i + 1, projection=proj)

            # Include coastlines
            ax.coastlines()

            # incldue borders here
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Set up the cf object
            cf = ax.contourf(
                lons,
                lats,
                corr - corr2,
                clevs,
                transform=proj,
                cmap="bwr",
                extend="both",
            )

            # Extract the variable name from the key
            variable = key[0]
            nboot_str = key[1]

            # Add a textbox with the axis label
            ax.text(
                0.95,
                0.05,
                f"{axis_labels[i]}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            # Set the var name depending on the variable
            if variable == "tas":
                var_name = "Temperature"
            elif variable == "pr":
                var_name = "Precipitation"
            elif variable == "psl":
                var_name = "Sea level pressure"
            elif variable == "nao":
                var_name = "NAO"
            elif variable == "sfcWind":
                # Set the variable
                var_name = "10m wind speed"
            elif variable == "rsds":
                # Set the variable
                var_name = "Solar irradiance"
            else:
                # Print the key
                print(f"variable = {variable}")
                AssertionError("Variable not recognised!")

            # Set a textbox with the variable name in the top left
            ax.text(
                0.05,
                0.95,
                f"{var_name}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            if methods_diff is not None:
                if short_period is False:
                    # Include the method in the top right of the figure
                    ax.text(
                        0.95,
                        0.95,
                        f"{methods_diff} ({nens1}, {nens2})\ncorr diff",
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        bbox=dict(facecolor="white", alpha=0.5),
                        fontsize=fontsize,
                    )
                else:
                    # Include the method in the top right of the figure
                    ax.text(
                        0.95,
                        0.95,
                        f"{methods_diff} ({nens1}, {nens2})\nshort period corr diff",
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        bbox=dict(facecolor="white", alpha=0.5),
                        fontsize=fontsize,
                    )

            # Add the cf object to the cf_list
            cf_list.append(cf)

            # Add the ax object to the axes list
            axes.append(ax)
    elif plot_long_short_diff is True:
        print(
            "Plotting the difference in correlations between the long and short periods..."
        )

        # Loop over the keys in the bs_skill_maps dictionary
        for i, (key, skill_maps) in enumerate(bs_skill_maps.items()):
            # Print the variable name
            print(f"Plotting variable {key}...")
            print(f"Plotting index {i}...")

            # extract the long corr arrays
            long_corr = skill_maps["corr1"]

            # Extract the nens
            nens1 = skill_maps["nens1"]

            # Extract the start and end years
            start_year = skill_maps["start_year"]
            end_year = skill_maps["end_year"]

            # If skill_maps contains the short period correlations
            if skill_maps.get("corr1_short") is not None:
                print("Using the short period correlations...")

                # Set up the short corr
                short_corr = skill_maps["corr1_short"]
            else:
                print("Short period correlations not available...")
                print("setting short corr to long corr")

                # Set the short corr to the long corr
                short_corr = skill_maps["corr1"]

            # Print the start and end years
            print(f"start_year = {start_year}")
            print(f"end_year = {end_year}")

            # If grid_box_plot is not None
            if gridbox_plot is not None:
                # Print
                print("Constraining data to gridbox_plot...")
                print("As defined by gridbox_plot = ", gridbox_plot)

                # Constrain the data to the gridbox_plot
                long_corr = long_corr[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]
                short_corr = short_corr[
                    lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb
                ]

            print("setting up axis 2nd/3rd time?")
            # Set up the axes
            ax = plt.subplot(nrows, 2, i + 1, projection=proj)

            # Include coastlines
            ax.coastlines()

            # plot borders here
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            # Set up the cf object
            cf = ax.contourf(
                lons,
                lats,
                long_corr - short_corr,
                clevs,
                transform=proj,
                cmap="bwr",
                extend="both",
            )

            # Extract the variable name from the key
            variable = key[0]
            nboot_str = key[1]
            method = key[2]

            # Add a textbox with the axis label
            ax.text(
                0.95,
                0.05,
                f"{axis_labels[i]}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            # Set the var name depending on the variable
            if variable == "tas":
                var_name = "Temperature"
            elif variable == "pr":
                var_name = "Precipitation"
            elif variable == "psl":
                var_name = "Sea level pressure"
            elif variable == "nao":
                var_name = "NAO"
            elif variable == "sfcWind":
                # Set the variable
                var_name = "10m wind speed"
            elif variable == "rsds":
                # Set the variable
                var_name = "Solar irradiance"
            elif variable == "ua":
                # Set the variable
                var_name = "850U"
            else:
                # Print the key
                print(f"variable = {variable}")
                AssertionError("Variable not recognised!")

            # Set a textbox with the variable name in the top left
            ax.text(
                0.05,
                0.95,
                f"{var_name}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            # Include the method and "long - short"
            # In the top right hand corner
            ax.text(
                0.95,
                0.95,
                f"{method} ({nens1})\nlong - short corr diff",
                transform=ax.transAxes,
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            # Add the cf object to the cf_list
            cf_list.append(cf)

            # Add the ax object to the axes list
            axes.append(ax)
    else:
        print("Plotting the correlations for a single method...")

        # Loop over the keys in the bs_skill_maps dictionary
        for i, (key, skill_maps) in enumerate(bs_skill_maps.items()):
            # Logging
            print(f"Plotting variable {key}...")
            print(f"Plotting index {i}...")

            if corr_list is not None:
                # Set up the correlations
                corr = skill_maps[f"{corr_list[i]}"]
                corr1_p = skill_maps[f"{corr_list[i]}_p"]

                # assert that ts_list is not none
                assert ts_list is not None, "ts_list is None, but corr_list is not None"

                # Set up the time series
                fcst1_ts = skill_maps[f"{ts_list[i]}"]

                if ts_list[i] == "f1_em_resid":
                    obs_ts = skill_maps["obs_resid"]
                else:
                    # Set up the time series
                    obs_ts = skill_maps["o_ts"]

                if ts_list[i] == "f2_ts":
                    nens1 = skill_maps["nens2"]
                elif ts_list[i] == "f1_em_resid":
                    if int(skill_maps["nens2"]) == 20:
                        print("Correcting nens2 for NAO-match")

                        # Extract the first item in key
                        variable = key[0]

                        # if variable is tas
                        if variable == "tas":
                            skill_maps["nens2"] = 141
                        elif variable == "sfcWind":
                            skill_maps["nens2"] = 9
                        elif variable == "rsds":
                            skill_maps["nens2"] = 32
                        elif variable == "pr":
                            skill_maps["nens2"] = 163
                        else:
                            print("manual value not set up")

                    nens1 = int(skill_maps["nens1"]), int(skill_maps["nens2"])
                else:
                    # Set up the values
                    nens1 = skill_maps["nens1"]

                # Set up the start and end year
                start_year = skill_maps["start_year"]
                end_year = skill_maps["end_year"]
            else:
                # Extract the correlation arrays from the skill_maps dictionary
                corr = skill_maps["corr1"]
                corr1_p = skill_maps["corr1_p"]

                # Extract the time series
                fcst1_ts = skill_maps["f1_ts"]
                obs_ts = skill_maps["o_ts"]

                # Extract the values
                nens1 = skill_maps["nens1"]
                start_year = skill_maps["start_year"]
                end_year = skill_maps["end_year"]

            # if we are using the short period
            if short_period and skill_maps.get("corr1_short") is not None:
                print("Using the short period correlations...")
                # Set up the years
                years = np.arange(
                    skill_maps["start_year"][0], skill_maps["end_year_short"] + 1
                )

                # Set up the time series
                corr = skill_maps["corr1_short"]
                corr1_p = skill_maps["corr1_p_short"]

                # Set up the time series
                fcst1_ts = skill_maps["f1_ts_short"]

                # Set up the time series
                obs_ts = skill_maps["o_ts_short"]

            # Print the start and end years
            print(f"start_year = {start_year}")
            print(f"end_year = {end_year}")
            print(f"nens1 = {nens1}")
            print(f"for variable {key}")
            # print the len of the f1 and o1 time series
            print(f"fcst1_ts.shape = {fcst1_ts.shape}")
            print(f"obs_ts.shape = {obs_ts.shape}")

            # If grid_box_plot is not None
            if gridbox_plot is not None:
                # Print
                print("Constraining data to gridbox_plot...")
                print("As defined by gridbox_plot = ", gridbox_plot)

                # Constrain the data to the gridbox_plot
                corr = corr[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]
                corr1_p = corr1_p[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]

            print("Setting up axis 3rd/4th time this?")
            
            if len(bs_skill_maps.keys()) not in [3, 5]:
                # Set up the axes
                ax = plt.subplot(nrows, 2, i + 1, projection=proj)
            elif len(bs_skill_maps.keys()) == 3:
                if i == 0:
                    # Set up the axes
                    ax = plt.subplot(spec[0, 1:3], projection=proj)
                elif i == 1:
                    # Set up the axes
                    ax = plt.subplot(spec[1, 0:2], projection=proj)
                elif i == 2:
                    # Set up the axes
                    ax = plt.subplot(spec[1, 2:4], projection=proj
                    )
                else:
                    AssertionError("Too many keys in bs_skill_maps")
            elif len(bs_skill_maps.keys()) == 5:
                if i == 0:
                    # Set up the axes
                    ax = plt.subplot(spec[0, 1:3], projection=proj)
                elif i == 1:
                    # Set up the axes
                    ax = plt.subplot(spec[1, 0:2], projection=proj)
                elif i == 2:
                    # Set up the axes
                    ax = plt.subplot(spec[1, 2:4], projection=proj)
                elif i == 3:
                    # Set up the axes
                    ax = plt.subplot(spec[2, 0:2], projection=proj)
                elif i == 4:
                    # Set up the axes
                    ax = plt.subplot(spec[2, 2:4], projection=proj)
                else:
                    AssertionError("Too many keys in bs_skill_maps")

            # Include coastlines
            ax.coastlines()

            # # Add borders (?)
            ax.add_feature(cfeature.BORDERS)

            if ts_list is not None and "corr_diff" in corr_list:
                # Set up the cf object
                cf = ax.contourf(
                    lons,
                    lats,
                    corr,
                    clevs,
                    transform=proj,
                    cmap="bwr",
                    extend="both",
                )
            else:
                # Set up the cf object
                cf = ax.contourf(lons, lats, corr, clevs, transform=proj, cmap="bwr")

            # Extract the variable name from the key
            variable = key[0]
            nboot_str = key[1]

            # extract the lon and lat values
            lat1_idx_corr = lat1_idxs[i]
            lat2_idx_corr = lat2_idxs[i]

            # Same for the lon values
            lon1_idx_corr = lon1_idxs[i]
            lon2_idx_corr = lon2_idxs[i]

            # extract the lon and lat values
            lon1_corr = lon1_corrs[i]
            lon2_corr = lon2_corrs[i]

            # Same for the lon values
            lat1_corr = lat1_corrs[i]
            lat2_corr = lat2_corrs[i]

            # If gridbox_corr is not None
            if gridbox_corr[i] is not None:
                # Loggging
                print("Calculating the correlations with a specific gridbox...")
                print("As defined by gridbox_corr = ", gridbox_corr[i])
                print("Variable is not psl")

                # print that we are only used nao matched members
                # for bootstrapping ts here
                print("Only using alt lag for bootstrapping ts here")
                print("------------------------------------------")

                if variable == "ua":
                    # Load the model data
                    # FIXME: NEEDS TO BE MODIFIED EACH TIME
                    fcst_ts_members = nal_fnc.load_data(
                        season=season,
                        forecast_range=forecast_range,
                        start_year=1961,
                        end_year=2014,
                        lag=4,
                        method=method_load,
                        region="global",
                        variable=variable,
                        data_dir="/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/test-sfcWind"
                    )

                    # # print the shape of the fcst_ts_members

                    # # limit to the first 47 members
                    # fcst_ts_members = fcst_ts_members[3:, :47, :, :]
                elif variable == "psl":
                    print("Using psl, so using the original method_load")
                    print(f"method load: {method_load}")
                    # Load the model data
                    fcst_ts_members = nal_fnc.load_data(
                        season=season,
                        forecast_range=forecast_range,
                        start_year=1961,
                        end_year=2014,
                        lag=4,
                        method=method_load,
                        region="global",
                        variable=variable,
                        data_dir="/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data"
                    )
                else:
                    fcst_ts_members = nal_fnc.load_data(
                        season=season,
                        forecast_range=forecast_range,
                        start_year=1961,
                        end_year=2014,
                        lag=4,
                        method=method_load,
                        region="global",
                        variable=variable,
                        data_dir="/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/"
                    )

                print("------------------------------------------")

                # if the variable is not psl
                if variable != "psl":
                    print("Bootstrapping where variable is not psl")


                    # Constrain the ts to the gridbox_corr
                    fcst1_ts = fcst1_ts[
                        :, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr
                    ]
                    obs_ts = obs_ts[
                        :, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr
                    ]


                    # constrain the corr
                    corr_gridbox = skill_maps["corr1"][lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]

                    # Same with the p values
                    p_gridbox = skill_maps["corr1_p"][lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]

                    # print the shape of cfst ts members
                    print(f"fcst_ts_members.shape = {fcst_ts_members.shape}")

                    if len(fcst_ts_members.shape) == 4:
                        # Constrain the members to the gridbox_corr
                        fcst_ts_members = fcst_ts_members[
                            :, :, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr
                        ]
                    elif len(fcst_ts_members.shape) == 5:
                        # Constrain the members to the gridbox_corr
                        fcst_ts_members = fcst_ts_members[
                            :, :, :, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr
                        ]

                    # Calculate the mean of both time series
                    fcst1_ts_mean = np.mean(fcst1_ts, axis=(1, 2))
                    obs_ts_mean = np.mean(obs_ts, axis=(1, 2))

                    # calculate the mean for the area averages
                    corr_gridbox_mean = np.mean(corr_gridbox, axis=(0, 1))
                    p_gridbox_mean = np.mean(p_gridbox, axis=(0, 1))

                    # if the fcts_ts_members has 4 dimensions
                    if len(fcst_ts_members.shape) == 4:
                        # same for the fcst_ts_members
                        fcst1_ts_members_mean = np.mean(fcst_ts_members, axis=(2, 3))
                    elif len(fcst_ts_members.shape) == 5:
                        # same for the fcst_ts_members
                        fcst1_ts_members_mean = np.mean(fcst_ts_members, axis=(2, 3, 4))
                    else:
                        AssertionError(
                            "fcst_ts_members does not have 4 or 5 dimensions, cannot calculate mean"
                        )

                    # temporal resampling here
                    # compare p to pvalue averaged over domain
                    # compare r to rvalue averaged over domain
                    # set up the ntimes
                    n_times = len(fcst1_ts_mean)

                    # print the values of fcst1
                    print(f"fcst1_ts_mean = {fcst1_ts_members_mean}")

                    # print the shape of the fcst1_ts_mean
                    print(f"fcst1_ts_members_mean.shape = {fcst1_ts_members_mean.shape}")
                    print(f"n_times = {n_times}")

                    # # assert that n_times is the same as the length of the obs_ts_mean
                    # assert np.shape(fcst1_ts_members_mean)[0] == n_times, "fcst_ts_members and obs_ts_mean have different lengths"

                    if np.shape(fcst1_ts_members_mean)[0] != n_times:
                        nens = np.shape(fcst1_ts_members_mean)[0]
                        
                        # flip the axes of the fcst_ts_members
                        fcst1_ts_members_mean = np.swapaxes(fcst1_ts_members_mean, 0, 1)
                    else:
                        nens = np.shape(fcst1_ts_members_mean)[1]

                    # assert that this has the same length as the obs_ts_mean
                    assert len(obs_ts_mean) == n_times, "obs_ts_mean and fcst1_ts_mean have different lengths"

                    # Set up the block length
                    block_length = 5

                    # Set up the nboot
                    nboot = 1000

                    # set up the arry
                    r_arr = np.empty([nboot])
                    r_arr_members = np.empty([nboot])

                    # Set up the number of blocks to be used
                    n_blocks = int(n_times / block_length)

                    # if the nblocks * block_length is less than n_times
                    # add one to the number of blocks
                    if n_blocks * block_length < n_times:
                        n_blocks = n_blocks + 1

                    # set up the indexes
                    # for the time - time needs to be the same for all forecasts and obs
                    index_time = range(n_times - block_length + 1)

                    # set up the index for the ensemble
                    index_ens = range(nens)

                    # print the shape of the fcst1_ts_members_mean
                    print(f"fcst1_ts_members_mean.shape = {fcst1_ts_members_mean.shape}")

                    print(f"values of fcst1_ts_members_mean = {fcst1_ts_members_mean}")

                    # print the nens shape
                    print(f"nens = {nens}")

                    # # if the first dimension is less than the second dimension
                    # # then flip the axes
                    # if fcst1_ts_members_mean.shape[0] < fcst1_ts_members_mean.shape[1]:
                    #     # print the shape of the fcst1_ts_members_mean
                    #     print(f"fcst1_ts_members_mean.shape = {fcst1_ts_members_mean.shape}")
                    #     print("Flipping the axes of fcst1_ts_members_mean")
                    #     # flip the axes of the fcst_ts_members
                    #     fcst1_ts_members_mean = np.swapaxes(fcst1_ts_members_mean, 0, 1)

                    # loop over the bootstraps
                    for iboot in tqdm(np.arange(nboot)):
                        if iboot == 0:
                            index_time_this = range(0, n_times, block_length)
                            index_ens_this = index_ens
                        else:
                            index_time_this = np.array(
                                [random.choice(index_time) for i in range(n_blocks)]
                            )

                            # ensemble index for tis
                            index_ens_this = np.array([
                                random.choice(index_ens) for i in index_ens
                                ]
                            )

                        # Create empty arrays to store the nao obs
                        fcst1_ts_boot = np.empty([n_times])
                        obs_ts_boot = np.empty([n_times])

                        # For the members version
                        fcst1_ts_members_boot = np.empty([nens, n_times])

                        # Set itime to 0
                        itime = 0

                        # loop over the time indexes
                        for i_this in index_time_this:
                            # Individual block index
                            index_block = np.arange(i_this, i_this + block_length)

                            # If the block index is greater than the number of times, then reduce the block index
                            index_block[(index_block > n_times - 1)] = (
                                index_block[(index_block > n_times - 1)] - n_times
                            )

                            # Select a subset of indices for the block
                            index_block = index_block[: min(block_length, n_times - itime)]

                            # loop over the block indices
                            for iblock in index_block:

                                # # print the value of fcst1_ts_members_mean
                                # print(f"fcst1_ts_members_mean.shape = {fcst1_ts_members_mean.shape}")
                                # # print the value of iblock
                                # print(f"iblock = {iblock}")
                                # # print the value of index_ens_this
                                # print(f"index_ens_this = {index_ens_this}")

                                # print(f"values of fcst1_ts_members_mean = {fcst1_ts_members_mean}")

                                # # print the vlues when subset to the iblock and index ens this
                                # print(f"values of fcst1_ts_members_mean[iblock, index_ens_this] = {fcst1_ts_members_mean[iblock, index_ens_this]}")
                                
                                # Assign the values to the arrays
                                fcst1_ts_boot[itime] = fcst1_ts_mean[iblock]
                                obs_ts_boot[itime] = obs_ts_mean[iblock]
                                fcst1_ts_members_boot[:, itime] = fcst1_ts_members_mean[iblock, index_ens_this]

                                # Increment itime
                                itime = itime + 1

                        # assert that there are non nans in either of the arrays
                        assert not np.isnan(fcst1_ts_boot).any(), "values in nao_boot are nan."
                        assert not np.isnan(obs_ts_boot).any(), "values in corr_var_ts_boot are nan."

                        # Calculate the correlation
                        r_arr[iboot] = pearsonr(fcst1_ts_boot, obs_ts_boot)[0]

                        # # print the shape of the fcst1_ts_members_boot
                        # print(f"fcst1_ts_members_boot.shape = {fcst1_ts_members_boot.shape}")

                        # # print the shape of the obs_ts_boot
                        # print(f"obs_ts_boot.shape = {obs_ts_boot.shape}")

                        # # print the values of the fcst1_ts_members_boot
                        # print(f"fcst1_ts_members_boot = {fcst1_ts_members_boot}")

                        # # print the values of the obs_ts_boot
                        # print(f"obs_ts_boot = {obs_ts_boot}")

                        # sys.exit()

                        # Compare the dimensions directly to determine which axis to average over
                        # To ensure that we average over members
                        if method_load == "nao_matched":
                            # # print("Averaging over members axis 0")
                            # print(f"fcst1_ts_members_boot.shape = {fcst1_ts_members_boot.shape}")
                            # print(f"obs_ts_boot.shape = {obs_ts_boot.shape}")
                            
                            # find the axes which has len 20
                            if fcst1_ts_members_boot.shape[0] == 20:
                                r_arr_members[iboot] = pearsonr(
                                    np.nanmean(fcst1_ts_members_boot, axis=0), obs_ts_boot
                                )[0]
                            elif fcst1_ts_members_boot.shape[1] == 20:
                                r_arr_members[iboot] = pearsonr(
                                    np.nanmean(fcst1_ts_members_boot, axis=1), obs_ts_boot
                                )[0]
                            else:
                                AssertionError(
                                    "fcst1_ts_members_boot does not have 20 members, cannot calculate mean"
                                )
                        elif fcst1_ts_members_boot.shape[0] > fcst1_ts_members_boot.shape[1]:
                            # print("Averaging over members axis 0")
                            # print(f"fcst1_ts_members_boot.shape = {fcst1_ts_members_boot.shape}")
                            # print(f"obs_ts_boot.shape = {obs_ts_boot.shape}")
                            r_arr_members[iboot] = pearsonr(np.nanmean(fcst1_ts_members_boot, axis=0), obs_ts_boot)[0]
                        elif fcst1_ts_members_boot.shape[0] < fcst1_ts_members_boot.shape[1]:
                            # print("Averaging over members axis 1")
                            # # print the shape of the fcst1_ts_members_boot
                            # print(f"fcst1_ts_members_boot.shape = {fcst1_ts_members_boot.shape}")
                            # print(f"obs_ts_boot.shape = {obs_ts_boot.shape}")
                            r_arr_members[iboot] = pearsonr(np.nanmean(fcst1_ts_members_boot, axis=1), obs_ts_boot)[0]
                        else:
                            AssertionError(
                                "fcst1_ts_members_boot does not have 2 dimensions, cannot calculate mean"
                            )
                    # Set up the corr
                    r = r_arr[0]
                    r_members = r_arr_members[0]

                    count_vals_r1 = np.sum(
                        i < 0.0 for i in r_arr
                    )  # Count the number of values less than 0

                    count_vals_r1_members = np.sum(
                        i < 0.0 for i in r_arr_members
                    )

                    # Calculate the p-value
                    p = count_vals_r1 / nboot
                    p_members = count_vals_r1_members / nboot

                    # print the computed r and p values
                    print(f"computed resampled (time only) r = {r}, p = {p}")
                    print(f"computed resampled (time + members) r = {r_members}, p = {p_members}")
                    print(f"gridbox avg r = {corr_gridbox_mean}, p = {p_gridbox_mean}")

                    # # Calculate the correlation between the two time series
                    # r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

                    # Show these values on the plot
                    ax.text(
                        0.05,
                        0.05,
                        f"r = {r_members:.2f}, p = {p_members:.2f}",
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        bbox=dict(facecolor="white", alpha=0.5),
                        fontsize=fontsize,
                    )

                    # Add the gridbox to the plot
                    ax.plot(
                        [lon1_corr, lon2_corr, lon2_corr, lon1_corr, lon1_corr],
                        [lat1_corr, lat1_corr, lat2_corr, lat2_corr, lat1_corr],
                        color="green",
                        linewidth=2,
                        transform=proj,
                    )
                else:
                    print("Bootstrapping where variable is psl")

                    if len(fcst_ts_members.shape) == 4:
                        # Constrain fcst ts members to gridbox corr N
                        fcst_ts_members_n = fcst_ts_members[
                            :, :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                        ]
                        fcst_ts_members_s = fcst_ts_members[
                            :, :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                        ]
                    elif len(fcst_ts_members.shape) == 5:
                        # Constrain fcst ts members to gridbox corr N
                        fcst_ts_members_n = fcst_ts_members[
                            :, :, :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                        ]
                        fcst_ts_members_s = fcst_ts_members[
                            :, :, :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                        ]

                    # Same for the observations
                    obs_ts_n = obs_ts[
                        :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                    ]
                    obs_ts_s = obs_ts[
                        :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                    ]

                    # Set up the bfcts1 ts n
                    fcst1_ts_n = fcst1_ts[
                        :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                    ]
                    fcst1_ts_s = fcst1_ts[
                        :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                    ]

                    # Calculate the mean of both time series
                    fcst1_ts_mean_n = np.mean(fcst1_ts_n, axis=(1, 2))
                    fcst1_ts_mean_s = np.mean(fcst1_ts_s, axis=(1, 2))
                    
                    # Calculate the mean of both time series for the obs
                    obs_ts_mean_n = np.mean(obs_ts_n, axis=(1, 2))
                    obs_ts_mean_s = np.mean(obs_ts_s, axis=(1, 2))

                    if len(fcst_ts_members_n.shape) == 4:
                        # Calculate the mean for the area averages
                        fcst_ts_members_mean_n = np.mean(fcst_ts_members_n, axis=(2, 3))
                        fcst_ts_members_mean_s = np.mean(fcst_ts_members_s, axis=(2, 3))
                    elif len(fcst_ts_members_n.shape) == 5:
                        # Calculate the mean for the area averages
                        fcst_ts_members_mean_n = np.mean(fcst_ts_members_n, axis=(2, 3, 4))
                        fcst_ts_members_mean_s = np.mean(fcst_ts_members_s, axis=(2, 3, 4))

                    # Set up the ntimes
                    n_times = len(fcst1_ts_mean_n)

                    # set up the nens
                    nens = np.shape(fcst_ts_members_mean_n)[0]

                    # prinbt the shape of the fcst_ts_members_mean
                    print(f"fcst_ts_members_mean_n.shape = {fcst_ts_members_mean_n.shape}")
                    print(f"fcst_ts_members_mean_s.shape = {fcst_ts_members_mean_s.shape}")

                    # print the n_times
                    print(f"n_times = {n_times}")

                    # print the nens
                    print(f"nens = {nens}")

                    # assert the len of the obs ts is obs_ts_mean
                    assert len(obs_ts_mean_n) == n_times, "obs_ts_mean and fcst1_ts_mean have different lengths"

                    # Set up the block length
                    block_length = 5

                    # Set up the nboot
                    nboot = 1000

                    # set up the arry
                    r_arr_n = np.empty([nboot])
                    r_arr_s = np.empty([nboot])

                    # Set up the r arr members
                    r_arr_members_n = np.empty([nboot])
                    r_arr_members_s = np.empty([nboot])

                    # if shape of fcst_ts_members_mean is not the same as n_times
                    if np.shape(fcst_ts_members_mean_n)[0] != n_times:
                        nens = np.shape(fcst_ts_members_mean_n)[0]

                        # flip the axes of the fcst_ts_members
                        fcst_ts_members_mean_n = np.swapaxes(fcst_ts_members_mean_n, 0, 1)
                        fcst_ts_members_mean_s = np.swapaxes(fcst_ts_members_mean_s, 0, 1)
                    else:
                        nens = np.shape(fcst_ts_members_mean_n)[1]

                    # Set up the number of blocks to be used
                    n_blocks = int(n_times / block_length)

                    # if the nblocks * block_length is less than n_times
                    # add one to the number of blocks
                    if n_blocks * block_length < n_times:
                        n_blocks = n_blocks + 1

                    # set up the indexes
                    # for the time - time needs to be the same for all forecasts and obs
                    index_time = range(n_times - block_length + 1)

                    # set up the index for the ensemble
                    index_ens = range(nens)

                    # print the shape of the fcst1_ts_members_mean
                    print(f"fcst_ts_members_mean_n.shape = {fcst_ts_members_mean_n.shape}")
                    print(f"fcst_ts_members_mean_s.shape = {fcst_ts_members_mean_s.shape}")

                    # print the values
                    print(f"fcst1_ts_mean_n values = {fcst1_ts_mean_n}")
                    print(f"fcst1_ts_mean_s values = {fcst1_ts_mean_s}")
                    
                    # printr the shape of fcst1_ts_mean
                    print(f"fcst1_ts_mean_n.shape = {fcst1_ts_mean_n.shape}")
                    print(f"fcst1_ts_mean_s.shape = {fcst1_ts_mean_s.shape}")

                    # print tyhe values of fcst_ts_members_mean
                    print(f"fcst_ts_members_mean_n values = {fcst_ts_members_mean_n}")
                    print(f"fcst_ts_members_mean_s values = {fcst_ts_members_mean_s}")

                    # print the values of the obs_ts_mean
                    print(f"obs_ts_mean_n values = {obs_ts_mean_n}")
                    print(f"obs_ts_mean_s values = {obs_ts_mean_s}")

                    # print the nens shape
                    print(f"nens = {nens}")

                    # loop over the bootstraps
                    for iboot in tqdm(np.arange(nboot)):
                        if iboot == 0:
                            index_time_this = range(0, n_times, block_length)
                            index_ens_this = index_ens
                        else:
                            index_time_this = np.array(
                                [random.choice(index_time) for i in range(n_blocks)]
                            )

                            # ensemble index for tis
                            index_ens_this = np.array([
                                random.choice(index_ens) for i in index_ens
                                ]
                            )

                        # Create empty arrays to store the nao obs
                        fcst1_ts_n_boot = np.empty([n_times])
                        fcst1_ts_s_boot = np.empty([n_times])
                        
                        obs_ts_n_boot = np.empty([n_times])
                        obs_ts_s_boot = np.empty([n_times])

                        # For the members version
                        fcst1_ts_members_n_boot = np.empty([nens, n_times])
                        fcst1_ts_members_s_boot = np.empty([nens, n_times])

                        # Set itime to 0
                        itime = 0

                        # loop over the time indexes
                        for i_this in index_time_this:
                            # Individual block index
                            index_block = np.arange(i_this, i_this + block_length)

                            # If the block index is greater than the number of times, then reduce the block index
                            index_block[(index_block > n_times - 1)] = (
                                index_block[(index_block > n_times - 1)] - n_times
                            )

                            # Select a subset of indices for the block
                            index_block = index_block[: min(block_length, n_times - itime)]

                            # loop over the block indices
                            for iblock in index_block:
                                # Assign the values to the arrays
                                fcst1_ts_n_boot[itime] = fcst1_ts_mean_n[iblock]
                                fcst1_ts_s_boot[itime] = fcst1_ts_mean_s[iblock]

                                # For the obs
                                obs_ts_n_boot[itime] = obs_ts_mean_n[iblock]
                                obs_ts_s_boot[itime] = obs_ts_mean_s[iblock]

                                # For the members
                                fcst1_ts_members_n_boot[:, itime] = fcst_ts_members_mean_n[iblock, index_ens_this]
                                fcst1_ts_members_s_boot[:, itime] = fcst_ts_members_mean_s[iblock, index_ens_this]

                                # Increment itime
                                itime = itime + 1

                        # assert that there are non nans in either of the arrays
                        # assert not np.isnan(fcst1_ts_n_boot).any(), "values in nao_boot are nan."
                        # assert not np.isnan(obs_ts_n_boot).any(), "values in corr_var_ts_boot are nan."

                        # Calculate the correlation
                        r_arr_n[iboot] = pearsonr(fcst1_ts_n_boot, obs_ts_n_boot)[0]
                        r_arr_s[iboot] = pearsonr(fcst1_ts_s_boot, obs_ts_s_boot)[0]

                        # For the members
                        r_arr_members_n[iboot] = pearsonr(np.mean(fcst1_ts_members_n_boot, axis=0), obs_ts_n_boot)[0]
                        r_arr_members_s[iboot] = pearsonr(np.mean(fcst1_ts_members_s_boot, axis=0), obs_ts_s_boot)[0]

                    # Set up the corr
                    r_n = r_arr_n[0]
                    r_s = r_arr_s[0]

                    # Set up the r values for the members
                    r_members_n = r_arr_members_n[0]
                    r_members_s = r_arr_members_s[0]

                    count_vals_r1_n = np.sum(
                        i < 0.0 for i in r_arr_n
                    )
                    count_vals_r1_s = np.sum(
                        i < 0.0 for i in r_arr_s
                    )

                    count_vals_r1_members_n = np.sum(
                        i < 0.0 for i in r_arr_members_n
                    )
                    count_vals_r1_members_s = np.sum(
                        i < 0.0 for i in r_arr_members_s
                    )

                    p_n = count_vals_r1_n / nboot
                    p_s = count_vals_r1_s / nboot

                    p_members_n = count_vals_r1_members_n / nboot
                    p_members_s = count_vals_r1_members_s / nboot

                    # Print trhe computed r and p values
                    print(f"computed resampled (time only) north box r = {r_n}, p = {p_n}")
                    print(f"computed resampled (time only) south box r = {r_s}, p = {p_s}")

                    print(f"computed resampled (time + members) north box r = {r_members_n}, p = {p_members_n}")
                    print(f"computed resampled (time + members) south box r = {r_members_s}, p = {p_members_s}")

                    # Show these values on the plot
                    ax.text(
                        0.05,
                        0.05,
                        "N: r = {r_n:.2f}, p = {p_n:.2f}\nS: r = {r_s:.2f}, p = {p_s:.2f}".format(
                            r_n=r_members_n, p_n=p_members_n, r_s=r_members_s, p_s=p_members_s
                        ),
                        transform=ax.transAxes,
                        va="bottom",
                        ha="left",
                        bbox=dict(facecolor="white", alpha=0.5),
                        fontsize=fontsize,
                    )

                    # Add the gridbox to the plot
                    ax.plot(
                        [lon1_corr_n, lon2_corr_n, lon2_corr_n, lon1_corr_n, lon1_corr_n],
                        [lat1_corr_n, lat1_corr_n, lat2_corr_n, lat2_corr_n, lat1_corr_n],
                        color="green",
                        linewidth=2,
                        transform=proj,
                    )

                    # Add the gridbox to the plot
                    ax.plot(
                        [lon1_corr_s, lon2_corr_s, lon2_corr_s, lon1_corr_s, lon1_corr_s],
                        [lat1_corr_s, lat1_corr_s, lat2_corr_s, lat2_corr_s, lat1_corr_s],
                        color="green",
                        linewidth=2,
                        transform=proj,
                    )
            elif gridbox_corr[i] is not None and variable == "psl":
                print("Calculating the correlations for NAO gridboxes...")
                print("For variable psl")

                # # Print the shape of fcst1_ts
                # print(f"fcst1_ts.shape = {fcst1_ts.shape}")

                # # Print the shape of obs_ts
                # print(f"obs_ts.shape = {obs_ts.shape}")

                # # Print the lat1_idx_corr_n
                # print(f"lat1_idx_corr_n = {lat1_idx_corr_n}")
                # print(f"lat2_idx_corr_n = {lat2_idx_corr_n}")
                # print(f"lon1_idx_corr_n = {lon1_idx_corr_n}")
                # print(f"lon2_idx_corr_n = {lon2_idx_corr_n}")

                # # Print the lat1_idx_corr_s
                # print(f"lat1_idx_corr_s = {lat1_idx_corr_s}")
                # print(f"lat2_idx_corr_s = {lat2_idx_corr_s}")
                # print(f"lon1_idx_corr_s = {lon1_idx_corr_s}")
                # print(f"lon2_idx_corr_s = {lon2_idx_corr_s}")

                # Constrain the ts to the gridbox_corr
                fcst1_ts_n = fcst1_ts[
                    :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                ]
                obs_ts_n = obs_ts[
                    :, lat1_idx_corr_n:lat2_idx_corr_n, lon1_idx_corr_n:lon2_idx_corr_n
                ]

                # Constrain the ts to the gridbox_corr
                fcst1_ts_s = fcst1_ts[
                    :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                ]
                obs_ts_s = obs_ts[
                    :, lat1_idx_corr_s:lat2_idx_corr_s, lon1_idx_corr_s:lon2_idx_corr_s
                ]

                # Calculate the mean of both time series
                fcst1_ts_mean_n = np.mean(fcst1_ts_n, axis=(1, 2))
                obs_ts_mean_n = np.mean(obs_ts_n, axis=(1, 2))

                # Calculate the mean of both time series
                fcst1_ts_mean_s = np.mean(fcst1_ts_s, axis=(1, 2))
                obs_ts_mean_s = np.mean(obs_ts_s, axis=(1, 2))

                # Calculate the correlation between the two time series
                r_n, p_n = pearsonr(fcst1_ts_mean_n, obs_ts_mean_n)
                r_s, p_s = pearsonr(fcst1_ts_mean_s, obs_ts_mean_s)

                # Print the values
                print(f"r_n = {r_n}, p_n = {p_n}")
                print(f"r_s = {r_s}, p_s = {p_s}")

                # Show these values on the plot
                ax.text(
                    0.05,
                    0.05,
                    "N: r = {r_n:.2f}, p = {p_n:.2f}\nS: r = {r_s:.2f}, p = {p_s:.2f}".format(
                        r_n=r_n, p_n=p_n, r_s=r_s, p_s=p_s
                    ),
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    bbox=dict(facecolor="white", alpha=0.5),
                    fontsize=fontsize,
                )

                # Add the gridbox to the plot
                ax.plot(
                    [lon1_corr_n, lon2_corr_n, lon2_corr_n, lon1_corr_n, lon1_corr_n],
                    [lat1_corr_n, lat1_corr_n, lat2_corr_n, lat2_corr_n, lat1_corr_n],
                    color="green",
                    linewidth=2,
                    transform=proj,
                )

                # Add the gridbox to the plot
                ax.plot(
                    [lon1_corr_s, lon2_corr_s, lon2_corr_s, lon1_corr_s, lon1_corr_s],
                    [lat1_corr_s, lat1_corr_s, lat2_corr_s, lat2_corr_s, lat1_corr_s],
                    color="green",
                    linewidth=2,
                    transform=proj,
                )

            # If any of the corr1 values are NaNs
            # then set the p values to NaNs at the same locations
            corr1_p[np.isnan(corr)] = np.nan

            # If any of the corr1_p values are greater or less than the sig_threshold
            # then set the corr1 values to NaNs at the same locations
            corr1_p[(corr1_p > sig_threshold) & (corr1_p < 1 - sig_threshold)] = np.nan

            if nboot_str != "nboot_1":
                # plot the p-values
                ax.contourf(
                    lons, lats, corr1_p, hatches=[".."], alpha=0.0, transform=proj
                )
            else:
                print("Not plotting p-values for nboot_1")

            # Add a text box with the axis label
            ax.text(
                0.95,
                0.05,
                f"{axis_labels[i]}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            # Extract the first part of the key
            key = key[0]

            # If this is tas
            if key == "tas":
                # Set the variable
                var_name = "Temperature"
            elif key == "pr":
                # Set the variable
                var_name = "Precipitation"
            elif key == "psl":
                # Set the variable
                var_name = "Sea level pressure"
            elif key == "sfcWind":
                # Set the variable
                var_name = "10m wind speed"
            elif key == "rsds":
                # Set the variable
                var_name = "Solar irradiance"
            elif key == "ua":
                # Set the variable
                var_name = "U850"
            else:
                # Print the key
                print(f"key = {key}")
                AssertionError("Variable not recognised!")

            # Add a textboc with the variable name in the top left
            ax.text(
                0.05,
                0.95,
                f"{var_name}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=fontsize,
            )

            if methods is not None:
                # If the list contains alt_lag
                if "alt_lag" in methods:
                    # Replace alt_lag with lagged, where alt_lag is found
                    methods = [
                        method.replace("alt_lag", "Lagged") for method in methods
                    ]
                elif "nao_matched" in methods:
                    # Replace nao_matched with NAO-matched, where nao_matched is found
                    methods = [
                        method.replace("nao_matched", "NAO-matched")
                        for method in methods
                    ]
                elif "raw" in methods:
                    # Replace raw with Raw, where raw is found
                    methods = [method.replace("raw", "Raw") for method in methods]
                else:
                    # AssertionError
                    AssertionError("Method not recognised!")

                if short_period is False:
                    if type(nens1) != tuple:
                        # Include the method in the top right of the figure
                        ax.text(
                            0.95,
                            0.95,
                            f"{methods[i]} ({nens1})",
                            transform=ax.transAxes,
                            va="top",
                            ha="right",
                            bbox=dict(facecolor="white", alpha=0.5),
                            fontsize=fontsize,
                        )
                    else:
                        # Include the method in the top right of the figure
                        ax.text(
                            0.95,
                            0.95,
                            f"{methods[i]} {nens1}",
                            transform=ax.transAxes,
                            va="top",
                            ha="right",
                            bbox=dict(facecolor="white", alpha=0.5),
                            fontsize=fontsize,
                        )
                else:
                    # Include the method in the top right of the figure
                    ax.text(
                        0.95,
                        0.95,
                        f"{methods[i]} ({nens1})\nShort period corr",
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        bbox=dict(facecolor="white", alpha=0.5),
                        fontsize=fontsize,
                    )
            else:
                # Include the number of ensemble members in the top right of the figure
                ax.text(
                    0.95,
                    0.95,
                    f"n = {nens1}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    bbox=dict(facecolor="white", alpha=0.5),
                    fontsize=fontsize,
                )

            # Add the contourf object to the list
            cf_list.append(cf)

            # Add the axes to the list
            axes.append(ax)

    # if plot_winter_nodes is True
    if plot_winter_nodes is True:
        print("Plotting the winter nodes")

        # Extract the lats
        lat1_n, lat2_n = winter_n_gridbox_corr["lat1"], winter_n_gridbox_corr["lat2"]
        lat1_s, lat2_s = winter_s_gridbox_corr["lat1"], winter_s_gridbox_corr["lat2"]

        # Extract the lons
        lon1_n, lon2_n = winter_n_gridbox_corr["lon1"], winter_n_gridbox_corr["lon2"]
        lon1_s, lon2_s = winter_s_gridbox_corr["lon1"], winter_s_gridbox_corr["lon2"]

        # Add the winter nodes to the plot
        for ax in axes:
            # Add the winter nodes to the plot
            ax.plot(
                [lon1_n, lon2_n, lon2_n, lon1_n, lon1_n],
                [lat1_n, lat1_n, lat2_n, lat2_n, lat1_n],
                color="blue",
                linewidth=1,
                transform=proj,
            )

            # Add the winter nodes to the plot
            ax.plot(
                [lon1_s, lon2_s, lon2_s, lon1_s, lon1_s],
                [lat1_s, lat1_s, lat2_s, lat2_s, lat1_s],
                color="blue",
                linewidth=1,
                transform=proj,
            )

    # # Remove content from the 4th axis
    # axs[1, 1].remove()

    # If nrows is 3 and len(bs_skill_maps.keys()) is 5
    # then remove the 5th axis
    # if nrows == 3 and len(bs_skill_maps.keys()) == 5:
    #     print("Removing the 5th axis...")
    #     # Remove the 5th axis
    #     axs[2, 1].remove()

    # print the fig
    print("fig: ", fig)

    # print axes
    print("axes: ", axes)

    # print the tyupe of axes
    print("type(axes): ", type(axes))

    # set up a tight layout
    # plt.tight_layout()

    # # # Adjust the whitespace
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Add a colorbar
    cbar = fig.colorbar(
        cf_list[0], ax=axes, orientation="horizontal", pad=0.05, shrink=0.8
    )

    # set the ticks
    # Set the ticks manually
    ticks = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0])

    # set the cbar labels
    cbar.set_ticks(ticks)
    # Set the label for the colorbar
    cbar.set_label("correlation coefficient", fontsize=12)

    # Set up the path for saving the figure
    plots_dir = "/home/users/benhutch/skill-maps/plots"

    # Set up the current date
    current_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set up the figure name
    fig_name = f"different_variables_corr_{start_year}_{end_year}_{season}_{forecast_range}_{current_date}.pdf"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")

    # # specify a tight layout
    # plt.tight_layout()

    # Show the figure
    plt.show()


# Write a new function which shows a gridbox
def show_gridbox(
    grid: dict,
    grid_name: str,
    figsize_x: int = 10,
    figsize_y: int = 12,
):
    """
    Show a gridbox on a map.

    Inputs:
    -------

    grid: dict
        Dictionary containing the gridbox to show.
        e.g. grid = {"lon1": lon1, "lon2": lon2, "lat1": lat1, "lat2": lat2}

    grid_name: str
        Name of the gridbox to show.
        e.g. "gridbox1"

    figsize_x: int
        Width of the figure in inches.
        e.g. default is 10

    figsize_y: int
        Height of the figure in inches.
        e.g. default is 12

    Outputs:
    --------

    None
    """

    # Set up the projection
    proj = ccrs.PlateCarree(central_longitude=0)

    # Set up the figure
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    # Set up the axes
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Include coastlines
    ax.coastlines()
    ax.stock_img()

    # Extract the lons and lats from the grid dictionary
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Add the gridbox to the plot
    ax.plot(
        [lon1, lon2, lon2, lon1, lon1],
        [lat1, lat1, lat2, lat2, lat1],
        color="green",
        linewidth=2,
        transform=proj,
    )

    # Set up the region extent
    lon_min = lon1 - 20
    lon_max = lon2 + 20

    # Set up the lat extent
    lat_min = lat1 - 20
    lat_max = lat2 + 20

    # Set the extent of the plot
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Add a text box with the gridbox name
    ax.text(
        0.05,
        0.05,
        f"{grid_name}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=14,
    )

    # Show the figure
    plt.show()

    return None


# Write a function for loading ts data for a specific gridbox
def load_ts_data(
    data: np.ndarray,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    lag: int,
    gridbox: dict,
    gridbox_name: str,
    variable: str,
    alt_lag: str = None,
    region: str = "global",
    level: str = None,
):
    """
    Load time series data for a specific gridbox.

    Inputs:
    -------

    data: np.ndarray
        Array containing the data to process.
        e.g. data = np.load("path/to/data.npy")

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    start_year: int
        Start year to process.
        e.g. 1993

    end_year: int
        End year to process.

    lag: int
        Lag to process.
        e.g. 0

    gridbox: dict
        Dictionary containing the gridbox to process.
        e.g. gridbox = {"lon1": lon1, "lon2": lon2, "lat1": lat1, "lat2": lat2}

    gridbox_name: str
        Name of the gridbox to process.
        e.g. "gridbox1"

    variable: str
        Variable to process.
        e.g. "tas"

    alt_lag: str
        Alternative lag to process.
        e.g. "nao_matched"

    region: str
        Region to process.
        e.g. default is "global"

    Outputs:
    --------

    ts: np.ndarray
        Array containing the time series data for the gridbox.
    """

    # Set up the mdi
    mdi = -9999.0

    # Set up the dictionary
    ts_dict = {
        "obs_ts": [],
        "obs_ts_short": [],
        "fcst_ts_members": [],
        "fcst_ts_members_short": [],
        "fcst_ts_min": [],
        "fcst_ts_min_short": [],
        "fcst_ts_max": [],
        "fcst_ts_max_short": [],
        "fcst_ts_mean": [],
        "fcst_ts_mean_short": [],
        "init_years": [],
        "init_years_short": [],
        "valid_years": [],
        "valid_years_short": [],
        "nens": mdi,
        "corr": mdi,
        "corr_short": mdi,
        "p": mdi,
        "p_short": mdi,
        "rpc": mdi,
        "rpc_short": mdi,
        "rps": mdi,
        "rps_short": mdi,
        "season": season,
        "forecast_range": forecast_range,
        "start_year": start_year,
        "end_year": end_year,
        "lag": lag,
        "variable": variable,
        "gridbox": gridbox,
        "gridbox_name": gridbox_name,
        "alt_lag": alt_lag,
    }

    # Extract the lats and lons from the gridbox
    lon1, lon2 = gridbox["lon1"], gridbox["lon2"]

    # Extract the lats and lons from the gridbox
    lat1, lat2 = gridbox["lat1"], gridbox["lat2"]

    # Set up the years depending
    if alt_lag in ["nao_matched", "alt_lag"]:
        # Set up the years
        years = np.arange(start_year + lag - 1, end_year + 1)
    elif forecast_range == "2-9" and season not in ["DJFM", "DJF", "ONDJFM"]:
        # Set up the years
        years = np.arange(start_year, end_year)
    else:
        # Set up the years
        years = np.arange(start_year, end_year + 1)

    # Append the years to the ts_dict
    ts_dict["init_years"] = years

    # Append the init years short
    ts_dict["init_years_short"] = years[: -10 + 1]  # final year should be 2005

    # Set up the lats and lons
    lats = np.arange(-90, 90, 2.5)
    lons = np.arange(-180, 180, 2.5)

    # If the forecast range is a single digit
    if forecast_range.isdigit():
        forecast_range_obs = "1"
    else:
        forecast_range_obs = forecast_range

    if level is None:
        # Process the observed data
        obs_anoms = fnc.read_obs(
            variable=variable,
            region=region,
            forecast_range=forecast_range_obs,
            season=season,
            observations_path=nao_match_fnc.find_obs_path(match_var=variable),
            start_year=1960,
            end_year=2023,
        )

        # Process the obs anoms short
        obs_anoms_short = fnc.read_obs(
            variable=variable,
            region=region,
            forecast_range=forecast_range_obs,
            season=season,
            observations_path=nao_match_fnc.find_obs_path(match_var=variable),
            start_year=1960,
            end_year=2014,  # similar to Doug's
        )
    else:
        # Process the observed data
        obs_anoms = fnc.read_obs(
            variable=variable,
            region=region,
            forecast_range=forecast_range_obs,
            season=season,
            observations_path=nao_match_fnc.find_obs_path(match_var=variable),
            start_year=1960,
            end_year=2023,
            level=level,
        )

        # Process the obs anoms short
        obs_anoms_short = fnc.read_obs(
            variable=variable,
            region=region,
            forecast_range=forecast_range_obs,
            season=season,
            observations_path=nao_match_fnc.find_obs_path(match_var=variable),
            start_year=1960,
            end_year=2014,  # similar to Doug's
            level=level,
        )

    # Set up the lats and lons
    obs_lats = obs_anoms.lat.values
    obs_lons = obs_anoms.lon.values

    # Assert that the lats and lons are equal
    assert np.array_equal(obs_lats, lats), "lats are not equal!"

    # Assert that the lons are equal
    assert np.array_equal(obs_lons, lons), "lons are not equal!"

    # Find the indices of the lats and lons which correspond to the gridbox
    lat1_idx = np.argmin(np.abs(lats - lat1))
    lat2_idx = np.argmin(np.abs(lats - lat2))

    # Find the indices of the lons which correspond to the gridbox
    lon1_idx = np.argmin(np.abs(lons - lon1))
    lon2_idx = np.argmin(np.abs(lons - lon2))

    # Process the data
    if data.ndim == 5 and alt_lag not in ["nao_matched", "alt_lag"]:
        print("Processing the raw data")

        # If forecast range is 2-9
        if forecast_range == "2-9":
            # Set up the raw first and last years
            raw_first_year = int(start_year) + 5
            raw_last_year = int(end_year) + 5
        elif forecast_range == "2-5":
            # Set up the raw first and last years
            raw_first_year = int(start_year) + 3
            raw_last_year = int(end_year) + 3
        elif forecast_range == "2-3":
            # Set up the raw first and last years
            raw_first_year = int(start_year) + 2
            raw_last_year = int(end_year) + 2
        elif forecast_range == "2":
            # Set up the raw first and last years
            raw_first_year = int(start_year) + 1
            raw_last_year = int(end_year) + 1
        elif forecast_range == "1":
            # Set up the raw first and last years
            raw_first_year = int(start_year)
            raw_last_year = int(end_year)
        else:
            print("Forecast range not recognised")

        # If the season is not DJFM
        if season not in ["DJFM", "DJF", "ONDJFM"]:
            # Add 1 to the raw first and last years
            raw_first_year += 1
            raw_last_year += 1

        # Print the raw first and last years
        print(f"raw_first_year = {raw_first_year}")
        print(f"raw_last_year = {raw_last_year}")

        # If the forecast range is not 2-9
        if forecast_range != "2-9":
            # Set up the valid years
            valid_years = np.arange(raw_first_year, raw_last_year + 1)
        elif forecast_range == "2-9" and alt_lag is False:
            # Set up the valid years
            valid_years = np.arange(raw_first_year, raw_last_year + 1)
        else:
            # Set up the valid years
            valid_years = np.arange(raw_first_year, raw_last_year)

        # Append the valid years to the ts_dict
        ts_dict["valid_years"] = valid_years

        # Append the valid years short
        ts_dict["valid_years_short"] = valid_years[
            : -10 + 1
        ]  # final year should be 2013/14

        if forecast_range != "2-9":
            # Constrain the obs_anoms to the valid years
            obs_anoms = obs_anoms.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31")
            )

            # Constrain the obs_anoms short to the valid years
            obs_anoms_short = obs_anoms_short.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year - 10 + 1}-12-31")
            )
        elif forecast_range == "2-9" and alt_lag is False:
            # Constrain the obs_anoms to the valid years
            obs_anoms = obs_anoms.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31")
            )

            # Constrain the obs_anoms short to the valid years
            obs_anoms_short = obs_anoms_short.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year - 10 + 1}-12-31")
            )

        else:
            # Constrain the obs_anoms to the valid years
            obs_anoms = obs_anoms.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year - 1}-12-31")
            )

            # Constrain the obs_anoms short to the valid years
            obs_anoms_short = obs_anoms_short.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year - 10}-12-31")
            )

        # Constrain the obs_anoms to the gridbox
        obs_anoms = obs_anoms.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
            dim=["lat", "lon"]
        )

        # Constrain the obs_anoms short to the gridbox
        obs_anoms_short = obs_anoms_short.sel(
            lat=slice(lat1, lat2), lon=slice(lon1, lon2)
        ).mean(dim=["lat", "lon"])

        # Loop over the years
        for year in obs_anoms.time.dt.year.values:
            # Extract the obs_anoms for this year
            obs_anoms_year = obs_anoms.sel(time=f"{year}")

            # If there are any NaNs in the obs_anoms_year
            if np.isnan(obs_anoms_year).any():
                print(f"NaNs found in obs_anoms_year for year {year}")
                if np.isnan(obs_anoms_year).all():
                    print(f"All values are NaNs for year {year}")
                    print(f"removing year {year} from the years array")
                    obs_anoms = obs_anoms.drop_sel(time=f"{year}")

        # Print the shape of the obs_anoms
        print(f"obs_anoms.shape = {obs_anoms.shape}")

        # Loop over the years for obs anoms short
        for year in obs_anoms_short.time.dt.year.values:
            # Extract the obs_anoms for this year
            obs_anoms_year = obs_anoms_short.sel(time=f"{year}")

            # If there are any NaNs in the obs_anoms_year
            if np.isnan(obs_anoms_year).any():
                print(f"NaNs found in obs_anoms_year for year {year}")
                if np.isnan(obs_anoms_year).all():
                    print(f"All values are NaNs for year {year}")
                    print(f"removing year {year} from the years array")
                    obs_anoms_short = obs_anoms_short.drop_sel(time=f"{year}")

        # Extract the obs_anoms as its values
        obs_ts = obs_anoms.values

        # Extract the obs_anoms short as its values
        obs_ts_short = obs_anoms_short.values

        # print the variable
        print(f"variable = {variable}")

        if variable == "psl":
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts / 100

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short / 100
        elif variable == "pr":
            # Convert the obs_ts to mm/day (from m/day)
            obs_ts = obs_ts * 1000

            # Convert the obs_ts short to mm/day (from m/day)
            obs_ts_short = obs_ts_short * 1000

            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short
        elif variable in ["rsds", "ssrd"]:
            # Conver the obs from J/m^2 to W/m^2
            ts_dict["obs_ts"] = obs_ts / 86400

            # Conver the obs from J/m^2 to W/m^2
            ts_dict["obs_ts_short"] = obs_ts_short / 86400
        else:
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short

        # Swap the axes of the data
        data = np.swapaxes(data, 0, 1)

        # Print the shape of the data
        print(f"data.shape = {data.shape}")

        # If the third axis has size > 1
        if data.shape[2] > 1:
            print("Taking the mean over the valid forecvast years:", forecast_range)
            # Calculate the mean of the data
            # Extract the second number in forecast_range
            forecast_range_number = int(forecast_range.split("-")[1])

            # Calculate the mean of the data
            data = data[:, :, : forecast_range_number - 1, :, :].mean(axis=2)
        elif data.shape[2] == 1:
            print("Data.shape[2] == 1")
            print("Squeezing the data")
            # Squeeze the data
            data = np.squeeze(data)

        # If years 2-9 and not winter (i.e. not shifted back)
        if forecast_range == "2-9" and season not in ["DJFM", "DJF", "ONDJFM"]:
            # Remove the final time step
            data = data[:, :-1, :, :]

        # Print the len of valid years
        print(f"len(valid_years) = {len(valid_years)}")

        # Assert that the data shape is as expected
        assert data.shape[1] == len(valid_years), "Data shape not as expected!"

        # Assert that the shape of the lats
        assert data.shape[2] == len(lats), "lats shape not as expected!"

        # Assert that the shape of the lons
        assert data.shape[3] == len(lons), "lons shape not as expected!"

        # Print the shape of the modified data
        print(f"modified model data.shape = {data.shape}")

        # Constrain the data to the gridbox
        data = data[:, :, lat1_idx:lat2_idx, lon1_idx:lon2_idx].mean(axis=(2, 3))

        # Print the shape of the data
        print(f"model data members.shape = {data.shape}")

        # Constrain the data to the short period
        data_short = data[:, : -10 + 1]  # final year should be 2005

        # Calculate the 5% lowest interval
        ci_lower = np.percentile(data, 5, axis=0)

        # Calculate the 5% lowest interval for the short period
        ci_lower_short = np.percentile(data_short, 5, axis=0)

        # Calculate the 95% highest interval
        ci_upper = np.percentile(data, 95, axis=0)

        # Calculate the 95% highest interval for the short period
        ci_upper_short = np.percentile(data_short, 95, axis=0)

        # Calculate the mean of the data
        data_mean = np.mean(data, axis=0)

        # Calculate the mean of the data for the short period
        data_mean_short = np.mean(data_short, axis=0)

        if variable == "psl":
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data / 100

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short / 100

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean / 100

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short / 100

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower / 100
            ts_dict["fcst_ts_max"] = ci_upper / 100

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short / 100

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_max_short"] = ci_upper_short / 100
        elif variable == "pr":
            # Convert the data to mm/day (from m/day)
            data = data * 86400

            # Convert the data short to mm/day (from m/day)
            data_short = data_short * 86400

            # Convert the data_mean to mm/day (from m/day)
            data_mean = data_mean * 86400

            # Convert the data_mean short to mm/day (from m/day)
            data_mean_short = data_mean_short * 86400

            # Convert the ci_lower to mm/day (from m/day)
            ci_lower = ci_lower * 86400

            # Convert the ci_upper to mm/day (from m/day)
            ci_upper = ci_upper * 86400

            # Convert the ci_upper to mm/day (from m/day)
            ci_upper = ci_upper * 86400

            # Convert the ci_lower short to mm/day (from m/day)
            ci_lower_short = ci_lower_short * 86400

            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short
            ts_dict["fcst_ts_max_short"] = ci_upper_short
        else:
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short
            ts_dict["fcst_ts_max_short"] = ci_upper_short

        # Calculate the correlation between the obs_ts and fcst_ts_mean
        corr, p = pearsonr(data_mean, obs_ts)

        # Calculate the correlation between the obs_ts and fcst_ts_mean short
        corr_short, p_short = pearsonr(data_mean_short, obs_ts_short)

        # Append the corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

        # Append the corr short and p short to the ts_dict
        ts_dict["corr_short"] = corr_short
        ts_dict["p_short"] = p_short

        # Calculate the standard deviation of the forecast time series
        sig_f = np.std(data_mean)
        sig_o = np.std(obs_ts)

        # Calculate th stdev of the members
        sig_f_short = np.std(data_mean_short)
        sig_o_short = np.std(obs_ts_short)

        # Calculate th stdev of the members
        sig_f_members = np.std(data)

        # Calculate the stdev of the short members
        sig_f_members_short = np.std(data_short)

        # Calculate the rpc
        rpc = corr * (sig_f / sig_f_members)

        # Calculate the rpc for the short period
        rpc_short = corr_short * (sig_f_short / sig_f_members_short)

        # Calculate the rps
        rps = rpc * (sig_o / sig_f_members)

        # Calculate the rps for the short period
        rps_short = rpc_short * (sig_o_short / sig_f_members_short)

        # Append the rpc and rps to the ts_dict
        ts_dict["rpc"] = np.abs(rpc)
        ts_dict["rps"] = np.abs(rps)

        # Append the rpc short and rps short to the ts_dict
        ts_dict["rpc_short"] = np.abs(rpc_short)
        ts_dict["rps_short"] = np.abs(rps_short)

        # Append the nens to the ts_dict
        ts_dict["nens"] = data.shape[0]

    elif data.ndim == 4 and alt_lag in ["nao_matched", "alt_lag"]:
        print("Processing the alt_lag data")

        # Set up the alt_lag first and last years
        alt_lag_first_year = int(start_year) + lag - 1
        alt_lag_last_year = int(end_year)

        # Set up the first and last years
        if forecast_range == "2-9":
            # Set up the raw first and last years
            alt_lag_first_year = alt_lag_first_year + 5
            alt_lag_last_year = alt_lag_last_year + 5
        elif forecast_range == "2-5":
            # Set up the first and last years
            alt_lag_first_year = alt_lag_first_year + 3
            alt_lag_last_year = alt_lag_last_year + 3
        elif forecast_range == "2-3":
            # Set up the first and last years
            alt_lag_first_year = alt_lag_first_year + 2
            alt_lag_last_year = alt_lag_last_year + 2
        elif forecast_range == "2":
            # Set up the first and last years
            alt_lag_first_year = alt_lag_first_year + 1
            alt_lag_last_year = alt_lag_last_year + 1
        elif forecast_range == "1":
            # Set up the first and last years
            alt_lag_first_year = alt_lag_first_year
            alt_lag_last_year = alt_lag_last_year
        else:
            raise ValueError("Forecast range not recognised")

        # Print the alt_lag first and last years
        print(f"alt_lag_first_year = {alt_lag_first_year}")
        print(f"alt_lag_last_year = {alt_lag_last_year}")

        # Set up the valid years
        valid_years = np.arange(alt_lag_first_year, alt_lag_last_year + 1)

        # Append the valid years to the ts_dict
        ts_dict["valid_years"] = valid_years

        # Set up the valid years short
        valid_years_short = np.arange(alt_lag_first_year, alt_lag_last_year - 10 + 1)

        # APpend the valid years short to the ts_dict
        ts_dict["valid_years_short"] = valid_years_short

        # Constrain the obs_anoms to the valid years
        obs_anoms = obs_anoms.sel(
            time=slice(f"{alt_lag_first_year}-01-01", f"{alt_lag_last_year}-12-31")
        )

        # Constrain the obs_anoms to the gridbox
        obs_anoms = obs_anoms.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
            dim=["lat", "lon"]
        )

        # Constrain the obs_anoms short to the valid years
        obs_anoms_short = obs_anoms_short.sel(
            time=slice(
                f"{alt_lag_first_year}-01-01", f"{alt_lag_last_year - 10 + 1}-12-31"
            )
        )

        # Constrain the obs_anoms short to the gridbox
        obs_anoms_short = obs_anoms_short.sel(
            lat=slice(lat1, lat2), lon=slice(lon1, lon2)
        ).mean(dim=["lat", "lon"])

        # Loop over the years
        for year in obs_anoms.time.dt.year.values:
            # Extract the obs_anoms for this year
            obs_anoms_year = obs_anoms.sel(time=f"{year}")

            # If there are any NaNs in the obs_anoms_year
            if np.isnan(obs_anoms_year).any():
                print(f"NaNs found in obs_anoms_year for year {year}")
                if np.isnan(obs_anoms_year).all():
                    print(f"All values are NaNs for year {year}")
                    print(f"removing year {year} from the years array")
                    obs_anoms = obs_anoms.drop_sel(time=f"{year}")

        # Loop over the years for obs anoms short
        for year in obs_anoms_short.time.dt.year.values:
            # Extract the obs_anoms for this year
            obs_anoms_year = obs_anoms_short.sel(time=f"{year}")

            # If there are any NaNs in the obs_anoms_year
            if np.isnan(obs_anoms_year).any():
                print(f"NaNs found in obs_anoms_year for year {year}")
                if np.isnan(obs_anoms_year).all():
                    print(f"All values are NaNs for year {year}")
                    print(f"removing year {year} from the years array")
                    obs_anoms_short = obs_anoms_short.drop_sel(time=f"{year}")

        # Print the shape of the obs_anoms
        print(f"obs_anoms.shape = {obs_anoms.shape}")

        # Extract the obs_anoms as its values
        obs_ts = obs_anoms.values

        # Extract the obs_anoms short as its values
        obs_ts_short = obs_anoms_short.values

        if variable == "psl":
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts / 100

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short / 100
        elif variable == "pr":
            # Convert the obs_ts to mm/day (from m/day)
            obs_ts = obs_ts * 1000

            # Convert the obs_ts short to mm/day (from m/day)
            obs_ts_short = obs_ts_short * 1000

            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short
        elif variable in ["rsds", "ssrd"]:
            # Conver the obs from J/m^2 to W/m^2
            ts_dict["obs_ts"] = obs_ts / 86400

            # Conver the obs from J/m^2 to W/m^2
            ts_dict["obs_ts_short"] = obs_ts_short / 86400
        else:
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

            # Append the obs_ts short to the ts_dict
            ts_dict["obs_ts_short"] = obs_ts_short

        if alt_lag != "nao_matched":
            # Swap the axes of the data
            data = np.swapaxes(data, 0, 1)

        # Print the shape of the data
        print(f"data.shape = {data.shape}")
        print(f"obs_anoms.shape = {obs_anoms.shape}")
        print(f"len valid_years = {len(valid_years)}")

        # Assert that the data shape is as expected
        assert data.shape[1] == len(valid_years), "Data shape not as expected!"

        # Assert that the shape of the short data is as expected
        assert data.shape[1] == len(valid_years), "Data shape not as expected!"

        # Assert that the shape of the lats
        assert data.shape[2] == len(lats), "lats shape not as expected!"

        # Assert that the shape of the lons
        assert data.shape[3] == len(lons), "lons shape not as expected!"

        # Print the shape of the modified data
        print(f"modified model data.shape = {data.shape}")

        # Constrain the data to the gridbox
        data = data[:, :, lat1_idx:lat2_idx, lon1_idx:lon2_idx].mean(axis=(2, 3))

        # print the mean of the data
        print(f"mean of the data = {np.mean(data)}")
        # print the spread of the data
        print(f"spread of the data = {np.std(data)}")

        # print the min of the data
        print(f"min of the data = {np.min(data)}")

        # print the max of the data
        print(f"max of the data = {np.max(data)}")

        # Constrasin the short data to the gridbox
        data_short = data[:, : -10 + 1]  # final year should be 2005

        # Print the shape of the data
        print(f"model data members.shape = {data.shape}")

        # # loop over the members
        # for member in range(data.shape[0]):
        #     # print the member index
        #     print(f"member = {member}")

        #     # print the member spread
        #     print(f"spread of the member = {np.std(data[member])}")

        #     # print the min and max
        #     print(f"min of the member = {np.min(data[member])}")

        #     # print the max of the member
        #     print(f"max of the member = {np.max(data[member])}")

        # Calculate the mean of the data
        data_mean = np.mean(data, axis=0)

        # print the mean of the data
        print(f"mean of the data_mean = {np.mean(data_mean)}")

        # print the spread of the data
        print(f"spread of the data_mean = {np.std(data_mean)}")

        # print the min of the data
        print(f"min of the data_mean = {np.min(data_mean)}")

        # print the max of the data
        print(f"max of the data_mean = {np.max(data_mean)}")

        # Calculate the mean of the short data
        data_mean_short = np.mean(data_short, axis=0)

        # Calculate the 5% lowest interval
        ci_lower = np.percentile(data, 5, axis=0)

        # Calculate the 5% lowest interval for the short period
        ci_lower_short = np.percentile(data_short, 5, axis=0)

        # Calculate the 95% highest interval
        ci_upper = np.percentile(data, 95, axis=0)

        # Calculate the 95% highest interval for the short period
        ci_upper_short = np.percentile(data_short, 95, axis=0)

        if variable == "psl":
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data / 100

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short / 100

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean / 100

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short / 100

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower / 100
            ts_dict["fcst_ts_max"] = ci_upper / 100

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short / 100
            ts_dict["fcst_ts_max_short"] = ci_upper_short / 100
        elif variable == "pr":
            # Convert the data to mm/day (from m/day)
            data = data * 86400

            # Convert the data short to mm/day (from m/day)
            data_short = data_short * 86400

            # Convert the data_mean to mm/day (from m/day)
            data_mean = data_mean * 86400

            # Convert the data_mean short to mm/day (from m/day)
            data_mean_short = data_mean_short * 86400

            # Convert the ci_lower to mm/day (from m/day)
            ci_lower = ci_lower * 86400

            # Convert the ci_upper shrt to mm/day (from m/day)
            ci_lower_short = ci_lower_short * 86400

            # Convert the ci_upper to mm/day (from m/day)
            ci_upper = ci_upper * 86400

            # Convert the ci_upper short to mm/day (from m/day)
            ci_upper_short = ci_upper_short * 86400

            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short
            ts_dict["fcst_ts_max_short"] = ci_upper_short
        else:
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the short members to the ts_dict
            ts_dict["fcst_ts_members_short"] = data_short

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the data_mean short to the ts_dict
            ts_dict["fcst_ts_mean_short"] = data_mean_short

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

            # Append the ci_lower short and ci_upper short to the ts_dict
            ts_dict["fcst_ts_min_short"] = ci_lower_short
            ts_dict["fcst_ts_max_short"] = ci_upper_short

        # Calculate the correlation between the obs_ts and fcst_ts_mean
        corr, p = pearsonr(data_mean, obs_ts)

        # Calculate the correlation between the obs_ts and fcst_ts_mean short
        corr_short, p_short = pearsonr(data_mean_short, obs_ts_short)

        # Append the corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

        # Append the corr short and p short to the ts_dict
        ts_dict["corr_short"] = corr_short
        ts_dict["p_short"] = p_short

        # Calculate the standard deviation of the forecast time series
        sig_f = np.std(data_mean)
        sig_o = np.std(obs_ts)

        # Calculate th stdev of the members
        sig_f_short = np.std(data_mean_short)
        sig_o_short = np.std(obs_ts_short)

        # Calculate th stdev of the members
        sig_f_members = np.std(data)
        sig_f_members_short = np.std(data_short)

        # Calculate the rpc
        rpc = corr * (sig_f / sig_f_members)

        # Calculate the rpc for the short period
        rpc_short = corr_short * (sig_f_short / sig_f_members_short)

        # Calculate the rps
        rps = rpc * (sig_o / sig_f_members)

        # Calculate the rps for the short period
        rps_short = rpc_short * (sig_o_short / sig_f_members_short)

        # Append the rpc and rps to the ts_dict
        ts_dict["rpc"] = np.abs(rpc)
        ts_dict["rps"] = np.abs(rps)

        # Append the rpc short and rps short to the ts_dict
        ts_dict["rpc_short"] = np.abs(rpc_short)
        ts_dict["rps_short"] = np.abs(rps_short)

        # Append the nens to the ts_dict
        ts_dict["nens"] = np.shape(data)[0]

    else:
        raise ValueError("Data dimensions are not as expected for alt_lag: ", alt_lag)

    # Return the ts_dict
    return ts_dict


# Define a function which loads the processed ts data into a dataframe
def df_from_ts_dict(
    ts_dict: dict,
    season: str,
    forecast_range: str,
    start_year: str,
    end_year: str,
    lag: int,
    gridbox_name: str,
    variable: str,
    alt_lag: str = None,
    region: str = "global",
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/nao_stats_df",
):
    """
    Load the processed time series data into a dataframe.

    Inputs:
    -------

    ts_dict: dict
        Dictionary containing the time series data to plot.
        e.g. ts_dict = {"obs_ts": obs_ts,
                        "fcst_ts_members": fcst_ts_members,
                        "fcst_ts_min": fcst_ts_min,
                        "fcst_ts_max": fcst_ts_max,
                        "fcst_ts_mean": fcst_ts_mean,
                        "init_years": init_years,
                        "valid_years": valid_years,
                        "nens": nens,
                        "corr": corr,
                        "p": p,
                        "rpc": rpc,
                        "rps": rps,
                        "season": season,
                        "forecast_range": forecast_range,
                        "start_year": start_year,
                        "end_year": end_year,
                        "lag": lag,
                        "variable": variable,
                        "gridbox": gridbox,
                        "gridbox_name": gridbox_name,
                        "alt_lag": alt_lag,

    season: str
        Season for the time series.
        e.g. "DJFM"

    forecast_range: str
        Forecast range for the time series.
        e.g. "2-9"

    start_year: str
        Start year for the time series.
        e.g. "1993"

    end_year: str
        End year for the time series.

    lag: int
        Lag for the time series.
        e.g. 0

    gridbox_name: str
        Name of the gridbox for the time series.
        e.g. "gridbox_1"

    variable: str
        Variable for the time series.
        e.g. "pr"

    alt_lag: str
        Alternative lag for the time series.
        e.g. None

    region: str
        Region for the time series.
        e.g. "global"

    output_dir: str
        Directory to save the dataframe to.
        e.g. "/gws/nopw/j04/canari/users/benhutch/nao_stats_df"

    Outputs:
    --------

    df: pd.DataFrame
        Dataframe containing the time series data.
    """

    # if the outpur_dir does not exist
    if not os.path.exists(output_dir):
        # Make the output_dir
        os.makedirs(output_dir)

    # Set up the dataframe
    df = pd.DataFrame(
        {
            "init_years": ts_dict["init_years"],
            "valid_years": ts_dict["valid_years"],
            "obs_ts": ts_dict["obs_ts"],
            "fcst_ts_mean": ts_dict["fcst_ts_mean"],
            "fcst_ts_min": ts_dict["fcst_ts_min"],
            "fcst_ts_max": ts_dict["fcst_ts_max"],
        }
    )

    # Set up the filename
    if alt_lag is not None:
        filename = f"{variable}_{region}_{season}_{forecast_range}_{start_year}_{end_year}_{lag}_{gridbox_name}_{alt_lag}.csv"
    else:
        filename = f"{variable}_{region}_{season}_{forecast_range}_{start_year}_{end_year}_{lag}_{gridbox_name}.csv"

    # Set up the output path
    output_path = os.path.join(output_dir, filename)

    # Save the dataframe to the output path
    df.to_csv(output_path, index=False)

    # print that the df has been saved
    print(f"df saved to {output_path}")

    # Return the dataframe
    return df


# TODO: Write a function for plotting the time series
def plot_ts(
    ts_dict: dict,
    figsize_x: int = 10,
    figsize_y: int = 12,
    save_dir: str = "/home/users/benhutch/skill-maps/plots",
    trendline: bool = False,
    constrain_years: list = None,
    short_period: bool = False,
    standardise: bool = False,
    do_detrend: bool = False,
    title: str = None,
    label: str = "b",
    fontsize: int = 10,
    calc_rmse: bool = False,
):
    """
    Plot the time series data.

    Inputs:
    -------

    ts_dict: dict
        Dictionary containing the time series data to plot.
        e.g. ts_dict = {"obs_ts": obs_ts,
                        "fcst_ts_members": fcst_ts_members,
                        "fcst_ts_min": fcst_ts_min,
                        "fcst_ts_max": fcst_ts_max,
                        "fcst_ts_mean": fcst_ts_mean,
                        "init_years": init_years,
                        "valid_years": valid_years,
                        "nens": nens,
                        "corr": corr,
                        "p": p,
                        "rpc": rpc,
                        "rps": rps,
                        "season": season,
                        "forecast_range": forecast_range,
                        "start_year": start_year,
                        "end_year": end_year,
                        "lag": lag,
                        "variable": variable,
                        "gridbox": gridbox,
                        "gridbox_name": gridbox_name,
                        "alt_lag": alt_lag,

    figsize_x: int
        Width of the figure in inches.
        e.g. default is 10

    figsize_y: int
        Height of the figure in inches.
        e.g. default is 12

    save_dir: str
        Directory to save the figure to.
        e.g. default is "/gws/nopw/j04/canari/users/benhutch/plots"

    trendline: bool
        Whether to include a trendline on the plot.
        e.g. default is False

    constrain_years: list
        List of years to constrain the plot to.
        e.g. default is None

    short_period: bool
        Whether to plot the short period time series.
        e.g. default is False

    standardise: bool
        Whether to standardise the time series before plotting.
        e.g. default is False

    do_detrend: bool
        Whether to detrend the time series before plotting.
        e.g. default is False

    title: str
        Title for the plot.

    label: str
        Label for the plot.

    fontsize: int
        Fontsize for the plot.

    calc_rmse: bool
        Whether to calculate the RMSE between the forecast and obs time series for use as the uncertainty measure.

    Outputs:
    --------

    None
    """



    # if do_detrend is True
    if do_detrend:
        # Detrend the forecast time series
        ts_dict["fcst_ts_mean"] = signal.detrend(ts_dict["fcst_ts_mean"])

        # # Detrend the forecast time series min
        # ts_dict["fcst_ts_min"] = signal.detrend(ts_dict["fcst_ts_min"])

        # # Detrend the forecast time series max
        # ts_dict["fcst_ts_max"] = signal.detrend(ts_dict["fcst_ts_max"])

        # Detrend the obs time series
        ts_dict["obs_ts"] = signal.detrend(ts_dict["obs_ts"])

    if constrain_years is not None:
        # Print the init_years
        print(f"ts_dict['init_years'] = {ts_dict['init_years']}")

        # Assert that years within constrain_years are in init_years
        assert all(
            year in ts_dict["init_years"] for year in constrain_years
        ), "Years within constrain_years are not in init_years!"

        # Find the indices of constrain_years in init_years
        idxs = [
            np.where(np.array(ts_dict["init_years"]) == year)[0][0]
            for year in constrain_years
        ]

        # Constrain the init_years to constrain_years
        ts_dict["init_years"] = [ts_dict["init_years"][idx] for idx in idxs]

        # Constrain the fcst_ts_mean to constrain_years
        ts_dict["fcst_ts_mean"] = [ts_dict["fcst_ts_mean"][idx] for idx in idxs]

        # Constrain the fcst_ts_members to constrain_years
        ts_dict["fcst_ts_members"] = [ts_dict["fcst_ts_members"][idx] for idx in idxs]

        # Constrain the obs_ts to constrain_years
        ts_dict["obs_ts"] = [ts_dict["obs_ts"][idx] for idx in idxs]

        # Constrain the fcst_ts_min to constrain_years
        ts_dict["fcst_ts_min"] = [ts_dict["fcst_ts_min"][idx] for idx in idxs]

        # Constrain the fcst_ts_max to constrain_years
        ts_dict["fcst_ts_max"] = [ts_dict["fcst_ts_max"][idx] for idx in idxs]

        # Recalculate the correlation between the two time series
        corr, p = pearsonr(ts_dict["fcst_ts_mean"], ts_dict["obs_ts"])

        # Append the new corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

        # Recalculate the rpc
        sig_f = np.std(ts_dict["fcst_ts_mean"])
        sig_o = np.std(ts_dict["obs_ts"])
        sig_f_members = np.std(ts_dict["fcst_ts_members"])

        # Calculate the rpc
        rpc = corr * (sig_f / sig_f_members)

        # Calculate the rps
        rps = rpc * (sig_o / sig_f_members)

        # Append the new rpc and rps to the ts_dict
        ts_dict["rpc"] = np.abs(rpc)
        ts_dict["rps"] = np.abs(rps)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Set up the init_years
    if not short_period:
        init_years = ts_dict["valid_years"]

        # Set up the fcst ts mean
        fcst_ts_mean = ts_dict["fcst_ts_mean"]

        # Set up the obs_ts
        obs_ts = ts_dict["obs_ts"]

        # Set up the min interval
        fcst_ts_min = ts_dict["fcst_ts_min"]

        # Set up the max interval
        fcst_ts_max = ts_dict["fcst_ts_max"]
    else:
        init_years = ts_dict["init_years_short"]

        # Set up the fcst ts mean
        fcst_ts_mean = ts_dict["fcst_ts_mean_short"]

        # Set up the obs_ts
        obs_ts = ts_dict["obs_ts_short"]

        # Set up the min interval
        fcst_ts_min = ts_dict["fcst_ts_min_short"]

        # Set up the max interval
        fcst_ts_max = ts_dict["fcst_ts_max_short"]

    # Standardise the time series
    if standardise:
        # Standardise the forecast time series
        fcst_ts_mean = (fcst_ts_mean - np.mean(fcst_ts_mean)) / np.std(fcst_ts_mean)

        # Standardise the min interval
        fcst_ts_min = (fcst_ts_min - np.mean(fcst_ts_min)) / np.std(fcst_ts_min)

        # Standardise the max interval
        fcst_ts_max = (fcst_ts_max - np.mean(fcst_ts_max)) / np.std(fcst_ts_max)

        # Standardise the observations
        obs_ts = (obs_ts - np.mean(obs_ts)) / np.std(obs_ts)

    if calc_rmse:
        rmse = np.sqrt(np.mean((fcst_ts_mean - obs_ts) ** 2))

        # set up the fcst_ts_min and fcst_ts_max
        fcst_ts_min = fcst_ts_mean - rmse
        fcst_ts_max = fcst_ts_mean + rmse

    if do_detrend is False:
        # Fill between the min and max
        ax.fill_between(
            init_years,
            fcst_ts_min,
            fcst_ts_max,
            color="red",
            alpha=0.3,
        )

    # Plot the observations
    ax.plot(init_years, obs_ts, color="black", label="ERA5")

    # Plot the ensemble mean
    ax.plot(init_years, fcst_ts_mean, color="red", label="dcppA")


    # Set up a horizontal line at 0
    ax.axhline(0, color="black", linestyle="--")

    # Include a trendline
    if trendline and not short_period:
        # Calculate the trendline for the forecast timeseries
        z_fcst = np.polyfit(ts_dict["init_years"], ts_dict["fcst_ts_mean"], 1)
        p_fcst = np.poly1d(z_fcst)
        ax.plot(ts_dict["init_years"], p_fcst(ts_dict["init_years"]), "r--", alpha=0.5)

        # Calculate the trendline for the obs_ts
        z_obs = np.polyfit(ts_dict["init_years"], ts_dict["obs_ts"], 1)
        p_obs = np.poly1d(z_obs)
        ax.plot(ts_dict["init_years"], p_obs(ts_dict["init_years"]), "k--", alpha=0.5)

        # Include the slopes of the trendlines in a textbox
        ax.text(
            0.05,
            0.05,
            f"Model trendline slope = {z_fcst[0]:.2f}\nObs trendline slope = {z_obs[0]:.2f}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8,
        )
    elif trendline and short_period:
        AssertionError("Trendline not implemented for short_period = True")

    # # Include a legend
    # ax.legend(loc="lower right")

    # Set the experiment name
    if ts_dict["alt_lag"]:
        exp_name = "lagged"
    else:
        exp_name = "raw"

    if not short_period:
        # Set the first year
        first_year = ts_dict["init_years"][0]
        last_year = ts_dict["init_years"][-1]
    else:
        # Set the first year
        first_year = ts_dict["init_years_short"][0]
        last_year = ts_dict["init_years_short"][-1] - 10 + 1

    # Include a box in the top left
    # with the location name
    # ax.text(
    #     0.05,
    #     0.95,
    #     f"{ts_dict['gridbox_name']}",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     bbox=dict(facecolor="white", alpha=0.5),
    #     fontsize=8,
    # )

    # if do_detrend is True
    if do_detrend:
        # Calculate the correlation skill
        corr, p = pearsonr(fcst_ts_mean, obs_ts)

        # append the corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

    # Quantify the pvals by bootstrap resampling
    # of time and members
    fcst_members = ts_dict["fcst_ts_members"]

    # print the shape of fcst_members (20, 51)
    print(f"fcst_members.shape = {fcst_members.shape}")

    # Set up the n_times
    n_times = fcst_members.shape[1]

    # set up the nens
    n_ens = fcst_members.shape[0]

    # assert that the n_times is as expected
    assert n_times == len(obs_ts), "n_times not as expected!"

    # Set up the nboot and block length - hardcoded
    nboot = 1000
    block_length = 5

    # Set up the arr for tthe corr
    corr_arr = np.empty([nboot])

    # set up the number of blocks to be used
    n_blocks = int(n_times / block_length)

    # if the nblocks * block_length is less than n_times
    # add one to the number of blocks
    if n_blocks * block_length < n_times:
        n_blocks = n_blocks + 1

    # set up the indexes
    # for the time - time needs to be the same for all forecasts and obs
    index_time = range(n_times - block_length + 1)

    # set up the index for the ensemble
    index_ens = range(n_ens)

    # Print that we are bootstrapping for significance
    print("Bootstrapping for significance")

    # loop over the bootstraps
    for iboot in tqdm(np.arange(nboot)):
        if iboot == 0:
            index_time_this = range(0, n_times, block_length)
            index_ens_this = index_ens
        else:
            index_time_this = np.array(
                    [random.choice(index_time) for i in range(n_blocks)]
                )
            
            index_ens_this = np.array([random.choice(index_ens) for _ in index_ens])
        
        obs_boot = np.zeros([n_times])
        fcst1_boot = np.zeros([n_ens, n_times])

        # Set the itime to 0
        itime = 0

        # loop over the time indexes
        for i_this in index_time_this:
            # Individual block index
            index_block = np.arange(i_this, i_this + block_length)

            # If the block index is greater than the number of times, then reduce the block index
            index_block[(index_block > n_times - 1)] = (
                index_block[(index_block > n_times - 1)] - n_times
            )

            # Select a subset of indices for the block
            index_block = index_block[: min(block_length, n_times - itime)]

            # loop over the block indices
            for iblock in index_block:
                # Assign the values to the arrays
                obs_boot[itime] = obs_ts[iblock]
                fcst1_boot[:, itime] = fcst_members[index_ens_this, iblock]

                # Increment itime
                itime = itime + 1

        # assert that there are non nans in either of the arrays
        assert not np.isnan(obs_boot).any(), "values in nao_boot are nan."
        assert not np.isnan(fcst1_boot).any(), "values in corr_var_ts_boot are nan."

        # Calculate the correlation
        corr_arr[iboot] = pearsonr(obs_boot, fcst1_boot.mean(axis=0))[0]

    print("Bootstrapping complete")
    print("Setting ts_dict['corr'] to the first value of corr_arr")
    # Set up the corr
    ts_dict["corr"] = corr_arr[0]

    # Count values
    count_values = np.sum(
        i < 0.0 for i in corr_arr
    )

    print("Setting ts_dict['p'] to the count_values / nboot")
    # Calculate the p value
    ts_dict["p"] = count_values / nboot

    if title is not None:
        # Set the title
        ax.set_title(title, fontweight="bold", fontsize=16)
    else:
        # Set the title
        if ts_dict["alt_lag"] and not short_period:
            ax.set_title(
                f"ACC = {ts_dict['corr']:.2f} "
                f"(p = {ts_dict['p']:.2f}), "
                f"RPC = {ts_dict['rpc']:.2f}, "
                f"N = {ts_dict['nens']}, "
                f"{exp_name} "
                f"({ts_dict['lag']}), "
                f"{ts_dict['season']}, "
                f"{ts_dict['forecast_range']}, "
                f"{first_year}-{last_year}"
            )
        elif ts_dict["alt_lag"] and short_period:
            ax.set_title(
                f"ACC = {ts_dict['corr_short']:.2f} "
                f"(p = {ts_dict['p_short']:.2f}), "
                f"RPC = {ts_dict['rpc_short']:.2f}, "
                f"N = {ts_dict['nens']}, "
                f"{exp_name} "
                f"({ts_dict['lag']}), "
                f"{ts_dict['season']}, "
                f"{ts_dict['forecast_range']}, "
                f"{first_year}-{last_year}"
            )
        elif not ts_dict["alt_lag"] and short_period:
            ax.set_title(
                f"ACC = {ts_dict['corr_short']:.2f} "
                f"(p = {ts_dict['p_short']:.2f}), "
                f"RPC = {ts_dict['rpc_short']:.2f}, "
                f"N = {ts_dict['nens']}, "
                f"{exp_name}, "
                f"{ts_dict['season']}, "
                f"{ts_dict['forecast_range']}, "
                f"{first_year}-{last_year}"
            )
        else:
            ax.set_title(
                f"ACC = {ts_dict['corr']:.2f} "
                f"(p = {ts_dict['p']:.2f}), "
                f"RPC = {ts_dict['rpc']:.2f}, "
                f"N = {ts_dict['nens']}, "
                f"{exp_name}, "
                f"{ts_dict['season']}, "
                f"{ts_dict['forecast_range']}, "
                f"{first_year}-{last_year}"
            )

    # include the correlation, p-value, rpc and N
    # in the top left hand corner
    ax.text(
    0.05,
    0.95,
    (
        f"ACC = {ts_dict['corr']:.2f} "
        f"(P = {ts_dict['p']:.2f}), "
        f"RPC = {ts_dict['rpc']:.2f}, "
        f"N = {ts_dict['nens']}"
    ),
    transform=ax.transAxes,
    fontsize=fontsize,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(facecolor="white", alpha=0.5),
    )

    if label is not None:
        ax.text(
            0.95,
            0.05,
            label,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=fontsize,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Set up the x label
    ax.set_xlabel("Centre of 8-year window", fontsize=fontsize)

    if ts_dict["variable"] == "tas":
        # Set the y-axis label
        ax.set_ylabel("Temperature anomaly (K)", fontsize=fontsize)
    elif ts_dict["variable"] == "psl":
        # Set the y-axis label
        ax.set_ylabel("Pressure anomaly (hPa)", fontsize=fontsize)
    elif ts_dict["variable"] == "rsds":
        # Set the y-axis label
        ax.set_ylabel("Radiation anomaly (W/m^2)", fontsize=fontsize)
    elif ts_dict["variable"] == "sfcWind":
        # Set the y-axis label
        ax.set_ylabel("Wind anomaly (m/s)", fontsize=fontsize)
    elif ts_dict["variable"] == "pr":
        # Set the y-axis label
        ax.set_ylabel("Monthly precip. anomaly (mm/day)", fontsize=fontsize)
    else:
        raise ValueError("Variable not recognised!")

    # if standardise
    if standardise:
        # Set the y-axis label
        # add standardised to the label
        ax.set_ylabel(f"Standardised {ax.get_ylabel()}", fontsize=fontsize)

    # set up the size of the ticks
    ax.tick_params(axis="both", pad=10, labelsize=fontsize)

    # Set up the current time
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set up the figure name
    fig_name = f"{ts_dict['variable']}_{ts_dict['gridbox_name']}_{ts_dict['season']}_{ts_dict['forecast_range']}_{current_time}.pdf"

    # Set up the figure path
    fig_path = os.path.join(save_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")

    # Show the figure
    plt.show()

    return None
