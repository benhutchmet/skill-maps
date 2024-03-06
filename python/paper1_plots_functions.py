"""
Functions for use in paper1_plots.ipynb notesbook.
"""

# Local Imports
import os
import sys

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr
from datetime import datetime

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

    # Use constrain_years to make sure that the models have the same time axis
    constrained_hist_data = fnc.constrain_years(hist_data, hist_models)

    constrained_dcpp_data = fnc.constrain_years(dcpp_data, dcpp_models)

    # Align the forecasts and observations
    fcst1, fcst2, obs, common_years = fnc.align_forecast1_forecast2_obs(
        constrained_dcpp_data, dcpp_models, constrained_hist_data, hist_models, obs
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
        ax.contourf(lons, lats, corr1_p, hatches=["...."], alpha=0.0, transform=proj)

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

    # Create a dictionary containing the reauired stars for each variable
    stats_dict = {
        "corr1": [],
        "corr1_p": [],
        "f1_ts": [],
        "o_ts": [],
        "nens1": mdi,
        "start_year": mdi,
        "end_year": mdi,
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
        key = (var, f"nboot_{nboot}")

        # Create an empty dictionary to store the skill maps for this variable
        skill_maps = {}

        # Find the file containing "corr1_{variable}" in the base_path
        corr1_file = [file for file in os.listdir(base_path) if f"corr1_{var}" in file]

        # Find the file containing "corr1_p_{variable}" in the base_path
        corr1_p_file = [
            file for file in os.listdir(base_path) if f"corr1_p_{var}" in file
        ]

        # Find the file containing "fcst1_ts_{variable}" in the base_path
        fcst1_ts_file = [
            file for file in os.listdir(base_path) if f"fcst1_ts_{var}" in file
        ]

        # Find the file containing "obs_ts_{variable}" in the base_path
        obs_ts_file = [
            file for file in os.listdir(base_path) if f"obs_ts_{var}" in file
        ]

        # Find the file containing "nens1_{variable}" in the base_path
        nens1_file = [file for file in os.listdir(base_path) if f"nens1_{var}" in file]

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
                common_years_file,
            ]:
                assert len(file) == 1, f"Length of file list is not equal to 1"
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
                assert len(file) == 1, f"Length of file list is not equal to 1"

        # Load the files
        corr1 = np.load(os.path.join(base_path, corr1_file[0]))

        # Load the files
        corr1_p = np.load(os.path.join(base_path, corr1_p_file[0]))

        # Load the files
        fcst1_ts = np.load(os.path.join(base_path, fcst1_ts_file[0]))

        # Load the files
        obs_ts = np.load(os.path.join(base_path, obs_ts_file[0]))

        # Load the files
        nens1 = np.loadtxt(os.path.join(base_path, nens1_file[0])).astype(int)

        if not start_end_years_file:
            # load the common years file
            common_years = np.load(os.path.join(base_path, common_years_file[0]))

            # Extract the start and end years
            start_year = common_years[0].astype(int)
            end_year = common_years[1].astype(int)
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

    gridbox_corr: dict
        Dictionary containing the gridbox for which to calculate the correlation.
        e.g. default is None

    gridbox_plot: dict
        Dictionary containing the gridbox for which to plot the skill maps.
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

    # Set up the figure
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(figsize_x, figsize_y))

    # Adjust the whitespace
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Extract the first variable name from bs_skill_maps
    # Extract the first element for each of the tuple keys
    variables = list(bs_skill_maps.keys())
    variables = [var[0] for var in variables]

    # print the variables
    print(f"variables = {variables}")

    # If the gridbox_corr is not None
    if gridbox_corr is not None and "psl" in variables:
        # Extract the lats and lons from the gridbox_corr
        lon1_corr, lon2_corr = gridbox_corr["lon1"], gridbox_corr["lon2"]
        lat1_corr, lat2_corr = gridbox_corr["lat1"], gridbox_corr["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
        lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
        lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))

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
        # Extract the lats and lons from the gridbox_corr
        lon1_corr, lon2_corr = gridbox_corr["lon1"], gridbox_corr["lon2"]
        lat1_corr, lat2_corr = gridbox_corr["lat1"], gridbox_corr["lat2"]

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
        lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
        lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))
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

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up a list to store the contourf objects
    cf_list = []

    # Set up the axes
    axes = []

    # Loop over the keys in the bs_skill_maps dictionary
    for i, (key, skill_maps) in enumerate(bs_skill_maps.items()):
        # Logging
        print(f"Plotting variable {key}...")
        print(f"Plotting index {i}...")

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

        # Extract the variable name from the key
        variable = key[0]

        # If gridbox_corr is not None
        if gridbox_corr is not None and variable != "psl":
            # Loggging
            print("Calculating the correlations with a specific gridbox...")
            print("As defined by gridbox_corr = ", gridbox_corr)
            print("Variable is not psl")

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
        elif gridbox_corr is not None and variable == "psl":
            print("Calculating the correlations for NAO gridboxes...")
            print("For variable psl")

            # Print the shape of fcst1_ts
            print(f"fcst1_ts.shape = {fcst1_ts.shape}")

            # Print the shape of obs_ts
            print(f"obs_ts.shape = {obs_ts.shape}")

            # Print the lat1_idx_corr_n
            print(f"lat1_idx_corr_n = {lat1_idx_corr_n}")
            print(f"lat2_idx_corr_n = {lat2_idx_corr_n}")
            print(f"lon1_idx_corr_n = {lon1_idx_corr_n}")
            print(f"lon2_idx_corr_n = {lon2_idx_corr_n}")

            # Print the lat1_idx_corr_s
            print(f"lat1_idx_corr_s = {lat1_idx_corr_s}")
            print(f"lat2_idx_corr_s = {lat2_idx_corr_s}")
            print(f"lon1_idx_corr_s = {lon1_idx_corr_s}")
            print(f"lon2_idx_corr_s = {lon2_idx_corr_s}")

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
                fontsize=8,
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

        # plot the p-values
        ax.contourf(lons, lats, corr1_p, hatches=["...."], alpha=0.0, transform=proj)

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
            fontsize=10,
        )

        if methods is not None:
            # If the list contains alt_lag
            if "alt_lag" in methods:
                # Replace alt_lag with lagged, where alt_lag is found
                methods = [method.replace("alt_lag", "Lagged") for method in methods]
            elif "nao_matched" in methods:
                # Replace nao_matched with NAO-matched, where nao_matched is found
                methods = [
                    method.replace("nao_matched", "NAO-matched") for method in methods
                ]
            elif "raw" in methods:
                # Replace raw with Raw, where raw is found
                methods = [method.replace("raw", "Raw") for method in methods]
            else:
                # AssertionError
                AssertionError("Method not recognised!")

            # Include the method in the top right of the figure
            ax.text(
                0.95,
                0.95,
                f"{methods[i]} ({nens1})",
                transform=ax.transAxes,
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8,
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
                fontsize=8,
            )

        # Add the contourf object to the list
        cf_list.append(cf)

        # Add the axes to the list
        axes.append(ax)

    # # Remove content from the 4th axis
    # axs[1, 1].remove()

    # If nrows is 3 and len(bs_skill_maps.keys()) is 5
    # then remove the 5th axis
    if nrows == 3 and len(bs_skill_maps.keys()) == 5:
        # Remove the 5th axis
        axs[2, 1].remove()

    # Add a colorbar
    cbar = fig.colorbar(
        cf_list[0], ax=axes, orientation="horizontal", pad=0.05, shrink=0.8
    )

    # Set the label for the colorbar
    cbar.set_label("correlation coefficient", fontsize=10)

    # Set up the path for saving the figure
    plots_dir = "/gws/nopw/j04/canari/users/benhutch/plots"

    # Set up the current date
    current_date = datetime.now().strftime("%Y%m%d")

    # Set up the figure name
    fig_name = f"different_variables_corr_{start_year}_{end_year}_{season}_{forecast_range}_{current_date}"

    # Set up the figure path
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

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
        fontsize=8,
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
    alt_lag: bool = False,
    region: str = "global",
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

    alt_lag: bool
        Whether to use the alt_lag method.
        e.g. default is False

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
        "fcst_ts_members": [],
        "fcst_ts_min": [],
        "fcst_ts_max": [],
        "fcst_ts_mean": [],
        "init_years": [],
        "valid_years": [],
        "nens": mdi,
        "corr": mdi,
        "p": mdi,
        "rpc": mdi,
        "rps": mdi,
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
    if alt_lag:
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

    # Set up the lats and lons
    lats = np.arange(-90, 90, 2.5)
    lons = np.arange(-180, 180, 2.5)

    # If the forecast range is a single digit
    if forecast_range.isdigit():
        forecast_range_obs = "1"
    else:
        forecast_range_obs = forecast_range

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
    if data.ndim == 5 and alt_lag is False:
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
        else:
            # Set up the valid years
            valid_years = np.arange(raw_first_year, raw_last_year)

        # Append the valid years to the ts_dict
        ts_dict["valid_years"] = valid_years

        if forecast_range != "2-9":
            # Constrain the obs_anoms to the valid years
            obs_anoms = obs_anoms.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31")
            )
        else:
            # Constrain the obs_anoms to the valid years
            obs_anoms = obs_anoms.sel(
                time=slice(f"{raw_first_year}-01-01", f"{raw_last_year - 1}-12-31")
            )

        # Constrain the obs_anoms to the gridbox
        obs_anoms = obs_anoms.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
            dim=["lat", "lon"]
        )

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

        # Extract the obs_anoms as its values
        obs_ts = obs_anoms.values

        if variable == "psl":
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts / 100
        else:
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

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

        # Calculate the 5% lowest interval
        ci_lower = np.percentile(data, 5, axis=0)

        # Calculate the 95% highest interval
        ci_upper = np.percentile(data, 95, axis=0)

        # Calculate the mean of the data
        data_mean = np.mean(data, axis=0)

        if variable == "psl":
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data / 100

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean / 100

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower / 100
            ts_dict["fcst_ts_max"] = ci_upper / 100
        else:
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

        # Calculate the correlation between the obs_ts and fcst_ts_mean
        corr, p = pearsonr(data_mean, obs_ts)

        # Append the corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

        # Calculate the standard deviation of the forecast time series
        sig_f = np.std(data_mean)
        sig_o = np.std(obs_ts)

        # Calculate th stdev of the members
        sig_f_members = np.std(data)

        # Calculate the rpc
        rpc = corr * (sig_f / sig_f_members)

        # Calculate the rps
        rps = rpc * (sig_o / sig_f_members)

        # Append the rpc and rps to the ts_dict
        ts_dict["rpc"] = np.abs(rpc)
        ts_dict["rps"] = np.abs(rps)

        # Append the nens to the ts_dict
        ts_dict["nens"] = data.shape[0]

    elif data.ndim == 4 and alt_lag is True:
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

        # Constrain the obs_anoms to the valid years
        obs_anoms = obs_anoms.sel(
            time=slice(f"{alt_lag_first_year}-01-01", f"{alt_lag_last_year}-12-31")
        )

        # Constrain the obs_anoms to the gridbox
        obs_anoms = obs_anoms.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
            dim=["lat", "lon"]
        )

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

        # Extract the obs_anoms as its values
        obs_ts = obs_anoms.values

        if variable == "psl":
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts / 100
        else:
            # Append the obs_ts to the ts_dict
            ts_dict["obs_ts"] = obs_ts

        # Swap the axes of the data
        data = np.swapaxes(data, 0, 1)

        # Print the shape of the data
        print(f"data.shape = {data.shape}")
        print(f"obs_anoms.shape = {obs_anoms.shape}")

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

        # Calculate the mean of the data
        data_mean = np.mean(data, axis=0)

        # Calculate the 5% lowest interval
        ci_lower = np.percentile(data, 5, axis=0)

        # Calculate the 95% highest interval
        ci_upper = np.percentile(data, 95, axis=0)

        if variable == "psl":
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data / 100

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean / 100

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower / 100
            ts_dict["fcst_ts_max"] = ci_upper / 100
        else:
            # Append the data to the ts_dict
            ts_dict["fcst_ts_members"] = data

            # Append the data_mean to the ts_dict
            ts_dict["fcst_ts_mean"] = data_mean

            # Append the ci_lower and ci_upper to the ts_dict
            ts_dict["fcst_ts_min"] = ci_lower
            ts_dict["fcst_ts_max"] = ci_upper

        # Calculate the correlation between the obs_ts and fcst_ts_mean
        corr, p = pearsonr(data_mean, obs_ts)

        # Append the corr and p to the ts_dict
        ts_dict["corr"] = corr
        ts_dict["p"] = p

        # Calculate the standard deviation of the forecast time series
        sig_f = np.std(data_mean)
        sig_o = np.std(obs_ts)

        # Calculate th stdev of the members
        sig_f_members = np.std(data)

        # Calculate the rpc
        rpc = corr * (sig_f / sig_f_members)

        # Calculate the rps
        rps = rpc * (sig_o / sig_f_members)

        # Append the rpc and rps to the ts_dict
        ts_dict["rpc"] = np.abs(rpc)
        ts_dict["rps"] = np.abs(rps)

        # Append the nens to the ts_dict
        ts_dict["nens"] = np.shape(data)[0]

    else:
        raise ValueError("Data dimensions are not as expected for alt_lag: ", alt_lag)

    # Return the ts_dict
    return ts_dict


# TODO: Write a function for plotting the time series
def plot_ts(
    ts_dict: dict,
    figsize_x: int = 10,
    figsize_y: int = 12,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
    trendline: bool = False,
    constrain_years: list = None,
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

    Outputs:
    --------

    None
    """

    if constrain_years is not None:
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

    # Plot the ensemble mean
    ax.plot(ts_dict["init_years"], ts_dict["fcst_ts_mean"], color="red", label="dcppA")

    # Plot the observations
    ax.plot(ts_dict["init_years"], ts_dict["obs_ts"], color="black", label="ERA5")

    # Fill between the min and max
    ax.fill_between(
        ts_dict["init_years"],
        ts_dict["fcst_ts_min"],
        ts_dict["fcst_ts_max"],
        color="red",
        alpha=0.3,
    )

    # Set up a horizontal line at 0
    ax.axhline(0, color="black", linestyle="--")

    # Include a trendline
    if trendline:
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

    # Include a legend
    ax.legend(loc="lower right")

    # Set the experiment name
    if ts_dict["alt_lag"]:
        exp_name = "lagged"
    else:
        exp_name = "raw"

    # Set the first year
    first_year = ts_dict["init_years"][0]
    last_year = ts_dict["init_years"][-1]

    # Include a box in the top left
    # with the location name
    ax.text(
        0.05,
        0.95,
        f"{ts_dict['gridbox_name']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.5),
        fontsize=8,
    )

    # Set the title
    if ts_dict["alt_lag"]:
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

    # Set the x-axis label
    ax.set_xlabel("Initialisation year")

    if ts_dict["variable"] == "tas":
        # Set the y-axis label
        ax.set_ylabel("Temperature anomaly (K)")
    elif ts_dict["variable"] == "psl":
        # Set the y-axis label
        ax.set_ylabel("Pressure anomaly (hPa)")
    elif ts_dict["variable"] == "rsds":
        # Set the y-axis label
        ax.set_ylabel("Radiation anomaly (W/m^2)")
    elif ts_dict["variable"] == "sfcWind":
        # Set the y-axis label
        ax.set_ylabel("Wind anomaly (m/s)")
    elif ts_dict["variable"] == "pr":
        # Set the y-axis label
        ax.set_ylabel("Monthly precip (mm)")
    else:
        raise ValueError("Variable not recognised!")

    # Set up the current time
    current_time = datetime.now().strftime("%Y%m%d")

    # Set up the figure name
    fig_name = f"{ts_dict['variable']}_{ts_dict['gridbox_name']}_{ts_dict['season']}_{ts_dict['forecast_range']}_{current_time}"

    # Set up the figure path
    fig_path = os.path.join(save_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()

    return None
