#!/usr/bin/env python

"""
process_bs_values.py
====================

A script which processes the bootstrapped values for a given variable and season.
Creates and saves a file containing these values.

Usage:
------

    $ python process_bs_values.py <match_var> <obs_var_name>
    <region> <season> <forecast_range> <start_year> <end_year> 
    <lag> <no_subset_members> <method> <nboot> <level>

Parameters:
===========

    variable: str
        The variable to perform the matching for. 
        Must be a variable in the input files.
    obs_var_name: str
        The name of the variable in the observations file. 
        Must be a variable in the input files.
    region: str
        The region to perform the matching for. 
        Must be a region in the input files.
    season: str
        The season to perform the matching for. 
        Must be a season in the input files.
    forecast_range: str
        The forecast range to perform the matching for. 
        Must be a forecast range in the input files.
    start_year: str
        The start year to perform the matching for. 
        Must be a year in the input files.
    end_year: str
        The end year to perform the matching for. 
        Must be a year in the input files.
    lag: int
        The lag to perform the matching for. 
        Must be a lag in the input files.
    no_subset_members: int
        The number of ensemble members to subset to. 
        Must be a number in the input files.
    method: str
        The method to use for the bootstrapping. 
        Must be a method in the input files.
    nboot: int
        The number of bootstraps to perform.
    level: int
        The level to perform the matching for. 
        Must be a level in the input files.

Output:
=======

    A file containing the bootstrapped significance values 
    for the given variable.

"""

# Imports
import argparse
import os
import sys

# Import from other scripts
# -------------------------
from typing import Union, Tuple
import xarray as xr

# Import the dictionaries
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts

# Import the functions
sys.path.append("/home/users/benhutch/skill-maps/python")
import functions as fnc

# Import the other functions
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
from nao_matching_seasons import match_variable_models, find_obs_path

# Import the historical functions
sys.path.append("/home/users/benhutch/skill-maps-differences")
import functions_diff as hist_fnc

# Third party imports
import numpy as np
from typing import Tuple
import xarray as xr

# Define a function to extract the variables from the command line
def extract_variables():
    """
    Extract the variables from the command line.
    
    Returns:
    --------

        Args: dict
            A dictionary containing the variables.
    """

    # Set up the parser for the CLAs
    parser = argparse.ArgumentParser()

    # Add the CLAs
    parser.add_argument('variable', type=str, 
                        help='The variable to perform the matching for.')

    parser.add_argument('obs_var_name', type=str,
                        help='The name of the variable in the \
                        observations file.')

    parser.add_argument('region', type=str,
                        help='The region to perform the matching for.')

    parser.add_argument('season', type=str,
                        help='The season to perform the matching for.')

    parser.add_argument('forecast_range', type=str,
                        help='The forecast range to perform the matching for.')

    parser.add_argument('start_year', type=str,
                        help='The start year to perform the matching for.')

    parser.add_argument('end_year', type=str,
                        help='The end year to perform the matching for.')

    parser.add_argument('lag', type=int,
                        help='The lag to perform the matching for.')

    parser.add_argument('no_subset_members', type=int,
                        help='The number of ensemble members to subset to.')

    parser.add_argument('method', type=str,
                        help='The method to use for the bootstrapping.')

    parser.add_argument('nboot', type=int,
                        help='The number of bootstraps to perform.')
    
    # add optional argument for level, which defaults to None
    parser.add_argument('level', type=str, default=None,
                        help='The level to perform the matching for.')

    # Extract the CLAs
    args = parser.parse_args()

    # Return the CLAs
    return args

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
def load_and_process_hist_data(base_dir, hist_models, variable, region,
                                forecast_range, season):
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

    hist_datasets = hist_fnc.load_processed_historical_data(base_dir, 
                                                            hist_models, 
                                                            variable, 
                                                            region, 
                                                            forecast_range, 
                                                            season)
    
    hist_data, _ = hist_fnc.process_data(hist_datasets, variable)

    return hist_data

# Define a new function to load and process the model data
def load_and_process_dcpp_data(base_dir, dcpp_models, variable, region,
                                forecast_range, season):
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

    dcpp_datasets = fnc.load_data(base_dir, dcpp_models, variable, 
                                    region, forecast_range, season)
    
    dcpp_data, _ = fnc.process_data(dcpp_datasets, variable)

    return dcpp_data

# Define a new function to align the time periods and convert to array
def align_and_convert_to_array(hist_data, dcpp_data, hist_models, dcpp_models,
                               obs):
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
        constrained_dcpp_data, dcpp_models, constrained_hist_data, hist_models,
        obs)
    
    # Return the aligned data
    return fcst1, fcst2, obs, common_years

# Define a new function to load the NAO matched data
def load_nao_matched_data(base_dir: str, variable: str, region: str, season: str, forecast_range: str, start_year: int, end_year: int) -> Tuple[xr.Dataset, xr.Dataset]:
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
def align_nao_matched_members(obs: xr.DataArray, 
                                nao_matched_members: xr.Dataset, 
                                constrained_hist_data: dict,
                                hist_models: list) -> tuple:
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
        obs_year = obs.sel(time=f'{year}')
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
        nao_matched_year = nao_matched_members['__xarray_dataarray_variable__'].sel(time=year)
        if np.isnan(nao_matched_year.values).any():
            print(f"there are NaN values in the NAO matched members for {year}")
            if np.isnan(nao_matched_year.values).all():
                print(f"all values are NaN for {year}")
                # Delete the year from the observations
                nao_matched_members = nao_matched_members.sel(time=nao_matched_members.time.values != year)
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
    constrained_hist_data_years = constrained_hist_data[hist_models[0]][0].time.dt.year.values

    # If the years for the NAO matched members are not the same as the constrained historical data
    if not np.array_equal(nao_matched_members_years, 
                          constrained_hist_data_years):
        print("years for NAO matched members and constrained historical data are not the same")

        # Find the common years
        common_years = np.intersect1d(nao_matched_members_years, constrained_hist_data_years)

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
        fcst2_years = constrained_hist_data_nmatch[hist_models[0]][0].time.dt.year.values

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
        constrained_hist_data_years = constrained_hist_data_nmatch_obs[hist_models[0]][0].time.dt.year.values

        # Assert that the arrays are the same
        assert np.array_equal(obs_years, constrained_hist_data_years), \
                                "the years are not the same"
        
        # Extract the nao matched members years again
        nao_matched_members_years = fcst1_nm.time.values

        # Assert that the arrays are the same
        assert np.array_equal(obs_years, nao_matched_members_years), \
                                "the years are not the same"
        
    # Extract the arrays from the datasets
    fcst1_nm = fcst1_nm['__xarray_dataarray_variable__'].values

    # Extract the obs
    obs = obs.values

    # Extract the no. ensemble members for f2
    n_members_hist = np.sum([len(constrained_hist_data_nmatch[model]) for model
                                in constrained_hist_data_nmatch])
    
    # Set up the fcst2 array
    fcst2 = np.zeros([n_members_hist, len(obs_years), fcst1_nm.shape[2], fcst1_nm.shape[3]])

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
    assert fcst1_nm[0].shape == fcst2[0].shape, "the forecast array shapes are not the same"

    # Assert that the array shapes are the same
    assert fcst1_nm[0].shape == obs.shape, "the forecast and obs array shapes are not the same"

    common_years = obs_years

    # Print the shapes
    print("fcst1_nm shape:", fcst1_nm.shape)
    print("fcst2 shape:", fcst2.shape)
    print("obs shape:", obs.shape)

    return (fcst1_nm, fcst2, obs, common_years)

# Define the main function
def main():
    """
    Main function which parses the command line arguments and 
    calls the functions to perform the bootstrapping.
    """

    # Set up any hardcoded variables
    base_dir = "/home/users/benhutch/skill-maps-processed-data"

    base_dir_historical = base_dir + "/historical"

    plots_dir = "/home/users/benhutch/skill-maps-processed-data/plots"

    output_dir = "/home/users/benhutch/skill-maps-processed-data/output"

    save_dir = "/gws/nopw/j04/canari/users/benhutch/bootstrapping"

    # Extract the command line arguments using the function
    args = extract_variables()

    # Extract the command line arguments
    variable = args.variable

    obs_var_name = args.obs_var_name

    region = args.region

    season = args.season

    forecast_range = args.forecast_range

    start_year = args.start_year

    end_year = args.end_year

    lag = args.lag

    no_subset_members = args.no_subset_members

    method = args.method

    no_bootstraps = args.nboot

    level = args.level

    # If the region is global, set the region to the global gridspec
    if region == "global":
        region_grid = dicts.gridspec_global
    else:
        raise ValueError("Region not recognised. Please try again.")
    
    # Assert that method must be 'raw', 'lagged' or 'nao_matched'
    assert method in ["raw", "lagged", "nao_matched"], (
        "Method not recognised. Please try again." +
        "Must be 'raw', 'lagged' or 'nao_matched'"
    )

    # If season conttains a number, convert it to the string
    season = convert_season(season, dicts)

    # Print the variables
    print("NAO matching for variable:", variable, "region:", region, "season:"
          , season, "forecast range:", forecast_range, "start year:",
          start_year, "end year:", end_year, "lag:", lag, "no subset members:",
          no_subset_members, "method:", method)

    # Set up the dcpp models
    dcpp_models = match_variable_models(variable)

    # Set up the historical models
    hist_models = extract_hist_models(variable, dicts)

    # Set up the observations path for the matching variable
    obs_path_name = find_obs_path(variable)

    # Process the observed data
    obs = fnc.process_observations(variable, region, region_grid, 
                                   forecast_range, season, obs_path_name, 
                                   variable, plev=level)


    # if the variable is 'rsds'
    # divide the obs data by 86400 to convert from J/m2 to W/m2
    if variable in ['rsds', 'ssrd']:
        print("converting obs to W/m2")
        obs /= 86400

    # Set up the model season
    if season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    else:
        model_season = season

    # Load and process the historical data
    hist_data = load_and_process_hist_data(base_dir_historical, hist_models, 
                                            variable, region, forecast_range, 
                                            season)
    
    # Set up the constrained historical data (contain only the common years)
    constrained_hist_data = fnc.constrain_years(hist_data, hist_models)
    
    # Load and process the model data
    dcpp_data = load_and_process_dcpp_data(base_dir, dcpp_models, variable, 
                                            region, forecast_range, 
                                            model_season)
    
    # Now we process the data to align the time periods and convert to array
    fcst1, fcst2, obs_array, common_years = align_and_convert_to_array(hist_data, 
                                                                    dcpp_data, 
                                                                    hist_models, 
                                                                    dcpp_models,
                                                                    obs)

    # Set up the
    # TODO: Set up a run which for the raw data calculates the forecast stats
    # for the longer time series (s1961-2014)
    # Would have to test to see whether this breaks the bootstrapping first though                                                                

    # If the method is 'raw', process the forecast stats
    if method == "raw":                                                             
        print("Processing forecast stats for raw method")

        # Now perform the bootstrapping to create the forecast stats
        forecast_stats = fnc.forecast_stats(obs_array, fcst1, fcst2, 
                                            no_boot = no_bootstraps)
        
    # Else if the method is 'lagged', lag the data
    # Before processing the forecast stats
    elif method == "lagged":
        print("Performing lagging before processing forecast stats")

        # Call the function to perform the lagging
        lag_fcst1, lag_obs, lag_fcst2 = fnc.lag_ensemble_array(fcst1, fcst2,
                                                                obs_array, lag=lag)
        
        # Now process the forecast stats for the lagged data
        forecast_stats = fnc.forecast_stats(lag_obs, lag_fcst1, lag_fcst2,
                                            no_boot = no_bootstraps)
    
    # Else if the method is nao_matched
    elif method == "nao_matched":
        print("Performing NAO matching before processing forecast stats")

        # Set up the NAO matching base directory
        nao_match_base_dir = "/gws/nopw/j04/canari/users/benhutch/NAO-matching"

        # Load the nao_matched data
        nao_matched_data = load_nao_matched_data(nao_match_base_dir, variable,
                                                region, season, forecast_range,
                                                start_year, end_year)
        
        # Extract the nao_matched members and mean
        nao_matched_members = nao_matched_data[0]
        # nao_matched_mean = nao_matched_data[1]

        # Use the function to constrain the NAO matched members
        aligned_data = align_nao_matched_members(obs, nao_matched_members,
                                                    constrained_hist_data,
                                                    hist_models)
        
        # Extract the aligned NAO matched members, forecast2, obs, and common years
        fcst1_nm = aligned_data[0]
        fcst2 = aligned_data[1]
        obs_array = aligned_data[2]
        common_years = aligned_data[3]

        # Now perform the bootstrapping to create the forecast stats
        forecast_stats = fnc.forecast_stats(obs_array, fcst1_nm, fcst2, 
                                            no_boot = no_bootstraps)
        
    else:
        raise ValueError("Method not recognised. Please try again.")

    # Check that forecast_stats exists and is a dictionary
    assert isinstance(forecast_stats, dict), "forecast_stats is not a dictionary"

    # Set up the save path
    save_path = save_dir + "/" + variable + "/" + region + "/" + season + "/" \
                + forecast_range + "/" + method + "/" + "no_bootstraps_" + \
                str(no_bootstraps) + "/"
    
    # If the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up the file names for the arrays
    corr1_name = f"corr1_{variable}_{region}_{season}_{forecast_range}.npy"

    corr1_p_name = f"corr1_p_{variable}_{region}_{season}_{forecast_range}.npy"

    corr2_name = f"corr2_{variable}_{region}_{season}_{forecast_range}.npy"

    corr2_p_name = f"corr2_p_{variable}_{region}_{season}_{forecast_range}.npy"

    corr10_name = f"corr10_{variable}_{region}_{season}_{forecast_range}.npy"

    corr10_p_name = (f"corr10_p_{variable}_{region}_{season}_{forecast_range}"+
                    ".npy")

    partial_r_name = (
        f"partial_r_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Partial r min and max values
    partial_r_min_name = (
        f"partial_r_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    partial_r_max_name = (
        f"partial_r_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also the partial r bias
    partial_r_bias_name = (
        f"partial_r_bias_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    obs_resid_name = (
        f"obs_resid_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Save the forecaast 1 residual array
    fcst1_em_resid_name = (
        f"fcst1_em_resid_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    partial_r_p_name = (
        f"partial_r_p_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    sigo = f"sigo_{variable}_{region}_{season}_{forecast_range}.npy"

    sigo_resid = (
        f"sigo_resid_{variable}_{region}_{season}_" +
        f"{forecast_range}.npy"
    )

    # Also save arrays for the correlation differences
    corr_diff_name = (
        f"corr_diff_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Min and max values
    corr_diff_min_name = (
        f"corr_diff_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr_diff_max_name = (
        f"corr_diff_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr_diff_p_name = (
        f"corr_diff_p_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save arrays for the RPC and RPC_p
    rpc1_name = (
        f"rpc1_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Min and max arrays
    rpc1_min_name = (
        f"rpc1_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    rpc1_max_name = (
        f"rpc1_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    rpc1_p_name = (
        f"rpc1_p_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    rpc2_name = (
        f"rpc2_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Min and max arrays
    rpc2_min_name = (
        f"rpc2_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    rpc2_max_name = (
        f"rpc2_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    rpc2_p_name = (
        f"rpc2_p_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save the arrays for MSSS1 and MSSS2

    msss1_name = (
        f"msss1_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save the arrays for the min and max values of MSS1 and MSS2
    msss1_min_name = (
        f"msss1_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    msss1_max_name = (
        f"msss1_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save the arrays for the MSSS1 and MSSS2 p values
    msss1_p_name = (
        f"msss1_p_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save the corr1 and corr2 min and max values
    corr1_min_name = (
        f"corr1_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr1_max_name = (
        f"corr1_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr2_min_name = (
        f"corr2_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr2_max_name = (
        f"corr2_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Also save the corr10 min and max values
    corr10_min_name = (
        f"corr10_min_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    corr10_max_name = (
        f"corr10_max_{variable}_{region}_{season}_{forecast_range}" +
        ".npy"
    )

    # Set up the names for the forecast time series
    fcst1_ts_name = f"fcst1_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    fcst2_ts_name = f"fcst2_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    fcst10_ts_name = f"fcst10_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    obs_ts_name = f"obs_ts_{variable}_{region}_{season}_{forecast_range}.npy"
    
    # Save the arrays
    # if the file already exists, don't overwrite it
    np.save(save_path + corr1_name, forecast_stats["corr1"])

    # Save the min and max values
    np.save(save_path + corr1_min_name, forecast_stats["corr1_min"])

    np.save(save_path + corr1_max_name, forecast_stats["corr1_max"])

    np.save(save_path + corr1_p_name, forecast_stats["corr1_p"])

    np.save(save_path + corr2_name, forecast_stats["corr2"])

    # Save the min and max values
    np.save(save_path + corr2_min_name, forecast_stats["corr2_min"])

    np.save(save_path + corr2_max_name, forecast_stats["corr2_max"])

    np.save(save_path + corr2_p_name, forecast_stats["corr2_p"])

    np.save(save_path + corr10_name, forecast_stats["corr10"])

    # Save the min and max values
    np.save(save_path + corr10_min_name, forecast_stats["corr10_min"])

    np.save(save_path + corr10_max_name, forecast_stats["corr10_max"])

    np.save(save_path + corr10_p_name, forecast_stats["corr10_p"])

    # Save the MSSS1 and MSSS2 arrays
    np.save(save_path + msss1_name, forecast_stats["msss1"])


    # Save the min and max values
    np.save(save_path + msss1_min_name, forecast_stats["msss1_min"])

    np.save(save_path + msss1_max_name, forecast_stats["msss1_max"])

    # Save the MSSS1 and MSSS2 p values

    np.save(save_path + msss1_p_name, forecast_stats["msss1_p"])

    # Save the RPC1 and RPC2 arrays
    np.save(save_path + rpc1_name, forecast_stats["rpc1"])

    np.save(save_path + rpc2_name, forecast_stats["rpc2"])

    # Save the min and max values
    np.save(save_path + rpc1_min_name, forecast_stats["rpc1_min"])

    np.save(save_path + rpc1_max_name, forecast_stats["rpc1_max"])

    np.save(save_path + rpc2_min_name, forecast_stats["rpc2_min"])

    np.save(save_path + rpc2_max_name, forecast_stats["rpc2_max"])

    # Save the RPC1 and RPC2 p values
    np.save(save_path + rpc1_p_name, forecast_stats["rpc1_p"])

    np.save(save_path + rpc2_p_name, forecast_stats["rpc2_p"])

    # Save the corr_diff arrays
    np.save(save_path + corr_diff_name, forecast_stats["corr_diff"])

    # Save the min and max values
    np.save(save_path + corr_diff_min_name, forecast_stats["corr_diff_min"])

    np.save(save_path + corr_diff_max_name, forecast_stats["corr_diff_max"])

    np.save(save_path + corr_diff_p_name, forecast_stats["corr_diff_p"])

    # Save the partial r min and max values
    np.save(save_path + partial_r_min_name, forecast_stats["partialr_min"])

    np.save(save_path + partial_r_max_name, forecast_stats["partialr_max"])

    # Save the partial r bias
    np.save(save_path + partial_r_bias_name, forecast_stats["partialr_bias"])

    # Save the fcst1_em_resid array
    np.save(save_path + fcst1_em_resid_name, forecast_stats["fcst1_em_resid"])

    np.save(save_path + partial_r_name, forecast_stats["partialr"])

    np.save(save_path + obs_resid_name, forecast_stats["obs_resid"])

    np.save(save_path + partial_r_p_name, forecast_stats["partialr_p"])

    np.save(save_path + sigo, forecast_stats["sigo"])

    np.save(save_path + sigo_resid, forecast_stats["sigo_resid"])

    # Set up the names for the values of the forecast stats
    nens1_name = f"nens1_{variable}_{region}_{season}_{forecast_range}.txt"

    nens2_name = f"nens2_{variable}_{region}_{season}_{forecast_range}.txt"

    start_end_years = (
        f"start_end_years_{variable}_{region}_{season}_" +
        f"{forecast_range}.txt"
    )
    
    # Save the values of the forecast stats
    np.savetxt(save_path + nens1_name, np.array([forecast_stats["nens1"]]))

    np.savetxt(save_path + nens2_name, np.array([forecast_stats["nens2"]]))

    np.savetxt(save_path + start_end_years, [common_years[0], 
                                             common_years[-1]])   
    # Save the forecast time series
    np.save(save_path + fcst1_ts_name, forecast_stats["f1_ts"])

    np.save(save_path + fcst2_ts_name, forecast_stats["f2_ts"])

    np.save(save_path + fcst10_ts_name, forecast_stats["f10_ts"])

    np.save(save_path + obs_ts_name, forecast_stats["o_ts"])
                                                                
if __name__ == "__main__":
    main()
