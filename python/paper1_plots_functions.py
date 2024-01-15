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

# Create a function to process the raw data for the full forecast period
# TODO: may also need to do this for lagged at some point as well
def forecast_stats_var(variables: list,
                       season: str,
                       forecast_range: str,
                       region: str = "global",
                       start_year: int = 1961,
                       end_year: int = 2023,
                       method: str = "raw",
                       no_bootstraps: int = 1,
                       base_dir: str = "/home/users/benhutch/skill-maps-processed-data"):
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
        obs = fnc.process_observations(variable=variable,
                                        region=region,
                                        region_grid=dicts.gridspec_global,
                                        forecast_range=forecast_range,
                                        season=season,
                                        observations_path=obs_path,
                                        obs_var_name=variable)
        
        # Load and process the dcpp model data
        dcpp_data = load_and_process_dcpp_data(base_dir=base_dir,
                                                        dcpp_models=dcpp_models,
                                                        variable=variable,
                                                        region=region,
                                                        forecast_range=forecast_range,
                                                        season=model_season)
        
        # Make sure that the individual models have the same valid years
        dcpp_data = fnc.constrain_years(model_data=dcpp_data,
                                        models=dcpp_models)
        
        # Align the obs and dcpp data
        obs, dcpp_data, _ = fnc.remove_years_with_nans_nao(observed_data=obs,
                                                           model_data=dcpp_data,
                                                           models=dcpp_models)
        

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
                dcpp_array[member_index-1, :, :, :] = data

        # Assert that obs and dcpp_array have the same shape
        assert obs_array.shape == dcpp_array[0, :, :, :].shape, "obs and dcpp_array have different shapes!"

        # Create an empty dictionary to store the forecast stats
        forecast_stats_var[variable] = {}

        # Calculate the forecast stats for the variable
        forecast_stats_var[variable] = fnc.forecast_stats(obs=obs_array,
                                                          forecast1=dcpp_array,
                                                          forecast2=dcpp_array, # use the same here as a placeholder for historical
                                                          no_boot=no_bootstraps)
        
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
        obs_psl_anom = fnc.read_obs(variable="psl",
                                    region=region,
                                    forecast_range=forecast_range,
                                    season=season,
                                    observations_path=obs_path,
                                    start_year=1960,
                                    end_year=2023)
        
        # Load adn process the dcpp model data
        dcpp_data = load_and_process_dcpp_data(base_dir=base_dir,
                                                        dcpp_models=dcpp_models,
                                                        variable="psl",
                                                        region=region,
                                                        forecast_range=forecast_range,
                                                        season=model_season)

        # Remove the years with NaNs from the obs and dcpp data
        obs_psl_anom, \
        dcpp_data, \
        _ = fnc.remove_years_with_nans_nao(observed_data=obs_psl_anom,
                                            model_data=dcpp_data,
                                            models=dcpp_models,
                                            NAO_matched=False)
        
        # Extract the nao stats
        nao_stats_dict = nao_fnc.nao_stats(obs_psl=obs_psl_anom,
                                           hindcast_psl=dcpp_data,
                                           models_list=dcpp_models,
                                           lag=4,
                                           short_period=(1965,2010),
                                           season=season)

    else:
        print("Not calculating the NAO index...")
        nao_stats_dict = None

    # Return the forecast_stats_var dictionary
    return forecast_stats_var, nao_stats_dict

# Define a plotting function for this data
def plot_forecast_stats_var(forecast_stats_var_dic: dict,
                            nao_stats_dict: dict,
                            psl_models: list,
                            season: str,
                            forecast_range: str,
                            figsize_x: int = 10,
                            figsize_y: int = 12,
                            gridbox_corr: dict = None,
                            gridbox_plot: dict = None,
                            sig_threshold: float = 0.05):
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
    total_nens = 0 ; total_lagged_nens = 0

    # Loop over the models
    for model in psl_models:
        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Add the number of ensemble members to the total
        total_nens += nao_stats['nens']

        # And for the lagged ensemble
        total_lagged_nens += nao_stats['nens_lag']

    # Set up the arrays for the NAO index
    nao_members = np.zeros([total_nens, nyears_long])

    # Set up the lagged arrays for the NAO index
    nao_members_lag = np.zeros([total_lagged_nens, nyears_long_lag])

    # Set up the counter for the current index
    current_index = 0 ; current_index_lag = 0

    # Iterate over the models
    for i, model in enumerate(psl_models):
        print(f"Extracting ensemble members for model {model}...")

        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Loop over the ensemble members
        for j in range(nao_stats['nens']):
            print(f"Extracting member {j} from model {model}...")

            # Extract the nao member
            nao_member = nao_stats['model_nao_ts_members'][j, :]

            # Set up the lnegth of the correct time series
            nyears_bcc = len(nao_stats_dict['BCC-CSM2-MR']['years'])

            # If the model is not BCC-CSM2-MR
            if model != "BCC-CSM2-MR":
                # Extract the len of the time series
                nyears = len(nao_stats['years'][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years'])                

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print("The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                    model))
                
                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears
                
                # Assert that the new len is correct
                assert len(nao_member[1:skip_years]) == nyears_bcc, "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members[current_index, :] = nao_member[1: skip_years]
                else:
                    nao_members[current_index, :] = nao_member[1:]
            else:
                # Append this member to the array
                nao_members[current_index, :] = nao_member

            # Increment the counter
            current_index += 1

        # Loop over the lagged ensemble members
        for j in range(nao_stats['nens_lag']):

            # Extract the NAO index for this member
            nao_member = nao_stats['model_nao_ts_lag_members'][j, :]
            print("NAO index extracted for member {}".format(j))

            # Set up the length of the correct time series
            nyears_bcc = len(nao_stats_dict['BCC-CSM2-MR']['years_lag'])

            if model != "BCC-CSM2-MR":
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years_lag'][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years_lag'])

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print("The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                    model))
                
                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears
                
                # Assert that the new len is correct
                assert len(nao_member[1:skip_years]) == nyears_bcc, "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members_lag[current_index_lag, :] = nao_member[1: skip_years]
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
        nrows = int(no_keys / 2) + 1 # Extra row for the NAO index
    else:
        nrows = int((no_keys + 1) / 2) + 1

    # Set up the figure
    fig, axs = plt.subplots(nrows=3,
                            ncols=2,
                            figsize=(figsize_x, figsize_y))
    
    # Update the params for mathtext default rcParams
    plt.rcParams.update({"mathtext.default": "regular"})

    # Set up the sup title
    sup_title = "Total correlation skill (r) for each variable"

    # Set up the sup title
    fig.suptitle(sup_title, fontsize=6, y=0.93)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5) ; lats = np.arange(-90, 90, 2.5)

    # If the gridbox_corr is not None
    if gridbox_corr is not None:
        # Extract the lats and lons from the gridbox_corr
        lon1_corr, lon2_corr = gridbox_corr['lon1'], gridbox_corr['lon2']
        lat1_corr, lat2_corr = gridbox_corr['lat1'], gridbox_corr['lat2']

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
        lon1_gb, lon2_gb = gridbox_plot['lon1'], gridbox_plot['lon2']
        lat1_gb, lat2_gb = gridbox_plot['lat1'], gridbox_plot['lat2']

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
        print(f"Plotting variable {key}...") ; print(f"Plotting index {i}...")

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
            fcst1_ts = fcst1_ts[:, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]
            obs_ts = obs_ts[:, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]

            # Calculate the mean of both time series
            fcst1_ts_mean = np.mean(fcst1_ts, axis=(1, 2))
            obs_ts_mean = np.mean(obs_ts, axis=(1, 2))

            # Calculate the correlation between the two time series
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Show these values on the plot
            ax.text(0.05, 0.05, f"r = {r:.2f}, p = {p:.2f}", transform=ax.transAxes,
                    va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                    fontsize=8)
            
            # Add the gridbox to the plot
            ax.plot([lon1_corr, lon2_corr, lon2_corr, lon1_corr, lon1_corr],
                    [lat1_corr, lat1_corr, lat2_corr, lat2_corr, lat1_corr],
                    color="green", linewidth=2, transform=proj)
    
        # If any of the corr1 values are NaNs
        # then set the p values to NaNs at the same locations
        corr1_p[np.isnan(corr)] = np.nan

        # If any of the corr1_p values are greater than the sig_threshold
        # then set the corr1 values to NaNs at the same locations
        # WHat if the sig threshold is two sided?
        # We wan't to set the corr1_p values to NaNs
        # Where corr1_p<sig_threshold and corr1_p>1-sig_threshold
        corr1_p[(corr1_p > sig_threshold) & (corr1_p < 1-sig_threshold)] = np.nan

        # plot the p-values
        ax.contourf(lons, lats, corr1_p, hatches=["...."], alpha=0., transform=proj)

        # Add a text box with the axis label
        ax.text(0.95, 0.05, f"{axis_labels[i]}", transform=ax.transAxes,
                va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Add a textboc with the variable name in the top left
        ax.text(0.05, 0.95, f"{key}", transform=ax.transAxes,
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Include the number of ensemble members in the top right of the figure
        ax.text(0.95, 0.95, f"n = {nens1}", transform=ax.transAxes,
                va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Add a text box with the season in the top right
        # ax.text(0.95, 0.95, f"{season}", transform=ax.transAxes,
        #         va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
        #         fontsize=6)

        # Add the contourf object to the list
        cf_list.append(cf)

        # Add the axes to the list
        axes.append(ax)

    # Add a colorbar
    cbar = fig.colorbar(cf_list[0], ax=axes, orientation="horizontal", pad=0.05,
                        shrink=0.8)
    cbar.set_label("correlation coefficient", fontsize=10)

    # Now plot the NAO index
    print("Plotting the raw NAO index...")

    # Extract the total_nesn
    total_nens = nao_members.shape[0]

    # Calculate the mean of the nao_members
    nao_members_mean = np.mean(nao_members, axis=0)

    # Calculate the correlation between the nao_members_mean and the obs_ts
    corr1, p1 = pearsonr(nao_members_mean,
                            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts'])

    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1 = (corr1) / (np.std(nao_members_mean) /np.std(nao_members))

    # Calculate the 5th and 95th percentiles of the nao_members
    nao_members_mean_min = np.percentile(nao_members, 5, axis=0)
    nao_members_mean_max = np.percentile(nao_members, 95, axis=0)

    # Plot the ensemble mean
    ax1 = axs[2, 0]

    # Plot the ensemble mean
    ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
            nao_members_mean / 100, color="red", label="dcppA")

    # Plot the obs
    ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts'] / 100, color="black",
            label="ERA5")
    
    # Plot the 5th and 95th percentiles
    ax1.fill_between(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
                    nao_members_mean_min / 100,
                    nao_members_mean_max / 100,
                    color="red", alpha=0.2)

    # Add a text box with the axis label
    ax1.text(0.95, 0.05, f"{axis_labels[-2]}", transform=ax.transAxes,
            va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)
    
    # Include a textbox containing the total nens in the top right
    ax1.text(0.95, 0.95, f"n = {total_nens}", transform=ax.transAxes,
            va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)

    # Include a title which contains the correlations
    # p values and the rpccorr1_pbbbbeuibweiub
    ax1.set_title(f"ACC = {corr1:.2f} (p = {p1:.2f}), "
                 f"RPC = {rpc1:.2f}, "
                 f"N = {total_nens}", fontsize=10)

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
    corr1_lag, p1_lag = pearsonr(nao_members_mean_lag,
                                    nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])
    
    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1_lag = (corr1_lag) / (np.std(nao_members_mean_lag) /np.std(nao_members_lag))

    # Calculate the rps between the model nao index and the obs nao index
    rps1 = rpc1_lag * (np.std(nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag']) / np.std(nao_members_lag))

    # Var adjust the NAO
    nao_var_adjust = nao_members_mean_lag * rps1

    # Calculate the RMSE between the ensemble mean and observations
    rmse = np.sqrt(np.mean((nao_var_adjust - nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])**2))

    # Calculate the upper and lower bounds
    ci_lower = nao_var_adjust - rmse
    ci_upper = nao_var_adjust + rmse

    # Set up the axes
    ax2 = axs[2, 1]

    # Plot the ensemble mean
    ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
            nao_var_adjust / 100, color="red", label="dcppA")
    
    # Plot the obs
    ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'] / 100, color="black",
            label="ERA5")

    # Plot the 5th and 95th percentiles
    ax2.fill_between(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
                    ci_lower / 100,
                    ci_upper / 100,
                    color="red", alpha=0.2)

    # Add a text box with the axis label
    ax2.text(0.95, 0.05, f"{axis_labels[-1]}", transform=ax.transAxes,
            va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)
    
    # Include a textbox containing the total nens in the top right
    ax2.text(0.95, 0.95, f"n = {total_nens_lag}", transform=ax.transAxes,
            va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)

    # Include a title which contains the correlations
    # p values and rpc values
    ax2.set_title(f"ACC = {corr1_lag:.2f} (p = {p1_lag:.2f}), "
                 f"RPC = {rpc1_lag:.2f}, "
                 f"N = {total_nens_lag}", fontsize=10)
    
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