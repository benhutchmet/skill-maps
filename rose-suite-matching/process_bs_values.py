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
    <lag> <no_subset_members> <method> <measure>

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

# Import the dictionaries
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts

# Import the functions
sys.path.append("/home/users/benhutch/skill-maps/python")
import functions as fnc

# Import the other functions
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
from nao_matching_seasons import match_variable_models, obs_path

# Import the historical functions
sys.path.append("/home/users/benhutch/skill-maps-differences")
import functions_diff as hist_fnc

# Third party imports
import numpy as np


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
    parser.add_argument('match_var', type=str, 
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

# Define the main function
def main():
    """
    Main function which parses the command line arguments and calls the functions to perform the bootstrapping.
    """

    # Set up any hardcoded variables
    base_dir = "/home/users/benhutch/skill-maps-processed-data"

    base_dir_historical = base_dir + "/historical"

    plots_dir = "/home/users/benhutch/skill-maps-processed-data/plots"

    output_dir = "/home/users/benhutch/skill-maps-processed-data/output"

    save_dir = "/gws/nopw/j04/canari/users/benhutch/NAO-matching"

    no_bootstraps = 10 # Test case

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


    # If the region is global, set the region to the global gridspec
    if region == "global":
        region_grid = dicts.gridspec_global
    else:
        raise ValueError("Region not recognised. Please try again.")

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
    obs_path_name = obs_path(variable)

    # Process the observed data
    obs = fnc.process_observations(variable, region, region_grid, 
                                   forecast_range, season, obs_path_name, 
                                   variable, plev=None)

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
    
    # Load and process the model data
    dcpp_data = load_and_process_dcpp_data(base_dir, dcpp_models, variable, 
                                            region, forecast_range, 
                                            model_season)
    
    # Now we process the data to align the time periods and convert to array

                                                                
if __name__ == "__main__":
    main()
