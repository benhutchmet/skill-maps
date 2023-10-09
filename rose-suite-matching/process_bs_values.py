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
        
    Returns:
        list: A list of the historical models for the given variable.
    """

    


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

    # Set up the observations path for the matching variable
    obs_path_name = obs_path(variable)

    # Get the models for the matching variable
    match_var_models = nms_fnc.match_variable_models(match_var)

    # Process the observed data
    obs = fnc.process_observations(match_var, region, region_grid, forecast_range, season, obs_path, obs_var_name)

    # if the variable is 'rsds'
    # divide the obs data by 86400 to convert from J/m2 to W/m2
    if match_var in ['rsds', 'ssrd']:
        print("converting obs to W/m2")
        obs /= 86400


    # Set up the model season
    if season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    else:
        model_season = season

    # if the method is raw
    if method == "raw":
        # Load and process the model data
        model_datasets = fnc.load_data(base_dir, match_var_models, match_var, region, forecast_range, model_season)
        # Process the model data
        model_data, _ = fnc.process_data(model_datasets, match_var)

        # Make sure that the models have the same time period
        model_data = fnc.constrain_years(model_data, match_var_models)

        # Remove years containing NaNs from the observations and model data
        obs, model_data, _ = fnc.remove_years_with_nans_nao(obs, model_data, match_var_models,
                                                            NAO_matched=False)

        # Call the bootstrapping function
        bs_pfield = fnc.calculate_spatial_correlations_bootstrap(obs, model_data, match_var_models, match_var, n_bootstraps=no_bootstraps,
                                                                experiment=None, lag=None, matched_var_ensemble_members=None,
                                                                ensemble_mean=None, measure=measure)
    elif method == 'lagged':
        # Load and process the model data
        model_datasets = fnc.load_data(base_dir, match_var_models, match_var, region, forecast_range, model_season)
        # Process the model data
        model_data, _ = fnc.process_data(model_datasets, match_var)

        # Make sure that the models have the same time period
        model_data = fnc.constrain_years(model_data, match_var_models)

        # Remove years containing NaNs from the observations and model data
        obs, model_data, _ = fnc.remove_years_with_nans_nao(obs, model_data, match_var_models,
                                                            NAO_matched=False)

        # Call the bootstrapping function
        bs_pfield = fnc.calculate_spatial_correlations_bootstrap(obs, model_data, match_var_models, match_var, n_bootstraps=no_bootstraps,  
                                                                experiment=None, lag=lag, matched_var_ensemble_members=None,
                                                                ensemble_mean=None, measure=measure)
    elif method == "nao_matched":
        # process the psl observations for the nao index
            obs_psl_anomaly = fnc.read_obs('psl', region, forecast_range, season, 
                                            dic.obs, start_year, end_year)

            # Load and process the model data for the NAO index
            model_datasets_psl = fnc.load_data(dic.base_dir, dic.psl_models, 'psl', region, forecast_range, 
                                        model_season)
            # Process the model data
            model_data_psl, _ = fnc.process_data(model_datasets_psl, 'psl')

            # Make sure that the models have the same time period for psl
            model_data_psl = fnc.constrain_years(model_data_psl, dic.psl_models)

            # Remove years containing NaNs from the observations and model data
            # and align the time periods
            obs_psl_anomaly, model_data_psl, _ = fnc.remove_years_with_nans_nao(obs_psl_anomaly, model_data_psl, 
                                                                            dic.psl_models, NAO_matched=False)

            # Calculate the lagged NAO index
            obs_nao, model_nao = fnc.calculate_nao_index_and_plot(obs_psl_anomaly, model_data_psl, dic.psl_models,
                                                                'psl', season, forecast_range, plots_dir)                                                    
            
            # Rescale the NAO index
            rescaled_nao, ensemble_mean_nao, ensemble_members_nao, years = fnc.rescale_nao(obs_nao, model_nao, dic.psl_models,
                                                                                    season, forecast_range, plots_dir, lag=lag)

            # Perform the NAO matching for the target variableOnao
            matched_var_ensemble_mean, matched_var_ensemble_members = fnc.nao_matching_other_var(rescaled_nao, model_nao,
                                                                match_var_models, match_var, match_var, dic.base_dir,
                                                                    match_var_models, obs_path, region, model_season, forecast_range,
                                                                        start_year, end_year, plots_dir, dic.save_dir, lagged_years=years,
                                                                            lagged_nao=True, no_subset_members=no_subset_members)        

            # Remove years containing NaNs from the observations and model data
            obs, matched_var_ensemble_mean, matched_var_ensemble_members = fnc.remove_years_with_nans_nao(obs, matched_var_ensemble_mean,
                                                                                                        match_var_models, NAO_matched=True,
                                                                                                        matched_var_ensemble_members=matched_var_ensemble_members)

            # Call the bootstrapping function
            bs_pfield = fnc.calculate_spatial_correlations_bootstrap(obs, matched_var_ensemble_mean, match_var_models, match_var, n_bootstraps=no_bootstraps,
                                                                    experiment=None, lag=None, matched_var_ensemble_members=matched_var_ensemble_members,
                                                                    ensemble_mean=ensemble_mean_nao, measure=measure)
    else:
        print("Method not recognised. Please try again.")
        sys.exit()

    # Save the bootstrapped values
    save_path = os.path.join(save_dir, match_var, region, season, forecast_range, start_year, end_year, str(lag), str(no_subset_members), method, measure)
    # if the save path does not already exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Saving bootstrapped values to:", save_path)
    # form the filename
    filename = "bootrapped_values" + region + "_" + season + "_" + forecast_range + "_" + start_year + "_" + end_year + "_" + str(lag) + "_" + str(no_subset_members) + "_" + method + "_" + measure + ".npy"

    # Save the bootstrapped values
    np.save(os.path.join(save_path, filename), bs_pfield)
                                                                
if __name__ == "__main__":
    main()
