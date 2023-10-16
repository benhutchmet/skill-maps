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
    <lag> <no_subset_members> <method> <nboot>

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
    fcst1, fcst2, obs, common_years = align_and_convert_to_array(hist_data, 
                                                                 dcpp_data, 
                                                                 hist_models, 
                                                                 dcpp_models,
                                                                 obs)

    # If the method is 'raw', process the forecast stats
    if method == "raw":                                                             
    
        # Now perform the bootstrapping to create the forecast stats
        forecast_stats = fnc.forecast_stats(obs, fcst1, fcst2, 
                                            no_boot = no_bootstraps)
        
    # Else if the method is 'lagged', lag the data
    # Before processing the forecast stats

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
    
    # Save the arrays
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

    # FIXME: These will now be arrays
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
                                                                
if __name__ == "__main__":
    main()
