#!/usr/bin/env python

"""
nao_matching_seasons.py
=======================

A script which performs the NAO matching for a provided variable and season. Creates
a new netCDF file with the ensemble mean of the NAO matched data for the given variable.

Usage:
------

    $ python nao_matching_seasons.py <match_var> <region> <season> <forecast_range> <start_year> <end_year> <lag> <no_subset_members>

Parameters:
===========

    match_var: str
        The variable to perform the matching for. Must be a variable in the input files.
    region: str
        The region to perform the matching for. Must be a region in the input files.
    season: str
        The season to perform the matching for. Must be a season in the input files.
    forecast_range: str
        The forecast range to perform the matching for. Must be a forecast range in the input files.
    start_year: str
        The start year to perform the matching for. Must be a year in the input files.
    end_year: str
        The end year to perform the matching for. Must be a year in the input files.
    lag: int
        The lag to perform the matching for. Must be a lag in the input files.
    no_subset_members: int
        The number of ensemble members to subset to. Must be a number in the input files.

Output:
=======

    A netCDF file with the ensemble mean of the NAO matched data for the given variable.

"""

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

import matplotlib.cm as mpl_cm
import matplotlib
import cartopy.crs as ccrs
import iris
import iris.coord_categorisation as coord_cat
import iris.plot as iplt
import scipy
import pdb
# import iris.quickplot as qplt

# Import CDO
from cdo import *
# cdo = Cdo()

# Import the dictionaries and functions
sys.path.append('/home/users/benhutch/skill-maps')
import dictionaries as dic

# Import the functions
sys.path.append('/home/users/benhutch/skill-maps/python')
import functions as fnc

# Write a function which sets up the models for the matched variable
def match_variable_models(match_var):
    """
    Matches up the matching variable to its models.
    """
    print("match_var:", match_var)
    # Case statement for the matching variable
    if match_var in ["tas", "t2m"]:
        match_var_models = dic.tas_models
    elif match_var in ["sfcWind", "si10"]:
        match_var_models = dic.sfcWind_models_noMIROC
    elif match_var in ["rsds", "ssrd"]:
        match_var_models = dic.rsds_models_noCMCC
    elif match_var in ["psl", "msl" ]:
        match_var_models = dic.models
    else:
        print("The variable is not supported for NAO matching.")
        sys.exit()

    # Return the matched variable models
    return match_var_models

# write a function which sets up the observations path for the variable
def obs_path(match_var):
    """
    Matches up the matching variable to its observations path.
    """

    # Case statement for the matching variable
    if match_var in ["tas", "t2m", "sfcWind", "si10", "rsds", "ssrd", "psl", "msl"]:
        obs_path = dic.obs
    else:
        print("The variable is not supported for NAO matching.")
        sys.exit()

    # Return the matched variable models
    return obs_path


# Set up the main function
def main():
    """
    Main function which parses the command line arguments and performs the NAO matching.
    """

    test_models = [ "BCC-CSM2-MR", "CMCC-CM2-SR5", "MIROC6" ]

    # Set up the hardcoded variables
    psl_var = "psl"
    psl_models = dic.models
    obs_path_psl = dic.obs
    base_dir = dic.base_dir
    plots_dir = dic.plots_dir
    save_dir = dic.save_dir
    seasons_list_obs = dic.seasons_list_obs
    seasons_list_model = dic.seasons_list_model

    # Extract the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('match_var', type=str, help='The variable to perform the matching for.')
    parser.add_argument('region', type=str, help='The region to perform the matching for.')
    parser.add_argument('season', type=str, help='The season to perform the matching for.')
    parser.add_argument('forecast_range', type=str, help='The forecast range to perform the matching for.')
    parser.add_argument('start_year', type=str, help='The start year to perform the matching for.')
    parser.add_argument('end_year', type=str, help='The end year to perform the matching for.')
    parser.add_argument('lag', type=int, help='The lag to perform the matching for.')
    parser.add_argument('no_subset_members', type=int, help='The number of ensemble members to subset to.')
    args = parser.parse_args()

    # Set up the command line arguments
    match_var = args.match_var
    region = args.region
    season = args.season
    forecast_range = args.forecast_range
    start_year = args.start_year
    end_year = args.end_year
    lag = args.lag
    no_subset_members = args.no_subset_members

    # If season conttains a number, convert it to the string
    if season in ["1", "2", "3", "4"]:
        season = dic.season_map[season]
        print("NAO matching for variable:", match_var, "region:", region, "season:", season, "forecast range:", forecast_range,)

    # Set up the models for the matching variable
    match_var_models = match_variable_models(match_var)

    # # Override this for testing
    # match_var_models = test_models

    # Set up the observations path for the matching variable
    obs_path_match_var = obs_path(match_var)

    # extract the obs var name
    obs_var_name = dic.var_name_map[match_var]

    # Process the psl observations for the nao index
    obs_psl_anomaly = fnc.read_obs(psl_var, region, forecast_range, season,
                                    obs_path_psl, start_year, end_year)
    
    # Set up the model season
    if season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    else:
        model_season = season
    
    # Load and process the model data for the nao index
    model_datasets_psl = fnc.load_data(base_dir, psl_models, psl_var, 
                                        region, forecast_range, model_season)
    # Process the model data
    model_data_psl, _ = fnc.process_data(model_datasets_psl, psl_var)

    # Make sure that the models have the same time period for psl
    model_data_psl = fnc.constrain_years(model_data_psl, psl_models)

    # Remove years containing Nan values from the obs and model data
    # Psl has not been NAO matched in this case
    obs_psl_anomaly, model_data_psl, _ = fnc.remove_years_with_nans_nao(obs_psl_anomaly, model_data_psl,
                                                                        psl_models, NAO_matched=False)
    
    # Calculate the NAO index for the obs and model NAO
    obs_nao, model_nao = fnc.calculate_nao_index_and_plot(obs_psl_anomaly, model_data_psl, psl_models,
                                                            psl_var, season, forecast_range, plots_dir,
                                                            plot_graphics=False, azores_grid = dic.azores_grid_corrected,
                                                            iceland_grid = dic.iceland_grid_corrected)
    
    # Perform the lagging of the ensemble and rescale the NAO index
    rescaled_nao, ensemble_mean_nao, ensemble_members_nao, years = fnc.rescale_nao(obs_nao, model_nao, psl_models,
                                                                                    season, forecast_range, plots_dir, lag=lag)

    # Perform the NAO matching for the target variable
    match_var_ensemble_mean, _  = fnc.nao_matching_other_var(rescaled_nao, model_nao, psl_models, match_var, obs_var_name,
                                                            base_dir, match_var_models, obs_path_match_var, region, model_season, forecast_range,
                                                                start_year, end_year, plots_dir, save_dir, lagged_years = years,
                                                                    lagged_nao=True, no_subset_members=no_subset_members, level=None,
                                                                        ensemble_mean_nao=ensemble_mean_nao)
    

if __name__ == '__main__':
    main()

