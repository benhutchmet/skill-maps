#!/usr/bin/env python

"""
process_bs_values.py
====================

A script which processes the bootstrapped values for a given variable and season.
Creates and saves a file containing these values.

Usage:
------

    $ python process_bs_values.py <match_var> <region> <season> <forecast_range> <start_year> <end_year> <lag> <no_subset_members> <method>

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
    method: str
        The method to use for the bootstrapping. Must be a method in the input files.

Output:
=======

    A file containing the bootstrapped significance values for the given variable.

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

# Import the dictionaries
sys.path.append('/home/users/benhutch/skill-maps')
import dictionaries as dic

# Import the functions
sys.path.append('/home/users/benhutch/skill-maps/python')
import functions as fnc

# import functions from the other script
sys.path.append('/home/users/benhutch/skill-maps/rose-suite-matching')
import nao_matching_seasons as nms_fnc

# Import the bootstrapping functions
sys.path.append('/home/users/benhutch/skill-maps-differences')
import functions as fnc_bs


# Define the main function
def main():
    """
    Main function which parses the command line arguments and calls the functions to perform the bootstrapping.
    """

    # Set up any hardcoded variables
    base_dir = dic.base_dir
    plots_dir = dic.plots_dir
    save_dir = dic.save_dir
    region_grid = dic.gridspec_global
    obs_path = dic.obs

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
    parser.add_argument('method', type=str, help='The method to use for the bootstrapping.')
    args = parser.parse_args()

    # Extract the command line arguments
    match_var = args.match_var
    region = args.region
    season = args.season
    forecast_range = args.forecast_range
    start_year = args.start_year
    end_year = args.end_year
    lag = args.lag
    no_subset_members = args.no_subset_members
    method = args.method

    # If season conttains a number, convert it to the string
    if season in ["1", "2", "3", "4"]:
        season = dic.season_map[season]
        print("NAO matching for variable:", match_var, "region:", region, "season:", season, "forecast range:", forecast_range,)

    # Set up the models
    match_var_models = nms_fnc.match_variable_models(match_var)

    # Set up the observations path for the matching variable
    obs_path_match_var = nms_fnc.obs_path(match_var)

    # get the obs var name from the dictionary
    obs_var_name = dic.var_name_map[match_var]

    # Get the models for the matching variable
    match_var_models = dic.match_var_models[match_var]

    # Process the observed data
    obs = fnc.process_observations(match_var, region, region_grid, forecast_range, season, obs_path, obs_var_name)

    # Set up the model season
    if season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    else:
        model_season = season

    # Load and process the model data
    model_datasets = fnc.load_data(base_dir, match_var_models, match_var, region, forecast_range, model_season)
    # Process the model data
    model_data, _ = fnc.process_data(model_datasets, match_var)
    
