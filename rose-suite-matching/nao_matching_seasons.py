#!/usr/bin/env python

"""
nao_matching_seasons.py
=======================

A script which performs the NAO matching for a provided variable and season. Creates
a new netCDF file with the ensemble mean of the NAO matched data for the given variable.

Usage:
------

    $ python nao_matching_seasons.py <match_var> <region> <season> <forecast_range> <start_year> <end_year> <lag> <no_subset_members> <level> <match_type>

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
    level: int
        The level to perform the matching for. Must be a level in the input files.
    match_type: str
        The type of matching to perform. The two supported types are "nao" and "spna".

Output:
=======

    A netCDF file with the ensemble mean of the NAO matched data for the given variable.

"""

# Local imports
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import glob
import re

sys.path.append('/home/users/benhutch/skill-maps/python')
# Imports
import functions as fnc

# Third party imports

# Import the dictionaries and functions
sys.path.append('/home/users/benhutch/skill-maps')
import dictionaries as dic

# Import the functions
sys.path.append('/home/users/benhutch/skill-maps/python')

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
    elif match_var in ["psl", "msl"]:
        match_var_models = dic.models
    elif match_var in ["ua", "va"]:
        match_var_models = dic.common_models_noIPSL_noCan
    else:
        print("The variable is not supported for NAO matching.")
        sys.exit()

    # Return the matched variable models
    return match_var_models

# write a function which sets up the observations path for the variable


def find_obs_path(match_var):
    """
    Matches up the matching variable to its observations path.
    """

    # Case statement for the matching variable
    if match_var in ["tas", "t2m", "sfcWind", "si10", "rsds", "ssrd", "psl", "msl"]:
        obs_path = dic.obs
    elif match_var in ["ua", "va"]:
        obs_path = dic.obs_ua_va
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

    test_models = ["BCC-CSM2-MR", "CMCC-CM2-SR5", "MIROC6"]

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
    parser.add_argument('match_var', type=str,
                        help='The variable to perform the matching for.')
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
    parser.add_argument(
        'lag', type=int, help='The lag to perform the matching for.')
    parser.add_argument('no_subset_members', type=int,
                        help='The number of ensemble members to subset to.')
    parser.add_argument('level', type=int, nargs='?', default=None,
                        help='The level to perform the matching for.')
    parser.add_argument('match_type', type=str, nargs='?',
                        default="nao", help='The type of matching to perform.')
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
    level = args.level
    match_type = args.match_type

    # If season conttains a number, convert it to the string
    if season in ["1", "2", "3", "4"]:
        season = dic.season_map[season]
        print("NAO matching for variable:", match_var, "region:", region,
              "season:", season, "forecast range:", forecast_range,)

    # Set up the models for the matching variable
    match_var_models = match_variable_models(match_var)

    # # Override this for testing
    # match_var_models = test_models

    # Set up the observations path for the matching variable
    obs_path_match_var = find_obs_path(match_var)

    # extract the obs var name
    obs_var_name = dic.var_name_map[match_var]

    # Set up the model season
    if season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    else:
        model_season = season

    # If the match type is nao, perform the nao matching
    if match_type == "nao":
        print("NAO matching for variable:", match_var, "region:", region,
              "season:", season, "forecast range:", forecast_range, "start year:",
              start_year, "end year:", end_year, "lag:", lag, "no subset members:",
              no_subset_members, "level:", level, "match type:", match_type)

        # Process the psl observations for the nao index
        obs_psl_anomaly = fnc.read_obs(psl_var, region, forecast_range, season,
                                       obs_path_psl, start_year, end_year)

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
        # NOTE: Corrected grids here
        obs_nao, model_nao = fnc.calculate_nao_index_and_plot(obs_psl_anomaly, model_data_psl, psl_models,
                                                              psl_var, season, forecast_range, plots_dir,
                                                              plot_graphics=False, azores_grid=dic.azores_grid_corrected,
                                                              iceland_grid=dic.iceland_grid_corrected)

        # Perform the lagging of the ensemble and rescale the NAO index
        rescaled_nao, ensemble_mean_nao, ensemble_members_nao, years = fnc.rescale_nao(obs_nao, model_nao, psl_models,
                                                                                       season, forecast_range, plots_dir, lag=lag)

        # Perform the NAO matching for the target variable
        match_var_ensemble_mean, _ = fnc.nao_matching_other_var(rescaled_nao, model_nao, psl_models, match_var, obs_var_name,
                                                                base_dir, match_var_models, obs_path_match_var, region, model_season, forecast_range,
                                                                start_year, end_year, plots_dir, save_dir, lagged_years=years,
                                                                lagged_nao=True, no_subset_members=no_subset_members, level=level,
                                                                ensemble_mean_nao=ensemble_mean_nao)
    elif match_type == "spna":
        print("SPNA matching for variable:", match_var, "region:", region,
              "season:", season, "forecast range:", forecast_range, "start year:",
              start_year, "end year:", end_year, "lag:", lag, "no subset members:",
              no_subset_members, "level:", level, "match type:", match_type)

        # Process the tas observations for the SPNA index
        obs_tas_anomaly = fnc.read_obs(variable="tas",
                                       region=region,
                                       forecast_range=forecast_range,
                                       season=season,
                                       observations_path=dic.obs,
                                       start_year=start_year,
                                       end_year=end_year)

        # Load and process the model data for the SPNA index
        model_datasets_tas = fnc.load_data(base_directory=base_dir,
                                           models=match_variable_models("tas"),
                                           variable="tas",
                                           region=region,
                                           forecast_range=forecast_range,
                                           season=model_season)

        # Process the model data - extract the tas variable
        model_data_tas, _ = fnc.process_data(datasets_by_model=model_datasets_tas,
                                             variable="tas")

        # Make sure that the models have the same time period for tas
        model_data_tas = fnc.constrain_years(model_data=model_data_tas,
                                             models=match_variable_models("tas"))

        # Remove years containing Nan values from the obs and model data
        obs_tas_anomaly, \
            model_data_tas, \
            _ = fnc.remove_years_with_nans_nao(observed_data=obs_tas_anomaly,
                                               model_data=model_data_tas,
                                               models=match_variable_models(
                                                   "tas"),
                                               NAO_matched=False)

        # Print the model data for debugging
        print("model_data_tas:", model_data_tas)
        # print("model_data_tas.shape:", model_data_tas.shape)
        print("obs_tas_anomaly:", obs_tas_anomaly)
        # print("obs_tas_anomaly.shape:", obs_tas_anomaly.shape)

        # Calculate the SPNA index and plot
        obs_spna, model_spna = fnc.calculate_spna_index_and_plot(obs_anom=obs_tas_anomaly,
                                                                 model_anom=model_data_tas,
                                                                 models=match_variable_models(
                                                                     "tas"),
                                                                 variable="tas",
                                                                 season=season,
                                                                 forecast_range=forecast_range,
                                                                 output_dir=dic.canari_plots_dir,
                                                                 plot_graphics=True)
        
        # Happy with the SPNA index calculation
        # Using the Strommen et al. (2023) grid box
        # (Smaller box than Smith et al. (2019))
        # Print the SPNA index outputs for debugging
        # print("obs_spna:", obs_spna)
        # print("model_spna:", model_spna)

        # Now perform the lagging for the SPNA index ensemble
        # No need to rescale the SPNA index as RPC ~1
        # Extract the data for a given model
        # Initialize an empty array to store the data
        model_spna_members = []

        # Initialize a counter
        counter = 0

        # Loop over the models
        for model in match_variable_models("tas"):
            # Extract the data for the given model
            model_spna_data = model_spna[model]

            # Loop over the ensemble members
            for member in model_spna_data:

                # If the counter is 0
                if counter == 0:
                    # Extract the coordinates
                    coords = member.coords

                    # Extract the dimensions
                    dims = member.dims

                    # Extract the years
                    years1 = member.time.dt.year.values

                # Extract the years
                years2 = member.time.dt.year.values

                # Assert that the years are the same
                assert np.all(years1 == years2), "The years are not the same."
                
                # Append the data to the list
                model_spna_members.append(member)

                # Increment the counter
                counter += 1

        # Convert the list to a numpy array
        model_spna_members = np.array(model_spna_members)

        # NOTE: Not going to lag the SPNA index for now - RPC ~1
        # Don't have to go through the same process of variance adjust
        # and lagging
        # We are going to assume that this is our 'best guess' for the SPNA index
        model_spna_mean = np.mean(model_spna_members, axis=0)








if __name__ == '__main__':
    main()
