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
    <lag> <no_subset_members> <method> <nboot> <level> <full_period>
    <nao_matched>

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
    full_period: bool
        Whether to use the full period or not.
        Must be a boolean in the input files.
    nao_matched: bool
        Whether to use the NAO matched data or not.
        Must be a boolean in the input files.
        Default is False.

Output:
=======

    A file containing the bootstrapped significance values 
    for the given variable.

"""

# Imports
import argparse
import os
import sys
import glob

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

# Import the functions for calculating the forecast stats
import paper1_plots_functions as p1_fnc

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
import pandas as pd


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
    parser.add_argument(
        "variable", type=str, help="The variable to perform the matching for."
    )

    parser.add_argument(
        "obs_var_name",
        type=str,
        help="The name of the variable in the \
                        observations file.",
    )

    parser.add_argument(
        "region", type=str, help="The region to perform the matching for."
    )

    parser.add_argument(
        "season", type=str, help="The season to perform the matching for."
    )

    parser.add_argument(
        "forecast_range",
        type=str,
        help="The forecast range to perform the matching for.",
    )

    parser.add_argument(
        "start_year", type=str, help="The start year to perform the matching for."
    )

    parser.add_argument(
        "end_year", type=str, help="The end year to perform the matching for."
    )

    parser.add_argument("lag", type=int, help="The lag to perform the matching for.")

    parser.add_argument(
        "no_subset_members",
        type=int,
        help="The number of ensemble members to subset to.",
    )

    parser.add_argument(
        "method", type=str, help="The method to use for the bootstrapping."
    )

    parser.add_argument("nboot", type=int, help="The number of bootstraps to perform.")

    # add optional argument for level, which defaults to None
    parser.add_argument(
        "level", type=str, default=None, help="The level to perform the matching for."
    )

    # add optional argument for full_period, which defaults to False
    parser.add_argument(
        "full_period",
        type=str,
        default="False",
        help="Whether to use the full period or not.",
    )

    # add optional argument for nao_matched, which defaults to False
    parser.add_argument(
        "nao_matched",
        type=str,
        default="False",
        help="Whether to use the NAO matched data or not.",
    )

    # Extract the CLAs
    args = parser.parse_args()

    # Return the CLAs
    return args


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

    # Extract the full period boolean
    full_period = args.full_period

    # Extract the NAO matched boolean
    nao_matched = args.nao_matched

    # Assert that full_period must be 'True' or 'False'
    assert full_period in ["True", "False"], (
        "Full period not recognised. Please try again." + "Must be 'True' or 'False'"
    )

    # Assert that nao_matched must be 'True' or 'False'
    assert nao_matched in ["True", "False"], (
        "NAO matched not recognised. Please try again." + "Must be 'True' or 'False'"
    )

    # Convert the full_period to a boolean
    if full_period == "True":
        full_period = True
    else:
        full_period = False

    # Convert the nao_matched to a boolean
    if nao_matched == "True":
        nao_matched = True
    else:
        nao_matched = False

    if level == "100000":
        # Set level to none
        level = None

    # If the region is global, set the region to the global gridspec
    if region == "global":
        region_grid = dicts.gridspec_global
    else:
        raise ValueError("Region not recognised. Please try again.")

    # Assert that method must be 'raw', 'lagged' or 'nao_matched'
    assert method in ["raw", "lagged", "alternate_lag", "nao_matched"], (
        "Method not recognised. Please try again."
        + "Must be 'raw', 'lagged', 'alternate_lag' or 'nao_matched'"
    )

    # If season conttains a number, convert it to the string
    season = p1_fnc.convert_season(season, dicts)

    # Print the variables
    print(
        "NAO matching for variable:",
        variable,
        "region:",
        region,
        "season:",
        season,
        "forecast range:",
        forecast_range,
        "start year:",
        start_year,
        "end year:",
        end_year,
        "lag:",
        lag,
        "no subset members:",
        no_subset_members,
        "method:",
        method,
    )

    # Set up the dcpp models
    dcpp_models = match_variable_models(variable)

    # Set up the historical models
    hist_models = p1_fnc.extract_hist_models(variable, dicts)

    # Set up the observations path for the matching variable
    obs_path_name = find_obs_path(variable)

    # Set nao_stats_dict to None
    nao_stats_dict = None

    # Set up the file names for the arrays
    corr1_name = f"corr1_{variable}_{region}_{season}_{forecast_range}.npy"

    corr1_p_name = f"corr1_p_{variable}_{region}_{season}_{forecast_range}.npy"

    corr2_name = f"corr2_{variable}_{region}_{season}_{forecast_range}.npy"

    corr2_p_name = f"corr2_p_{variable}_{region}_{season}_{forecast_range}.npy"

    corr10_name = f"corr10_{variable}_{region}_{season}_{forecast_range}.npy"

    corr10_p_name = f"corr10_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    partial_r_name = f"partial_r_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Partial r min and max values
    partial_r_min_name = (
        f"partial_r_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    partial_r_max_name = (
        f"partial_r_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    # Also the partial r bias
    partial_r_bias_name = (
        f"partial_r_bias_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    obs_resid_name = f"obs_resid_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Save the forecaast 1 residual array
    fcst1_em_resid_name = (
        f"fcst1_em_resid_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    partial_r_p_name = (
        f"partial_r_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    sigo = f"sigo_{variable}_{region}_{season}_{forecast_range}.npy"

    sigo_resid = f"sigo_resid_{variable}_{region}_{season}_" + f"{forecast_range}.npy"

    # Also save arrays for the correlation differences
    corr_diff_name = f"corr_diff_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Min and max values
    corr_diff_min_name = (
        f"corr_diff_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    corr_diff_max_name = (
        f"corr_diff_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    corr_diff_p_name = (
        f"corr_diff_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    # Also save arrays for the RPC and RPC_p
    rpc1_name = f"rpc1_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Min and max arrays
    rpc1_min_name = f"rpc1_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    rpc1_max_name = f"rpc1_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    rpc1_p_name = f"rpc1_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    rpc2_name = f"rpc2_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Min and max arrays
    rpc2_min_name = f"rpc2_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    rpc2_max_name = f"rpc2_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    rpc2_p_name = f"rpc2_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the arrays for MSSS1 and MSSS2

    msss1_name = f"msss1_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the arrays for the min and max values of MSS1 and MSS2
    msss1_min_name = f"msss1_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    msss1_max_name = f"msss1_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the arrays for the MSSS1 and MSSS2 p values
    msss1_p_name = f"msss1_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the corr1 and corr2 min and max values
    corr1_min_name = f"corr1_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    corr1_max_name = f"corr1_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    corr2_min_name = f"corr2_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    corr2_max_name = f"corr2_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the corr10 min and max values
    corr10_min_name = (
        f"corr10_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    corr10_max_name = (
        f"corr10_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    corr12_name = f"corr12_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Also save the corr12 min and max values
    corr12_min_name = (
        f"corr12_min_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    corr12_max_name = (
        f"corr12_max_{variable}_{region}_{season}_{forecast_range}" + ".npy"
    )

    # Also save the corr12 p values
    corr12_p_name = f"corr12_p_{variable}_{region}_{season}_{forecast_range}" + ".npy"

    # Set up the names for the forecast time series
    fcst1_ts_name = f"fcst1_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    fcst2_ts_name = f"fcst2_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    fcst10_ts_name = f"fcst10_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    obs_ts_name = f"obs_ts_{variable}_{region}_{season}_{forecast_range}.npy"

    # Set up the names for the values of the forecast stats
    nens1_name = f"nens1_{variable}_{region}_{season}_{forecast_range}.txt"

    nens2_name = f"nens2_{variable}_{region}_{season}_{forecast_range}.txt"

    start_end_years = (
        f"start_end_years_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
    )

    # Set the names for the new variables
    start_end_years_short = (
        f"start_end_years_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
    )

    # Set up the names  for the short time series
    fcst1_ts_short_name = (
        f"fcst1_ts_{variable}_{region}_{season}_{forecast_range}_short.npy"
    )

    obs_ts_short_name = (
        f"obs_ts_{variable}_{region}_{season}_{forecast_range}_short.npy"
    )

    corr1_short_name = f"corr1_{variable}_{region}_{season}_{forecast_range}_short.npy"

    corr1_p_short_name = (
        f"corr1_p_{variable}_{region}_{season}_{forecast_range}_short.npy"
    )

    # If the method is 'alternate_lag'
    if method == "alternate_lag":
        print("Loading alternate lagged data")

        # Set alt_lag_data to None
        alt_lag_data = None

        # Set up the directory
        # TODO: hardcoded for now
        # for summer months AMJJAS all in normal dir
        # alt_lag_dir = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/"
        alt_lag_dir = (
            "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/test-sfcWind"
        )

        # Extract the first year of the array
        alt_lag_first_year = int(start_year) + (lag - 1)

        # Extract the last year of the array
        alt_lag_last_year = int(end_year)

        if level is not None:
            # Set up the filename
            # ua_DJFM_global_1961_1961_2-9_4_85000_1712048833.931586.npy
            raw_filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{level}*.npy"
        else:
            # Set up the filename
            # ua_DJFM_global_1961_1961_2-9_4_1712048833.931586.npy
            raw_filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}*.npy"

        # Find files matching the raw filename
        raw_files = glob.glob(alt_lag_dir + "/" + raw_filename)

        if "-" in forecast_range:
            if level is not None:
                # Set up the filename
                # ua_DJFM_global_1961_1961_2-9_4_85000_1712048833.931586.npy
                alt_lag_filename = f"{variable}_{season}_{region}_{alt_lag_first_year}_{alt_lag_last_year}_{forecast_range}_{lag}_{level}*.npy"
            else:
                # Set up the filename
                # ua_DJFM_global_1961_1961_2-9_4_1712048833.931586.npy
                alt_lag_filename = f"{variable}_{season}_{region}_{alt_lag_first_year}_{alt_lag_last_year}_{forecast_range}_{lag}*.npy"

            # Find files matching the filename
            alt_lag_files = glob.glob(alt_lag_dir + "/" + alt_lag_filename)

            # If there is more than one file
            if len(alt_lag_files) > 1:
                print("More than one file found")

                # If the psl_DJFM_global_1962_1980_2-9_2_1706281292.628301_alternate_lag.npy
                # 1706281292.628301 is the datetime
                # Extract the datetimes
                datetimes = [file.split("_")[7] for file in alt_lag_files]

                # Remove the .npy from the datetimes
                datetimes = [datetime.split(".")[0] for datetime in datetimes]

                # Convert the datasetimes to datetimes using pandas
                datetimes = [
                    pd.to_datetime(datetime, unit="s") for datetime in datetimes
                ]

                # Find the latest datetime
                latest_datetime = max(datetimes)

                # Find the index of the latest datetime
                latest_datetime_index = datetimes.index(latest_datetime)

                # Print that we are using the latest datetime file
                print(
                    "Using the latest datetime file:",
                    alt_lag_files[latest_datetime_index],
                )

                # Load the file
                alt_lag_data = np.load(alt_lag_files[latest_datetime_index])
            elif len(alt_lag_files) == 1:
                # Load the file
                alt_lag_data = np.load(alt_lag_files[0])

                # print the shape of the alt lag data
                print("Shape of alt lag data:", alt_lag_data.shape)
            else:
                print("No files found for alternate lag data")

        # If there is more than one file
        if len(raw_files) > 1:
            print("More than one file found")

            # If the psl_DJFM_global_1962_1980_2-9_2_1706281292.628301_alternate_lag.npy
            # 1706281292.628301 is the datetime
            # Extract the datetimes
            datetimes = [file.split("_")[7] for file in raw_files]

            # Remove the .npy from the datetimes
            datetimes = [datetime.split(".")[0] for datetime in datetimes]

            # Convert the datasetimes to datetimes using pandas
            datetimes = [pd.to_datetime(datetime, unit="s") for datetime in datetimes]

            # Find the latest datetime
            latest_datetime = max(datetimes)

            # Find the index of the latest datetime
            latest_datetime_index = datetimes.index(latest_datetime)

            # Print that we are using the latest datetime file
            print("Using the latest datetime file:", raw_files[latest_datetime_index])

            # Load the file
            raw_data = np.load(raw_files[latest_datetime_index])
        elif len(raw_files) == 1:
            # Load the file
            raw_data = np.load(raw_files[0])

            # Print the shape of the raw data
            print("Shape of raw data:", raw_data.shape)
        else:
            print("No files found for raw data")

        # if the season is ULG, change it to JJA
        if season == "ULG":
            season = "JJA"
        elif season == "AYULGS":
            season = "AMJJAS"
        elif season == "MAY":
            season = "MAM"

        # Process the observations for this variable
        # TODO: check this for forecast range = 1,2
        obs = fnc.process_observations(
            variable=variable,
            region=region,
            region_grid=region_grid,
            forecast_range=forecast_range,
            season=season,
            observations_path=obs_path_name,
            obs_var_name=variable,
            plev=level,
        )

        # If the window size for rolling mean is even, e.g. 2
        # Then the latter place from the medium will be taken for the label
        # E.g. for a rolling mean over:
        # [1960], [1961], [1962], [1963] (window = 4)
        # The label will be 1962
        # Depending on the forecast range, change the alt lag last year
        # TODO: Check this is correct
        if forecast_range == "2-9":
            # Set up the alt lag first and last years
            alt_lag_first_year = alt_lag_first_year + 5
            alt_lag_last_year = alt_lag_last_year + 5

            # Set up the raw first and last years
            raw_first_year = int(start_year) + 5
            raw_last_year = int(end_year) + 5
        elif forecast_range == "2-5":
            # Set up the alt lag first and last years
            alt_lag_first_year = alt_lag_first_year + 3
            alt_lag_last_year = alt_lag_last_year + 3

            # Set up the raw first and last years
            raw_first_year = int(start_year) + 3
            raw_last_year = int(end_year) + 3
        elif forecast_range == "2-3":
            # Set up the alt lag first and last years
            alt_lag_first_year = alt_lag_first_year + 2
            alt_lag_last_year = alt_lag_last_year + 2

            # Set up the raw first and last years
            raw_first_year = int(start_year) + 2
            raw_last_year = int(end_year) + 2
        # FIXME: Add the other forecast ranges
        elif forecast_range in ["1", "2"] and season in ["DJFM", "DJF", "ONDJFM"]:
            # Set up the raw first and last years
            raw_first_year = int(start_year)
            raw_last_year = int(end_year)
        elif forecast_range in ["1", "2"] and season not in ["DJFM", "DJF", "ONDJFM"]:
            # Set up the raw first and last years
            raw_first_year = int(start_year) + 1
            raw_last_year = int(end_year) + 1
        else:
            raise ValueError("Forecast range not recognised. Please try again.")

        # If the forecast range is 2
        if forecast_range == "2":
            # alt_lag_first_year = alt_lag_first_year + 1
            # alt_lag_last_year = alt_lag_last_year + 1

            # Set up the raw first and last years
            raw_first_year = raw_first_year + 1
            raw_last_year = raw_last_year + 1

        # Print the start and end years
        print("Start year alt lag:", alt_lag_first_year)
        print("End year alt lag:", alt_lag_last_year)

        print("Start year raw:", raw_first_year)
        print("End year raw:", raw_last_year)

        # Set up common years
        common_years_alt_lag = np.arange(alt_lag_first_year, alt_lag_last_year + 1)

        # Set up common years
        common_years_raw = np.arange(raw_first_year, raw_last_year + 1)

        # Create a copy of the obs
        obs_copy = obs.copy()

        # Constraint the observations to the common years
        obs_lag = obs_copy.sel(
            time=slice(f"{alt_lag_first_year}-01-01", f"{alt_lag_last_year}-12-31")
        )

        # Constrain the observations to the common years of the raw data
        obs_raw = obs_copy.sel(
            time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31")
        )

        # Loop over the obs to check that there are no nans
        for year in obs_lag.time.dt.year.values:
            # Extract the data for the year
            year_data = obs.sel(time=f"{year}")

            # If there are any nans, raise an error
            if np.isnan(year_data).any():
                print("Nans found in obs for year:", year)
                if np.isnan(year_data).all():
                    print("All values are nan")
                    print("Removing year:", year, "from obs")
                    obs = obs.sel(time=obs.time.dt.year != year)

        # Loop over the obs to check that there are no nans
        for year in obs_raw.time.dt.year.values:
            # Extract the data for the year
            year_data = obs.sel(time=f"{year}")

            # If there are any nans, raise an error
            if np.isnan(year_data).any():
                print("Nans found in obs for year:", year)
                if np.isnan(year_data).all():
                    print("All values are nan")
                    print("Removing year:", year, "from obs")
                    obs = obs.sel(time=obs.time.dt.year != year)

        # Extract the values for the obs
        obs_lag_values = obs_lag.values

        # Extract the values for the obs
        obs_values = obs_raw.values

        # check whether historical data exists
        # hardcoded path sorry
        saved_hist_dir = (
            "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/saved_files/arrays/"
        )

        # Check whether files exist for the variable
        hist_files_dir = f"{saved_hist_dir}{variable}/{season}/{forecast_range}/"

        # set hist_data_raw and hist_data_lag to None
        hist_data_raw = None
        hist_data_lag = None

        # check whether the directory exists
        if os.path.exists(hist_files_dir):
            print(
                "Historical files exist for variable:",
                variable,
                "season:",
                season,
                "forecast range:",
                forecast_range,
            )

            # psl_ONDJFM_2-9_1961-2023_historical_ssp245_lag.npy

            # load in the historical data (raw historical data)
            hist_data_raw_path = f"{hist_files_dir}{variable}_{season}_{forecast_range}_????-????_historical_ssp245_raw.npy"

            # set up the lag path (raw data constrained to lag period)
            hist_data_lag_path = f"{hist_files_dir}{variable}_{season}_{forecast_range}_????-????_historical_ssp245_lag.npy"

            # glob the files
            hist_data_raw_files = glob.glob(hist_data_raw_path)

            # glob the files
            hist_data_lag_files = glob.glob(hist_data_lag_path)

            # assert that only one file exists for each
            assert (
                len(hist_data_raw_files) == 1
            ), "More than one file found for historical raw data"

            # assert that only one file exists for each
            assert (
                len(hist_data_lag_files) == 1
            ), "More than one file found for historical lag data"

            # load in the historical data
            hist_data_raw = np.load(hist_data_raw_files[0])

            # load in the historical data
            hist_data_lag = np.load(hist_data_lag_files[0])

            # In the case of EC-Earth for tas, this starts in 1970, so is not
            # suitable - remove these members
            nmems = hist_data_raw.shape[0]

            no_dropped_mems = 0

            # Initialize a list to hold the indices of members to drop
            drop_indices = []

            for mem in range(nmems):
                # If the values are all nans, add the member index to drop_indices
                if np.isnan(hist_data_raw[mem, 0, :, :]).all():
                    print("All values are nan for member:", mem)
                    drop_indices.append(mem)

                    # Increment the number of dropped members
                    no_dropped_mems += 1

            # Drop the members
            hist_data_raw_updated = np.delete(hist_data_raw, drop_indices, axis=0)
            hist_data_lag_updated = np.delete(hist_data_lag, drop_indices, axis=0)

            # Print the number of dropped members
            print("Number of dropped members:", len(drop_indices))

            # Reset the names
            hist_data_raw = hist_data_raw_updated
            hist_data_lag = hist_data_lag_updated

            # Print the shape of the historical data
            print("Shape of historical raw data updated:", hist_data_raw.shape)

            # Print the shape of the historical data
            print("Shape of historical lag data updated:", hist_data_lag.shape)

        # If NAO matching is True
        if nao_matched:
            print("NAO matching is True")
            print("Processing the NAO matched data")

            # Set up the filename of the NAO matched data
            nao_m_filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_20_*_nao_matched_members.npy"

            # Find the files matching the filename
            nao_m_files = glob.glob(os.path.join(alt_lag_dir, nao_m_filename))

            # If there is more than one file
            if len(nao_m_files) > 1:
                print("More than one file found for NAO matched data")

                # Extract the datetimes
                datetimes = [file.split("_")[8] for file in nao_m_files]

                # Convert the datetimes to datetimes using pandas
                datetimes = [
                    pd.to_datetime(datetime, unit="s") for datetime in datetimes
                ]

                # Find the latest datetime
                latest_datetime = max(datetimes)

                # Find the index of the latest datetime
                latest_datetime_index = datetimes.index(latest_datetime)

                # Print that we are using the latest datetime file
                print(
                    "Using the latest datetime file:",
                    nao_m_files[latest_datetime_index],
                )

                # Load the file
                nao_matched_data = np.load(nao_m_files[latest_datetime_index])
            elif len(nao_m_files) == 1:
                # Load the file
                nao_matched_data = np.load(nao_m_files[0])
            else:
                raise ValueError("No files found for NAO matched data")

            # Print the shape of the NAO matched data
            print("Shape of NAO matched data:", nao_matched_data.shape)

            # Verify that the length of the observations is correct
            assert (
                len(obs_lag.time.dt.year.values) == nao_matched_data.shape[1]
            ), "Length of observations is incorrect for NAO matched data"

            # Print the shape of the nao_matched_data
            print("Shape of nao_matched_data:", nao_matched_data.shape)

            # Print the shape of the obs_lag_values
            print("Shape of obs_lag_values:", obs_lag_values.shape)

            if hist_data_lag is not None:
                # Run the function to calculate the forecast stats
                forecast_stats_nao_matched = fnc.forecast_stats(
                    obs=obs_lag_values,
                    forecast1=nao_matched_data,
                    forecast2=hist_data_lag,
                    no_boot=no_bootstraps,
                )
            else:
                # Run the function to calculate the forecast stats
                forecast_stats_nao_matched = fnc.forecast_stats(
                    obs=obs_lag_values,
                    forecast1=nao_matched_data,
                    forecast2=nao_matched_data,
                    no_boot=no_bootstraps,
                )

            # Set up the save path
            save_path_nao_matched = (
                save_dir
                + "/"
                + variable
                + "/"
                + region
                + "/"
                + season
                + "/"
                + forecast_range
                + "/"
                + "nao_matched"
                + "/"
                + "no_bootstraps_"
                + str(no_bootstraps)
                + "/"
            )

            # If the save path doesn't exist, create it
            if not os.path.exists(save_path_nao_matched):
                os.makedirs(save_path_nao_matched)

            # Save the important forecast stats
            np.save(
                save_path_nao_matched + corr1_name, forecast_stats_nao_matched["corr1"]
            )

            # Save the corr1_p values
            np.save(
                save_path_nao_matched + corr1_p_name,
                forecast_stats_nao_matched["corr1_p"],
            )

            # Save the corr1_min values
            np.save(
                save_path_nao_matched + corr1_min_name,
                forecast_stats_nao_matched["corr1_min"],
            )

            # Save the corr1_max values
            np.save(
                save_path_nao_matched + corr1_max_name,
                forecast_stats_nao_matched["corr1_max"],
            )

            # Save the corr2 values
            np.save(
                save_path_nao_matched + corr2_name, forecast_stats_nao_matched["corr2"]
            )

            # Save the corr2_p values
            np.save(
                save_path_nao_matched + corr2_p_name,
                forecast_stats_nao_matched["corr2_p"],
            )

            # Save the corr2_min values
            np.save(
                save_path_nao_matched + corr2_min_name,
                forecast_stats_nao_matched["corr2_min"],
            )

            # Save the corr2_max values
            np.save(
                save_path_nao_matched + corr2_max_name,
                forecast_stats_nao_matched["corr2_max"],
            )

            # Save the corr10 values
            np.save(
                save_path_nao_matched + corr10_name,
                forecast_stats_nao_matched["corr10"],
            )

            # Save the corr10_p values
            np.save(
                save_path_nao_matched + corr10_p_name,
                forecast_stats_nao_matched["corr10_p"],
            )

            # Save the corr10_min values
            np.save(
                save_path_nao_matched + corr10_min_name,
                forecast_stats_nao_matched["corr10_min"],
            )

            # Save the corr10_max values
            np.save(
                save_path_nao_matched + corr10_max_name,
                forecast_stats_nao_matched["corr10_max"],
            )

            # Save the msss1 values
            np.save(
                save_path_nao_matched + msss1_name,
                forecast_stats_nao_matched["msss1"],
            )

            # Save the msss1_p values
            np.save(
                save_path_nao_matched + msss1_p_name,
                forecast_stats_nao_matched["msss1_p"],
            )

            # Save the msss1_min values
            np.save(
                save_path_nao_matched + msss1_min_name,
                forecast_stats_nao_matched["msss1_min"],
            )

            # Save the msss1_max values
            np.save(
                save_path_nao_matched + msss1_max_name,
                forecast_stats_nao_matched["msss1_max"],
            )

            # Save the corr12 values
            np.save(
                save_path_nao_matched + corr12_name,
                forecast_stats_nao_matched["corr12"],
            )

            # Save the corr12_p values
            np.save(
                save_path_nao_matched + corr12_p_name,
                forecast_stats_nao_matched["corr12_p"],
            )

            # Save the corr12_min values
            np.save(
                save_path_nao_matched + corr12_min_name,
                forecast_stats_nao_matched["corr12_min"],
            )

            # Save the corr12_max values
            np.save(
                save_path_nao_matched + corr12_max_name,
                forecast_stats_nao_matched["corr12_max"],
            )

            # Save the rpc1
            np.save(
                save_path_nao_matched + rpc1_name, forecast_stats_nao_matched["rpc1"]
            )

            # And the rpc1_p
            np.save(
                save_path_nao_matched + rpc1_p_name,
                forecast_stats_nao_matched["rpc1_p"],
            )

            # And the rpc1_min
            np.save(
                save_path_nao_matched + rpc1_min_name,
                forecast_stats_nao_matched["rpc1_min"],
            )

            # And the rpc1_max
            np.save(
                save_path_nao_matched + rpc1_max_name,
                forecast_stats_nao_matched["rpc1_max"],
            )

            # Save the rpc2
            np.save(
                save_path_nao_matched + rpc2_name, forecast_stats_nao_matched["rpc2"]
            )

            # And the rpc2_p
            np.save(
                save_path_nao_matched + rpc2_p_name,
                forecast_stats_nao_matched["rpc2_p"],
            )

            # And the rpc2_min
            np.save(
                save_path_nao_matched + rpc2_min_name,
                forecast_stats_nao_matched["rpc2_min"],
            )

            # And the rpc2_max
            np.save(
                save_path_nao_matched + rpc2_max_name,
                forecast_stats_nao_matched["rpc2_max"],
            )

            # Save the corr_diff
            np.save(
                save_path_nao_matched + corr_diff_name,
                forecast_stats_nao_matched["corr_diff"],
            )

            # Save the corr_diff_min
            np.save(
                save_path_nao_matched + corr_diff_min_name,
                forecast_stats_nao_matched["corr_diff_min"],
            )

            # Save the corr_diff_max
            np.save(
                save_path_nao_matched + corr_diff_max_name,
                forecast_stats_nao_matched["corr_diff_max"],
            )

            # Save the corr_diff_p
            np.save(
                save_path_nao_matched + corr_diff_p_name,
                forecast_stats_nao_matched["corr_diff_p"],
            )

            # Save the partial_r
            np.save(
                save_path_nao_matched + partial_r_name,
                forecast_stats_nao_matched["partialr"],
            )

            # Save the partial_r_min
            np.save(
                save_path_nao_matched + partial_r_min_name,
                forecast_stats_nao_matched["partialr_min"],
            )

            # Save the partial_r_max
            np.save(
                save_path_nao_matched + partial_r_max_name,
                forecast_stats_nao_matched["partialr_max"],
            )

            # Save the partial_r_bias
            np.save(
                save_path_nao_matched + partial_r_bias_name,
                forecast_stats_nao_matched["partialr_bias"],
            )

            # Save the sig_o
            np.save(save_path_nao_matched + sigo, forecast_stats_nao_matched["sigo"])

            # Save the sig_o_resid
            np.save(
                save_path_nao_matched + sigo_resid,
                forecast_stats_nao_matched["sigo_resid"],
            )

            # Save the obs_resid
            np.save(
                save_path_nao_matched + obs_resid_name,
                forecast_stats_nao_matched["obs_resid"],
            )

            # Save the forecast 1 em resid
            np.save(
                save_path_nao_matched + fcst1_em_resid_name,
                forecast_stats_nao_matched["fcst1_em_resid"],
            )

            # Save the partial_r_p
            np.save(
                save_path_nao_matched + partial_r_p_name,
                forecast_stats_nao_matched["partialr_p"],
            )

            # save the f2 ts
            np.save(
                save_path_nao_matched + fcst2_ts_name,
                forecast_stats_nao_matched["f2_ts"],
            )

            # save the f10 ts
            np.save(
                save_path_nao_matched + fcst10_ts_name,
                forecast_stats_nao_matched["f10_ts"],
            )

            # Save the common years
            np.save(save_path_nao_matched + "common_years.npy", common_years_alt_lag)

            # Save the forecast time series
            np.save(
                save_path_nao_matched + fcst1_ts_name,
                forecast_stats_nao_matched["f1_ts"],
            )

            # Save the observed time series
            np.save(
                save_path_nao_matched + obs_ts_name, forecast_stats_nao_matched["o_ts"]
            )

            # Set up the nens1
            nens1_nao_matched = nao_matched_data.shape[0]

            # Save the nens1
            np.savetxt(
                save_path_nao_matched + nens1_name, np.array([nens1_nao_matched])
            )

            # Save the nens2
            np.savetxt(
                save_path_nao_matched + nens2_name, np.array([nens1_nao_matched])
            )

            # Save the start and end years
            np.savetxt(
                save_path_nao_matched + start_end_years_short,
                np.array([alt_lag_first_year, alt_lag_last_year - 10 + 1]),
            )

            # Save the short time series
            np.save(
                save_path_nao_matched + fcst1_ts_short_name,
                forecast_stats_nao_matched["f1_ts_short"],
            )

            # Save the observed time series short
            np.save(
                save_path_nao_matched + obs_ts_short_name,
                forecast_stats_nao_matched["o_ts_short"],
            )

            # Save the corr1 short
            np.save(
                save_path_nao_matched + corr1_short_name,
                forecast_stats_nao_matched["corr1_short"],
            )

            # Save the corr1_p short
            np.save(
                save_path_nao_matched + corr1_p_short_name,
                forecast_stats_nao_matched["corr1_p_short"],
            )

            # FIXME: Fix the exiting here
            # Print that the arrays have been saved to the save path
            print("Arrays saved to:", save_path_nao_matched)
            print("For NAO matched data")
            print("EXITING")
            sys.exit()

        # If alt_lag_data exists
        if alt_lag_data is not None:
            print("Alt lag data exists")
            # Verify that the length of the observations is correct
            assert (
                len(obs_lag.time.dt.year.values) == alt_lag_data.shape[0]
            ), "Length of observations is incorrect"

            # Swap the axes of the alt_lag_data
            # Swap the 1th axis with the 0th axis
            alt_lag_data = np.swapaxes(alt_lag_data, 1, 0)

            # Print the shape of the alt_lag_data
            print("Shape of alt_lag_data:", alt_lag_data.shape)

            # Print the shape of the alt lag data
            print("Shape of alt_lag_data:", alt_lag_data.shape)

            # NOTE: Temporary fix on nens
            nens1_alt_lag = alt_lag_data.shape[0]

            if hist_data_lag is not None:
                # Run the function to calculate the forecast stats
                forecast_stats_alt_lag = fnc.forecast_stats(
                    obs=obs_lag_values,
                    forecast1=alt_lag_data,
                    forecast2=hist_data_lag,
                    no_boot=no_bootstraps,
                )
            else:
                # Run the function to calculate the forecast stats
                forecast_stats_alt_lag = fnc.forecast_stats(
                    obs=obs_lag_values,
                    forecast1=alt_lag_data,
                    forecast2=alt_lag_data,
                    no_boot=no_bootstraps,
                )

        # Verify that the length of the observations is correct
        assert (
            len(obs_raw.time.dt.year.values) == raw_data.shape[0]
        ), "Length of observations is incorrect"

        # Swap the axes of the raw_data
        # Swap the 1th axis with the 0th axis
        raw_data = np.swapaxes(raw_data, 1, 0)

        # Print the shape of the raw_data
        print("Shape of raw_data:", raw_data.shape)

        # First take the mean over the year axis for the raw data
        if forecast_range == "2-3":
            raw_data_mean = raw_data[:, :, :2, :, :].mean(axis=2)
        elif forecast_range == "2-5":
            raw_data_mean = raw_data[:, :, :4, :, :].mean(axis=2)
        elif forecast_range == "2-9":
            raw_data_mean = raw_data[:, :, :8, :, :].mean(axis=2)
        elif forecast_range in ["1", "2"]:
            raw_data_mean = raw_data.mean(axis=2)
        else:
            raise ValueError("Forecast range not recognised. Please try again.")

        # Print the shape of the obs_values
        print("Shape of obs_values lag:", obs_lag_values.shape)

        # Print the shape of the obs_values
        print("Shape of obs_values raw:", obs_values.shape)

        # Print the shape of the raw data
        print("Shape of raw_data_mean:", raw_data_mean.shape)

        # NOTE: Temporary fix on nens
        nens1_raw = raw_data_mean.shape[0]

        if hist_data_raw is not None:
            # Run the function to calculate the forecast stats
            forecast_stats_raw = fnc.forecast_stats(
                obs=obs_values,
                forecast1=raw_data_mean,
                forecast2=hist_data_raw,
                no_boot=no_bootstraps,
            )
        else:
            # Run the function to calculate the forecast stats
            forecast_stats_raw = fnc.forecast_stats(
                obs=obs_values,
                forecast1=raw_data_mean,
                forecast2=raw_data_mean,
                no_boot=no_bootstraps,
            )

        # Set up the save path
        save_path_alt_lag = (
            save_dir
            + "/"
            + variable
            + "/"
            + region
            + "/"
            + season
            + "/"
            + forecast_range
            + "/"
            + "alt_lag"
            + "/"
            + "no_bootstraps_"
            + str(no_bootstraps)
            + "/"
        )

        # Set up the save path
        save_path_raw = (
            save_dir
            + "/"
            + variable
            + "/"
            + region
            + "/"
            + season
            + "/"
            + forecast_range
            + "/"
            + "new_raw"
            + "/"
            + "no_bootstraps_"
            + str(no_bootstraps)
            + "/"
        )

        # If the save path doesn't exist, create it
        if not os.path.exists(save_path_alt_lag):
            os.makedirs(save_path_alt_lag)

        # If the save path doesn't exist, create it
        if not os.path.exists(save_path_raw):
            os.makedirs(save_path_raw)

        if alt_lag_data is not None:
            # Store the paths in a list
            save_paths = [save_path_alt_lag, save_path_raw]

            # Store the forecast stats in a list
            forecast_stats = [forecast_stats_alt_lag, forecast_stats_raw]

            # Form the list of nens
            nens = [nens1_alt_lag, nens1_raw]

            # Create a list of the common years
            common_years = [common_years_alt_lag, common_years_raw]
        else:
            # Store the paths in a list
            save_paths = [save_path_raw]

            # Store the forecast stats in a list
            forecast_stats = [forecast_stats_raw]

            # Form the list of nens
            nens = [nens1_raw]

            # Create a list of the common years
            common_years = [common_years_raw]

        # Loop over the save paths and forecast stats
        for save_path, forecast_stat, common_year, nen in zip(
            save_paths, forecast_stats, common_years, nens
        ):
            print("Saving the arrays to:", save_path)
            print("Common years:", common_year)
            print(forecast_stat)
            # Save the arrays
            # if the file already exists, don't overwrite it
            np.save(save_path + corr1_name, forecast_stat["corr1"])

            # Save the min and max values
            np.save(save_path + corr1_min_name, forecast_stat["corr1_min"])

            np.save(save_path + corr1_max_name, forecast_stat["corr1_max"])

            np.save(save_path + corr1_p_name, forecast_stat["corr1_p"])

            np.save(save_path + corr2_name, forecast_stat["corr2"])

            # Save the min and max values
            np.save(save_path + corr2_min_name, forecast_stat["corr2_min"])

            np.save(save_path + corr2_max_name, forecast_stat["corr2_max"])

            np.save(save_path + corr2_p_name, forecast_stat["corr2_p"])

            np.save(save_path + corr10_name, forecast_stat["corr10"])

            # Save the min and max values
            np.save(save_path + corr10_min_name, forecast_stat["corr10_min"])

            np.save(save_path + corr10_max_name, forecast_stat["corr10_max"])

            np.save(save_path + corr10_p_name, forecast_stat["corr10_p"])

            np.save(save_path + corr12_name, forecast_stat["corr12"])

            # Save the min and max values
            np.save(save_path + corr12_min_name, forecast_stat["corr12_min"])

            np.save(save_path + corr12_max_name, forecast_stat["corr12_max"])

            np.save(save_path + corr12_p_name, forecast_stat["corr12_p"])

            # Save the MSSS1 and MSSS2 arrays
            np.save(save_path + msss1_name, forecast_stat["msss1"])

            # Save the min and max values
            np.save(save_path + msss1_min_name, forecast_stat["msss1_min"])

            np.save(save_path + msss1_max_name, forecast_stat["msss1_max"])

            np.save(save_path + msss1_p_name, forecast_stat["msss1_p"])

            # Save the RPC1 and RPC2 arrays
            np.save(save_path + rpc1_name, forecast_stat["rpc1"])

            # Save the min and max values
            np.save(save_path + rpc1_min_name, forecast_stat["rpc1_min"])

            np.save(save_path + rpc1_max_name, forecast_stat["rpc1_max"])

            np.save(save_path + rpc1_p_name, forecast_stat["rpc1_p"])

            np.save(save_path + rpc2_name, forecast_stat["rpc2"])

            # Save the min and max values
            np.save(save_path + rpc2_min_name, forecast_stat["rpc2_min"])

            np.save(save_path + rpc2_max_name, forecast_stat["rpc2_max"])

            np.save(save_path + rpc2_p_name, forecast_stat["rpc2_p"])

            # Save the corr_diff arrays
            np.save(save_path + corr_diff_name, forecast_stat["corr_diff"])

            # Save the min and max values
            np.save(save_path + corr_diff_min_name, forecast_stat["corr_diff_min"])

            np.save(save_path + corr_diff_max_name, forecast_stat["corr_diff_max"])

            np.save(save_path + corr_diff_p_name, forecast_stat["corr_diff_p"])

            # Save the partial r min and max values
            np.save(save_path + partial_r_min_name, forecast_stat["partialr_min"])

            np.save(save_path + partial_r_max_name, forecast_stat["partialr_max"])

            # Save the partial r bias
            np.save(save_path + partial_r_bias_name, forecast_stat["partialr_bias"])

            # Save the partial r and partial r p values
            np.save(save_path + partial_r_name, forecast_stat["partialr"])

            np.save(save_path + partial_r_p_name, forecast_stat["partialr_p"])

            # Save the obs residual array
            np.save(save_path + obs_resid_name, forecast_stat["obs_resid"])

            # Save the forecast1 residual array
            np.save(save_path + fcst1_em_resid_name, forecast_stat["fcst1_em_resid"])

            # Save the sigo and sigo residual arrays
            np.save(save_path + sigo, forecast_stat["sigo"])

            np.save(save_path + sigo_resid, forecast_stat["sigo_resid"])

            # Save the values of the forecast stats
            np.savetxt(save_path + nens1_name, np.array([forecast_stat["nens1"]]))

            np.savetxt(save_path + nens2_name, np.array([forecast_stat["nens2"]]))

            np.savetxt(save_path + start_end_years, [common_year[0], common_year[-1]])

            # Save the common years for the short time series
            np.savetxt(
                save_path + start_end_years_short, [common_year[0], common_year[-10]]
            )

            # Save the forecast time series
            np.save(save_path + fcst1_ts_name, forecast_stat["f1_ts"])

            np.save(save_path + fcst2_ts_name, forecast_stat["f2_ts"])

            np.save(save_path + fcst10_ts_name, forecast_stat["f10_ts"])

            np.save(save_path + obs_ts_name, forecast_stat["o_ts"])

            # Save the short time series
            np.save(
                save_path + fcst1_ts_short_name,
                forecast_stat["f1_ts_short"],
            )

            # Save the observed time series short
            np.save(
                save_path + obs_ts_short_name,
                forecast_stat["o_ts_short"],
            )

            # Save the corr1 short
            np.save(
                save_path + corr1_short_name,
                forecast_stat["corr1_short"],
            )

            # Save the corr1_p short
            np.save(
                save_path + corr1_p_short_name,
                forecast_stat["corr1_p_short"],
            )

        # Print that the arrays have been saved
        print("Arrays saved to:", save_path)
        print("For alternate lag:", alt_lag_first_year, alt_lag_last_year)
        print("Finished saving arrays")

        # Exit here
        sys.exit()

        # TODO: also save the output for the NAO time series
    else:
        # If full_period is True, process the raw data
        # for the long period (s1961-2014 for years 2-9)
        # TODO: Processing for the lagged data as well
        if full_period:
            print("Processing the raw data for the full period")

            # Create a list containing the single variable specified
            variables = [variable]

            # Call the function
            # TODO: test for single bootstrap in this case
            forecast_stats, nao_stats_dict = p1_fnc.forecast_stats_var(
                variables=variables,
                season=season,
                forecast_range=forecast_range,
                region=region,
                start_year=start_year,
                end_year=end_year,
                method=method,
                no_bootstraps=no_bootstraps,
            )
        else:
            print("Processing the raw data for the overlapping hist period")

            # Process the observed data
            obs = fnc.process_observations(
                variable,
                region,
                region_grid,
                forecast_range,
                season,
                obs_path_name,
                variable,
                plev=level,
            )

            # if the variable is 'rsds'
            # divide the obs data by 86400 to convert from J/m2 to W/m2
            if variable in ["rsds", "ssrd"]:
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
            hist_data = p1_fnc.load_and_process_hist_data(
                base_dir_historical,
                hist_models,
                variable,
                region,
                forecast_range,
                season,
            )

            # Set up the constrained historical data (contain only the common years)
            constrained_hist_data = fnc.constrain_years(hist_data, hist_models)

            # Load and process the model data
            dcpp_data = p1_fnc.load_and_process_dcpp_data(
                base_dir, dcpp_models, variable, region, forecast_range, model_season
            )

            # Now we process the data to align the time periods and convert to array
            fcst1, fcst2, obs_array, common_years = p1_fnc.align_and_convert_to_array(
                hist_data, dcpp_data, hist_models, dcpp_models, obs
            )

            # Set up the
            # TODO: Set up a run which for the raw data calculates the forecast stats
            # for the longer time series (s1961-2014)
            # Would have to test to see whether this breaks the bootstrapping first though

            # If the method is 'raw', process the forecast stats
            if method == "raw":
                print("Processing forecast stats for raw method")

                # Now perform the bootstrapping to create the forecast stats
                forecast_stats = fnc.forecast_stats(
                    obs_array, fcst1, fcst2, no_boot=no_bootstraps
                )

            # Else if the method is 'lagged', lag the data
            # Before processing the forecast stats
            elif method == "lagged":
                print("Performing lagging before processing forecast stats")

                # Call the function to perform the lagging
                lag_fcst1, lag_obs, lag_fcst2 = fnc.lag_ensemble_array(
                    fcst1, fcst2, obs_array, lag=lag
                )

                # Now process the forecast stats for the lagged data
                forecast_stats = fnc.forecast_stats(
                    lag_obs, lag_fcst1, lag_fcst2, no_boot=no_bootstraps
                )

            # Else if the method is nao_matched
            elif method == "nao_matched":
                print("Performing NAO matching before processing forecast stats")

                # Set up the NAO matching base directory
                nao_match_base_dir = "/gws/nopw/j04/canari/users/benhutch/NAO-matching"

                # Load the nao_matched data
                nao_matched_data = p1_fnc.load_nao_matched_data(
                    nao_match_base_dir,
                    variable,
                    region,
                    season,
                    forecast_range,
                    start_year,
                    end_year,
                )

                # Extract the nao_matched members and mean
                nao_matched_members = nao_matched_data[0]
                # nao_matched_mean = nao_matched_data[1]

                # Use the function to constrain the NAO matched members
                aligned_data = p1_fnc.align_nao_matched_members(
                    obs, nao_matched_members, constrained_hist_data, hist_models
                )

                # Extract the aligned NAO matched members, forecast2, obs, and common years
                fcst1_nm = aligned_data[0]
                fcst2 = aligned_data[1]
                obs_array = aligned_data[2]
                common_years = aligned_data[3]

                # Now perform the bootstrapping to create the forecast stats
                forecast_stats = fnc.forecast_stats(
                    obs_array, fcst1_nm, fcst2, no_boot=no_bootstraps
                )

            else:
                raise ValueError("Method not recognised. Please try again.")

    # Check that forecast_stats exists and is a dictionary
    assert isinstance(forecast_stats, dict), "forecast_stats is not a dictionary"

    # If the full period is True, extract the forecast stats
    if full_period and method != "alternate_lag":
        forecast_stats = forecast_stats[variable]

    # Set up the save path
    save_path = (
        save_dir
        + "/"
        + variable
        + "/"
        + region
        + "/"
        + season
        + "/"
        + forecast_range
        + "/"
        + method
        + "/"
        + "no_bootstraps_"
        + str(no_bootstraps)
        + "/"
    )

    # If the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    # Save the values of the forecast stats
    np.savetxt(save_path + nens1_name, np.array([forecast_stats["nens1"]]))

    np.savetxt(save_path + nens2_name, np.array([forecast_stats["nens2"]]))

    np.savetxt(save_path + start_end_years, [common_years[0], common_years[-1]])
    # Save the forecast time series
    np.save(save_path + fcst1_ts_name, forecast_stats["f1_ts"])

    np.save(save_path + fcst2_ts_name, forecast_stats["f2_ts"])

    np.save(save_path + fcst10_ts_name, forecast_stats["f10_ts"])

    np.save(save_path + obs_ts_name, forecast_stats["o_ts"])

    # If the nao_stats_dict exists, save the nao_stats
    if nao_stats_dict is not None:
        # Set up the names for the nao_stats
        nao_stats_years_name = (
            f"nao_stats_years_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # Years short
        nao_stats_years_short_name = (
            f"nao_stats_years_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # years lag
        nao_stats_years_lag_name = (
            f"nao_stats_years_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # Years lag short
        nao_stats_years_lag_short_name = (
            f"nao_stats_years_lag_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # obs_nao_ts
        obs_nao_ts_name = (
            f"obs_nao_ts_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # obs_nao_ts_short
        obs_nao_ts_short_name = (
            f"obs_nao_ts_short_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # obs_nao_ts_lag
        obs_nao_ts_lag_name = (
            f"obs_nao_ts_lag_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # obs_nao_ts_lag_short
        obs_nao_ts_lag_short_name = (
            f"obs_nao_ts_lag_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts
        model_nao_ts_name = (
            f"model_nao_ts_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # model_nao_ts_short
        model_nao_ts_short_name = (
            f"model_nao_ts_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_members
        model_nao_ts_members_name = (
            f"model_nao_ts_members_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_members_short
        model_nao_ts_members_short_name = (
            f"model_nao_ts_members_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_short_min
        model_nao_ts_short_min_name = (
            f"model_nao_ts_short_min_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_short_max
        model_nao_ts_short_max_name = (
            f"model_nao_ts_short_max_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_min
        model_nao_ts_min_name = (
            f"model_nao_ts_min_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # model_nao_ts_max
        model_nao_ts_max_name = (
            f"model_nao_ts_max_{variable}_{region}_{season}_" + f"{forecast_range}.npy"
        )

        # model_nao_ts_var_adjust
        model_nao_ts_var_adjust_name = (
            f"model_nao_ts_var_adjust_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_var_adjust_short
        model_nao_ts_var_adjust_short_name = (
            f"model_nao_ts_var_adjust_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust
        model_nao_ts_lag_var_adjust_name = (
            f"model_nao_ts_lag_var_adjust_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust_short
        model_nao_ts_lag_var_adjust_short_name = (
            f"model_nao_ts_lag_var_adjust_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_members
        model_nao_ts_lag_members_name = (
            f"model_nao_ts_lag_members_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_members_short
        model_nao_ts_lag_members_short_name = (
            f"model_nao_ts_lag_members_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust_min
        model_nao_ts_lag_var_adjust_min_name = (
            f"model_nao_ts_lag_var_adjust_min_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust_max
        model_nao_ts_lag_var_adjust_max_name = (
            f"model_nao_ts_lag_var_adjust_max_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust_short_min
        model_nao_ts_lag_var_adjust_short_min_name = (
            f"model_nao_ts_lag_var_adjust_short_min_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # model_nao_ts_lag_var_adjust_short_max
        model_nao_ts_lag_var_adjust_short_max_name = (
            f"model_nao_ts_lag_var_adjust_short_max_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for corr1 nao stats
        nao_stats_corr1_name = (
            f"nao_stats_corr1_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for corr1 nao stats short
        nao_stats_corr1_short_name = (
            f"nao_stats_corr1_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for corr1 nao stats lag var adjust
        nao_stats_corr1_lag_var_adjust_name = (
            f"nao_stats_corr1_lag_var_adjust_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for corr1 nao stats lag var adjust short
        nao_stats_corr1_lag_var_adjust_short_name = (
            f"nao_stats_corr1_lag_var_adjust_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for p1 nao stats
        nao_stats_p1_name = (
            f"nao_stats_p1_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for p1 nao stats short
        nao_stats_p1_short_name = (
            f"nao_stats_p1_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for p1 nao stats lag var adjust
        nao_stats_p1_lag_var_adjust_name = (
            f"nao_stats_p1_lag_var_adjust_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for p1 nao stats lag var adjust short
        nao_stats_p1_lag_var_adjust_short_name = (
            f"nao_stats_p1_lag_var_adjust_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPC1 nao stats
        nao_stats_RPC1_name = (
            f"nao_stats_RPC1_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for RPC1 nao stats short
        nao_stats_RPC1_short_name = (
            f"nao_stats_RPC1_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPC1 nao stats lag
        nao_stats_RPC1_lag_name = (
            f"nao_stats_RPC1_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPC1 nao stats lag short
        nao_stats_RPC1_lag_short_name = (
            f"nao_stats_RPC1_lag_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPS1
        nao_stats_RPS1_name = (
            f"nao_stats_RPS1_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for RPS1 short
        nao_stats_RPS1_short_name = (
            f"nao_stats_RPS1_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPS1 lag
        nao_stats_RPS1_lag_name = (
            f"nao_stats_RPS1_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for RPS1 lag short
        nao_stats_RPS1_lag_short_name = (
            f"nao_stats_RPS1_lag_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats short period
        nao_stats_short_period_name = (
            f"nao_stats_short_period_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats long period
        nao_stats_long_period_name = (
            f"nao_stats_long_period_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats lag short period
        nao_stats_short_period_lag_name = (
            f"nao_stats_short_period_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats lag long period
        nao_stats_long_period_lag_name = (
            f"nao_stats_long_period_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats nens
        nao_stats_nens_name = (
            f"nao_stats_nens_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for nao_stats nens lag
        nao_stats_nens_lag_name = (
            f"nao_stats_nens_lag_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats obs_spna
        nao_stats_obs_spna_name = (
            f"nao_stats_obs_spna_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats obs_spna short
        nao_stats_obs_spna_short_name = (
            f"nao_stats_obs_spna_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna_members
        nao_stats_model_spna_members_name = (
            f"nao_stats_model_spna_members_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna_members short
        nao_stats_model_spna_members_short_name = (
            f"nao_stats_model_spna_members_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna
        nao_stats_model_spna_name = (
            f"nao_stats_model_spna_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna short
        nao_stats_model_spna_short_name = (
            f"nao_stats_model_spna_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna min
        nao_stats_model_spna_min_name = (
            f"nao_stats_model_spna_min_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna max
        nao_stats_model_spna_max_name = (
            f"nao_stats_model_spna_max_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna short min
        nao_stats_model_spna_short_min_name = (
            f"nao_stats_model_spna_short_min_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats model_spna short max
        nao_stats_model_spna_short_max_name = (
            f"nao_stats_model_spna_short_max_{variable}_{region}_{season}_"
            + f"{forecast_range}.npy"
        )

        # name for nao_stats corr1_spna .txt file
        nao_stats_corr1_spna_name = (
            f"nao_stats_corr1_spna_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats corr1_spna short .txt file
        nao_stats_corr1_spna_short_name = (
            f"nao_stats_corr1_spna_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats p1_spna .txt file
        nao_stats_p1_spna_name = (
            f"nao_stats_p1_spna_{variable}_{region}_{season}_" + f"{forecast_range}.txt"
        )

        # name for nao_stats p1_spna short .txt file
        nao_stats_p1_spna_short_name = (
            f"nao_stats_p1_spna_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats RPC1_spna .txt file
        nao_stats_RPC1_spna_name = (
            f"nao_stats_RPC1_spna_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats RPC1_spna short .txt file
        nao_stats_RPC1_spna_short_name = (
            f"nao_stats_RPC1_spna_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats corr_spna_nao_obs
        nao_stats_corr_spna_nao_obs_name = (
            f"nao_stats_corr_spnafunc_nao_obs_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats corr_spna_nao_obs short
        nao_stats_corr_spna_nao_obs_short_name = (
            f"nao_stats_corr_spna_nao_obs_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats p_spna_nao_obs
        nao_stats_p_spna_nao_obs_name = (
            f"nao_stats_p_spna_nao_obs_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats p_spna_nao_obs short
        nao_stats_p_spna_nao_obs_short_name = (
            f"nao_stats_p_spna_nao_obs_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats corr_spna_nao_model
        nao_stats_corr_spna_nao_model_name = (
            f"nao_stats_corr_spna_nao_model_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats corr_spna_nao_model short
        nao_stats_corr_spna_nao_model_short_name = (
            f"nao_stats_corr_spna_nao_model_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats p_spna_nao_model
        nao_stats_p_spna_nao_model_name = (
            f"nao_stats_p_spna_nao_model_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats p_spna_nao_model short
        nao_stats_p_spna_nao_model_short_name = (
            f"nao_stats_p_spna_nao_model_short_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # name for nao_stats tas nens
        nao_stats_tas_nens_name = (
            f"nao_stats_tas_nens_{variable}_{region}_{season}_"
            + f"{forecast_range}.txt"
        )

        # save all of the .npy files
        np.save(save_path + nao_stats_years_name, nao_stats_dict["years"])

        np.save(save_path + nao_stats_years_lag_name, nao_stats_dict["years_lag"])

        # Save the short period years
        np.save(save_path + nao_stats_years_short_name, nao_stats_dict["years_short"])

        np.save(
            save_path + nao_stats_years_lag_short_name,
            nao_stats_dict["years_lag_short"],
        )

        # Save the obs_nao_ts
        np.save(save_path + obs_nao_ts_name, nao_stats_dict["obs_nao_ts"])

        np.save(save_path + obs_nao_ts_short_name, nao_stats_dict["obs_nao_ts_short"])

        np.save(save_path + obs_nao_ts_lag_name, nao_stats_dict["obs_nao_ts_lag"])

        np.save(
            save_path + obs_nao_ts_lag_short_name,
            nao_stats_dict["obs_nao_ts_lag_short"],
        )

        # Save the model_nao_ts
        np.save(save_path + model_nao_ts_name, nao_stats_dict["model_nao_ts"])

        np.save(
            save_path + model_nao_ts_short_name, nao_stats_dict["model_nao_ts_short"]
        )

        np.save(
            save_path + model_nao_ts_members_name,
            nao_stats_dict["model_nao_ts_members"],
        )

        np.save(
            save_path + model_nao_ts_members_short_name,
            nao_stats_dict["model_nao_ts_members_short"],
        )

        np.save(
            save_path + model_nao_ts_short_min_name,
            nao_stats_dict["model_nao_ts_short_min"],
        )

        np.save(
            save_path + model_nao_ts_short_max_name,
            nao_stats_dict["model_nao_ts_short_max"],
        )

        np.save(save_path + model_nao_ts_min_name, nao_stats_dict["model_nao_ts_min"])

        np.save(save_path + model_nao_ts_max_name, nao_stats_dict["model_nao_ts_max"])

        np.save(
            save_path + model_nao_ts_var_adjust_name,
            nao_stats_dict["model_nao_ts_var_adjust"],
        )

        np.save(
            save_path + model_nao_ts_var_adjust_short_name,
            nao_stats_dict["model_nao_ts_var_adjust_short"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_short_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust_short"],
        )

        np.save(
            save_path + model_nao_ts_lag_members_name,
            nao_stats_dict["model_nao_ts_lag_members"],
        )

        np.save(
            save_path + model_nao_ts_lag_members_short_name,
            nao_stats_dict["model_nao_ts_lag_members_short"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_min_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust_min"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_max_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust_max"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_short_min_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust_short_min"],
        )

        np.save(
            save_path + model_nao_ts_lag_var_adjust_short_max_name,
            nao_stats_dict["model_nao_ts_lag_var_adjust_short_max"],
        )

        # Save the obs_spna
        np.save(save_path + nao_stats_obs_spna_name, nao_stats_dict["obs_spna"])

        np.save(
            save_path + nao_stats_obs_spna_short_name, nao_stats_dict["obs_spna_short"]
        )

        # Save the model_spna
        np.save(
            save_path + nao_stats_model_spna_members_name,
            nao_stats_dict["model_spna_members"],
        )

        np.save(
            save_path + nao_stats_model_spna_members_short_name,
            nao_stats_dict["model_spna_members_short"],
        )

        np.save(save_path + nao_stats_model_spna_name, nao_stats_dict["model_spna"])

        np.save(
            save_path + nao_stats_model_spna_short_name,
            nao_stats_dict["model_spna_short"],
        )

        np.save(
            save_path + nao_stats_model_spna_min_name, nao_stats_dict["model_spna_min"]
        )

        np.save(
            save_path + nao_stats_model_spna_max_name, nao_stats_dict["model_spna_max"]
        )

        np.save(
            save_path + nao_stats_model_spna_short_min_name,
            nao_stats_dict["model_spna_short_min"],
        )

        np.save(
            save_path + nao_stats_model_spna_short_max_name,
            nao_stats_dict["model_spna_short_max"],
        )

        # Save all of the .txt files
        # Save the corr1 nao stats
        np.savetxt(
            save_path + nao_stats_corr1_name, np.array([nao_stats_dict["corr1"]])
        )

        np.savetxt(
            save_path + nao_stats_corr1_short_name,
            np.array([nao_stats_dict["corr1_short"]]),
        )

        np.savetxt(
            save_path + nao_stats_corr1_lag_var_adjust_name,
            np.array([nao_stats_dict["corr1_lag_var_adjust"]]),
        )

        np.savetxt(
            save_path + nao_stats_corr1_lag_var_adjust_short_name,
            np.array([nao_stats_dict["corr1_lag_var_adjust_short"]]),
        )

        # Save the p1 nao stats
        np.savetxt(save_path + nao_stats_p1_name, np.array([nao_stats_dict["p1"]]))

        np.savetxt(
            save_path + nao_stats_p1_short_name, np.array([nao_stats_dict["p1_short"]])
        )

        np.savetxt(
            save_path + nao_stats_p1_lag_var_adjust_name,
            np.array([nao_stats_dict["p1_lag_var_adjust"]]),
        )

        np.savetxt(
            save_path + nao_stats_p1_lag_var_adjust_short_name,
            np.array([nao_stats_dict["p1_lag_var_adjust_short"]]),
        )

        # Save the RPC1 nao stats
        np.savetxt(save_path + nao_stats_RPC1_name, np.array([nao_stats_dict["RPC1"]]))

        np.savetxt(
            save_path + nao_stats_RPC1_short_name,
            np.array([nao_stats_dict["RPC1_short"]]),
        )

        np.savetxt(
            save_path + nao_stats_RPC1_lag_name, np.array([nao_stats_dict["RPC1_lag"]])
        )

        np.savetxt(
            save_path + nao_stats_RPC1_lag_short_name,
            np.array([nao_stats_dict["RPC1_lag_short"]]),
        )

        # Save the RPS1 nao stats
        np.savetxt(save_path + nao_stats_RPS1_name, np.array([nao_stats_dict["RPS1"]]))

        np.savetxt(
            save_path + nao_stats_RPS1_short_name,
            np.array([nao_stats_dict["RPS1_short"]]),
        )

        np.savetxt(
            save_path + nao_stats_RPS1_lag_name, np.array([nao_stats_dict["RPS1_lag"]])
        )

        np.savetxt(
            save_path + nao_stats_RPS1_lag_short_name,
            np.array([nao_stats_dict["RPS1_lag_short"]]),
        )

        # Save the short period nao stats
        np.savetxt(
            save_path + nao_stats_short_period_name,
            np.array([nao_stats_dict["short_period"]]),
        )

        np.savetxt(
            save_path + nao_stats_long_period_name,
            np.array([nao_stats_dict["long_period"]]),
        )

        np.savetxt(
            save_path + nao_stats_short_period_lag_name,
            np.array([nao_stats_dict["short_period_lag"]]),
        )

        np.savetxt(
            save_path + nao_stats_long_period_lag_name,
            np.array([nao_stats_dict["long_period_lag"]]),
        )

        # Save the nens nao stats
        np.savetxt(save_path + nao_stats_nens_name, np.array([nao_stats_dict["nens"]]))

        np.savetxt(
            save_path + nao_stats_nens_lag_name, np.array([nao_stats_dict["nens_lag"]])
        )

        # Save the corr1_spna nao stats
        np.savetxt(
            save_path + nao_stats_corr1_spna_name,
            np.array([nao_stats_dict["corr1_spna"]]),
        )

        np.savetxt(
            save_path + nao_stats_corr1_spna_short_name,
            np.array([nao_stats_dict["corr1_spna_short"]]),
        )

        # Save the p1_spna nao stats
        np.savetxt(
            save_path + nao_stats_p1_spna_name, np.array([nao_stats_dict["p1_spna"]])
        )

        np.savetxt(
            save_path + nao_stats_p1_spna_short_name,
            np.array([nao_stats_dict["p1_spna_short"]]),
        )

        # Save the RPC1_spna nao stats
        np.savetxt(
            save_path + nao_stats_RPC1_spna_name,
            np.array([nao_stats_dict["RPC1_spna"]]),
        )

        np.savetxt(
            save_path + nao_stats_RPC1_spna_short_name,
            np.array([nao_stats_dict["RPC1_spna_short"]]),
        )

        # Save the corr_spna_nao_obs nao stats
        np.savetxt(
            save_path + nao_stats_corr_spna_nao_obs_name,
            np.array([nao_stats_dict["corr_spna_nao_obs"]]),
        )

        np.savetxt(
            save_path + nao_stats_corr_spna_nao_obs_short_name,
            np.array([nao_stats_dict["corr_spna_nao_short_obs"]]),
        )

        # Save the p_spna_nao_obs nao stats
        np.savetxt(
            save_path + nao_stats_p_spna_nao_obs_name,
            np.array([nao_stats_dict["p_spna_nao_obs"]]),
        )

        np.savetxt(
            save_path + nao_stats_p_spna_nao_obs_short_name,
            np.array([nao_stats_dict["p_spna_nao_short_obs"]]),
        )

        # Save the corr_spna_nao_model nao stats
        np.savetxt(
            save_path + nao_stats_corr_spna_nao_model_name,
            np.array([nao_stats_dict["corr_spna_nao_model"]]),
        )

        np.savetxt(
            save_path + nao_stats_corr_spna_nao_model_short_name,
            np.array([nao_stats_dict["corr_spna_nao_short_model"]]),
        )

        # Save the p_spna_nao_model nao stats
        np.savetxt(
            save_path + nao_stats_p_spna_nao_model_name,
            np.array([nao_stats_dict["p_spna_nao_model"]]),
        )

        np.savetxt(
            save_path + nao_stats_p_spna_nao_model_short_name,
            np.array([nao_stats_dict["p_spna_nao_short_model"]]),
        )

        # Save the tas nens nao stats
        np.savetxt(
            save_path + nao_stats_tas_nens_name, np.array([nao_stats_dict["tas_nens"]])
        )


if __name__ == "__main__":
    main()
