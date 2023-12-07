# Functions for exploring the NAO skill
# Imports
import dictionaries as dic
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
from _datetime import datetime
import scipy.stats as stats
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image
from sklearn.utils import resample
from scipy.stats import pearsonr
from xarray import DataArray

import matplotlib.cm as mpl_cm
import matplotlib
import cartopy.crs as ccrs
import iris
import iris.coord_categorisation as coord_cat
import iris.plot as iplt
import scipy
import pdb
import iris.quickplot as qplt
from typing import Dict, List, Union

# Local imports
# Functions
from functions import calculate_obs_nao

# Import the dictionaries
sys.path.append('/home/users/benhutch/skill-maps')


# Define a function for the NAO stats
def nao_stats(obs_psl: DataArray,
              hindcast_psl: Dict[str, List[DataArray]],
              models_list: List[str],
              obs_tas: DataArray = None,
              hindcast_tas: Dict[str, List[DataArray]] = None,
              lag: int = 3,
              short_period: tuple = (1965, 2010),
              season: str = 'DJFM') -> Dict[str, Dict]:
    """
    Assess and compare the skill of the NAO index between different models
    and observations during the winter season (DJFM). The skill is assessed
    using the correlation, p-value and RPC.

    Based on Doug Smith's 'fcsts_assess' function.

    Inputs:
    -------

    obs_psl[time, lat, lon]: DataArray
        Observations of the psl anomaly fields.

    hindcast_psl: Dict[str, List[DataArray]]
        A dictionary containing the psl hindcasts for each model. The keys are
        the model names and the values are a list of DataArrays containing
        the hindcast data.

    models_list: List[str]
        A list of the model names

    obs_tas[time, lat, lon]: DataArray
        Observations of the tas anomaly fields. Default is None.

    hindcast_tas: Dict[str, List[DataArray]]
        A dictionary containing the tas hindcasts for each model. The keys are
        the model names and the values are a list of DataArrays containing
        the hindcast data. Default is None.

    lag: int
            The lag to use when assessing the skill of the NAO index.
            Default is 3.

    short_period: tuple
        A tuple containing the start and end years of the short period to
        assess the skill of the NAO index. Default is (1965, 2010).

    season: str
        The season to use when assessing the skill of the NAO index.
        Default is 'DJFM'.

    Outputs:
    --------

    nao_stats_dict: dict[dict]

        A dictionary containing the NAO stats for each model. The keys are
        the model names and the values are a dictionary containing the NAO
        stats for that model.

        The keys of the nested dictionary are:

        years: list[int]
            The years of the hindcast e.g. [1966, 1982, ..., 2019]

        years_lag: list[int]
            The years of the lagged hindcast e.g. [1969, 1981, ..., 2019]

        obs_nao_ts: array[time]
            The NAO index for the observations.

        model_nao_ts: array[time]
            The ensemble mean NAO index for the model.

        model_nao_ts_members: array[member, time]
            The NAO index for each ensemble member of the model.

        model_nao_ts_min, model_nao_ts_max: array[time]
            The 5% and 95% percentiles of the model NAO index.

        model_nao_ts_var_adjust: array[time]
            The NAO index for the model with the variance adjusted.

        model_nao_ts_lag_var_adjust: array[time - lag]
            The lagged NAO index for the model with the variance adjusted.

        model_nao_ts_lag_var_adjust_min, 
            model_nao_ts_lag_var_adjust_max: array[time - lag]
            The 5% and 95% percentiles of the lagged model NAO index with the
            variance adjusted.

        corr1: float
            The correlation between the model NAO index and the observed NAO
            index for the full hindcast period.

        corr1_short: float
            The correlation between the model NAO index and the observed NAO
            index for the short hindcast period.

        corr1_lag: float
            The correlation between the lagged model NAO index and the observed
            NAO index for the full hindcast period.

        corr1_lag_short: float
            The correlation between the lagged model NAO index and the observed
            NAO index for the short hindcast period.

        p1: float
            The p-value for the correlation between the model NAO index and the
            observed NAO index for the full hindcast period.

        p1_short: float
            The p-value for the correlation between the model NAO index and the
            observed NAO index for the short hindcast period.

        p1_lag: float
            The p-value for the correlation between the lagged model NAO index
            and the observed NAO index for the full hindcast period.

        p1_lag_short: float
            The p-value for the correlation between the lagged model NAO index
            and the observed NAO index for the short hindcast period.

        RPC1: float
            The RPC between the model NAO index and the observed NAO index for
            the full hindcast period.

        RPC1_short: float
            The RPC between the model NAO index and the observed NAO index for
            the short hindcast period.

        RPC1_lag: float
            The RPC between the lagged model NAO index and the observed NAO
            index for the full hindcast period.

        RPC1_lag_short: float
            The RPC between the lagged model NAO index and the observed NAO
            index for the short hindcast period.

        short_period: tuple
            The start and end years of the short period e.g. (1965, 2010)

        long_period: tuple
            The start and end years of the long period
            which is the full hindcast period e.g. (1966, 2019)

        short_period_lag: tuple
            The start and end years of the short period with the lag applied
            e.g. (1969, 2010)

        long_period_lag: tuple
            The start and end years of the long (full) period with the 
            lag applied e.g. (1969, 2019)

        nens: int
            The number of ensemble members in the raw hindcast for that model.

        nens_lag: int
            The number of ensemble members in the lagged hindcast for that
            model (i.e. nens * lag)

    """

    # Assert that the season is DJFM
    assert season == 'DJFM', "The season must be DJFM"

    # Assert that either both obs_tas and hindcast_tas are None or both are not None
    assert (obs_tas is None and hindcast_tas is None) or (obs_tas is not None and hindcast_tas is not None), \
        "Either both obs_tas and hindcast_tas must be None or both must be not None"

    # Hard code in the dictionaries containing
    # the grid boxes for the NAO index
    azores_grid = dic.azores_grid_corrected
    iceland_grid = dic.iceland_grid_corrected

    # Create a dictionary to store the NAO stats for each model
    nao_stats_dict = {}

    # Set up the missing data indicator
    mdi = -9999.0

    # Loop over the models
    for i, model in enumerate(models_list):
        print("Setting up the NAO stats for the {} model".format(model))
        print("model number: {}".format(i))

        # Create a dictionary for the NAO stats for this model
        nao_stats_dict[model] = {

            'years': [], 'years_lag': [], 'years_lag_short': [], 'years_short': [],

            'obs_nao_ts': [], 'obs_nao_ts_short': [],

            'obs_nao_ts_lag': [],

            'obs_nao_ts_lag_short': [], 'model_nao_ts': [], 'model_nao_ts_short': [],

            'model_nao_ts_members': [], 'model_nao_ts_members_short': [],

            'model_nao_ts_short_min': [],

            'model_nao_ts_short_max': [],

            'model_nao_ts_min': [], 'model_nao_ts_max': [],

            'model_nao_ts_var_adjust': [], 'model_nao_ts_var_adjust_short': [],

            'model_nao_ts_lag_var_adjust': [], 'model_nao_ts_lag_var_adjust_short': [],

            'model_nao_ts_lag_members': [], 'model_nao_ts_lag_members_short': [],

            'model_nao_ts_lag_var_adjust_min': [],

            'model_nao_ts_lag_var_adjust_max': [],

            'model_nao_ts_lag_var_adjust_min_short': [],

            'model_nao_ts_lag_var_adjust_max_short': [],

            'corr1': mdi,

            'corr1_short': mdi, 'corr1_lag_var_adjust': mdi, 'corr1_lag_var_adjust_short': mdi,

            'p1': mdi, 'p1_short': mdi, 'p1_lag_var_adjust': mdi, 'p1_lag_var_adjust_short': mdi,

            'RPC1': mdi, 'RPC1_short': mdi, 'RPC1_lag': mdi,

            'RPC1_lag_short': mdi, 'RPS1': mdi, 'RPS1_short': mdi,

            'RPS1_lag': mdi, 'RPS1_lag_short': mdi,

            'short_period': short_period,

            'long_period': (), 'short_period_lag': (), 'long_period_lag': (),

            'nens': mdi, 'nens_lag': mdi,

            'obs_spna': [], 'obs_spna_short': [],

            'model_spna_members': [], 'model_spna_members_short': [],

            'model_spna': [], 'model_spna_short': [],

            'model_spna_min': [], 'model_spna_max': [],

            'model_spna_short_min': [], 'model_spna_short_max': [],

            'corr1_spna': mdi, 'corr1_spna_short': mdi,

            'p1_spna': mdi, 'p1_spna_short': mdi,

            'RPC1_spna': mdi, 'RPC1_spna_short': mdi,

            'corr_spna_nao_obs': mdi, 'corr_spna_nao_short_obs': mdi,

            'p_spna_nao_obs': mdi, 'p_spna_nao_short_obs': mdi,

            'corr_spna_nao_model': mdi, 'corr_spna_nao_short_model': mdi,

            'p_spna_nao_model': mdi, 'p_spna_nao_short_model': mdi,

            'tas_nens': mdi

        }

        # Extract the list of hindcast DataArrays for this model
        hindcast_list = hindcast_psl[model]

        # Ensure that each of the data arrays has the same time axis
        # Extract the years for the first member
        years1 = hindcast_list[0].time.dt.year.values

        # Limit years to those below 2020
        years1 = years1[years1 < 2020]

        # Assert that this doesn't have any duplicate values
        assert len(years1) == len(set(years1)), \
            "The years in the hindcast data for the {} model are not unique".format(
                model)

        # If there is a gap of more than one year between the years then raise a value error
        if np.any(np.diff(years1) > 1):
            print("There is a gap of more than one year in the hindcast data for the {} model".format(
                model))
            # Find the indices of the gaps
            gap_indices = np.where(np.diff(years1) > 1)[0]

            # print the years of the gaps
            print(
                f"The years of the gaps are: {years1[gap_indices-1]} and {years1[gap_indices]} and {years1[gap_indices+1]}")

        # Assert that there are no gaps of more than one year between the years
        assert np.all(np.diff(years1) <= 1), \
            "There is a gap of more than one year in the hindcast data for the {} model".format(
                model)

        # Assert that there are at least 10 years in the hindcast data
        assert len(years1) >= 10, \
            "There are less than 10 years in the hindcast data for the {} model".format(
                model)

        # If hindcast_tas is not None then assert that the length of hindcast_psl and hindcast_tas are the same
        if hindcast_tas is not None:
            print(
                "checking that the length of hindcast_psl and hindcast_tas are the same")

            # Extract the list of hindcast DataArrays for this model
            hindcast_list_tas = hindcast_tas[model]

            # Extract the years for the first member
            years1_tas = hindcast_list_tas[0].time.dt.year.values

            # Limit years to those below 2020
            years1_tas = years1_tas[years1_tas < 2020]

            # Assert that this doesn't have any duplicate values
            assert len(years1_tas) == len(set(years1_tas)), \
                "The years in the hindcast data for the {} model are not unique".format(
                    model)

            # If there is a gap of more than one year between the years then raise a value error
            if np.any(np.diff(years1_tas) > 1):
                print("There is a gap of more than one year in the hindcast data for the {} model".format(
                    model))
                # Find the indices of the gaps
                gap_indices = np.where(np.diff(years1_tas) > 1)[0]

                # print the years of the gaps
                print(
                    f"The years of the gaps are: {years1_tas[gap_indices-1]} and {years1_tas[gap_indices]} and {years1_tas[gap_indices+1]}")

            # Assert that there are no gaps of more than one year between the years
            assert np.all(np.diff(years1_tas) <= 1), \
                "There is a gap of more than one year in the hindcast data for the {} model".format(
                    model)

            # Assert that there are at least 10 years in the hindcast data
            assert len(years1_tas) >= 10, \
                "There are less than 10 years in the hindcast data for the {} model".format(
                    model)

            # if the model is BCC-CSM2-MR
            # then check years1_tas has years 1966 to 2019
            if model == 'BCC-CSM2-MR':
                print("checking that the years for BCC-CSM2-MR are 1966 to 2019")
                assert np.array_equal(years1_tas, np.arange(1966, 2020)), \
                    "The years for BCC-CSM2-MR are not 1966 to 2019"

            else:
                # Check that years 1965 to 2019 are in years1_tas
                print("checking that the years for {} are 1965 to 2019".format(model))
                assert np.array_equal(years1_tas, np.arange(1965, 2020)), \
                    "The years for {} are not 1965 to 2019".format(model)

        # Extract the list of tas hindcast DataArrays for this model
        if hindcast_tas is not None:
            hindcast_list_tas = hindcast_tas[model]

        # Loop over the remaining members
        for member in hindcast_list[1:]:

            # Extract the years for this member
            years2 = member.time.dt.year.values

            # Limit years to those below 2020
            years2 = years2[years2 < 2020]

            # Assert that this doesn't have any duplicate values
            assert len(years2) == len(set(years2)), \
                "The years in the hindcast data for the {} model are not unique".format(
                    model)

            # Assert that there are no gaps of more than one year between the years
            assert np.all(np.diff(years2) <= 1), \
                "There is a gap of more than one year in the hindcast data for the {} model".format(
                    model)

            # Assert that there are at least 10 years in the hindcast data
            assert len(years2) >= 10, \
                "There are less than 10 years in the hindcast data for the {} model".format(
                    model)

            # If years1 and years2 are not the same then raise a value error
            assert np.all(years1 == years2), \
                "The years in the hindcast data for the {} model are not the same".format(
                    model)

        # If hindcast_tas is not None then assert that the length of hindcast_psl and hindcast_tas are the same
        if hindcast_tas is not None:
            # Loop over the remaining members
            for member_tas in hindcast_list_tas[1:]:

                # print that we are checking the rest of the members
                print(
                    "checking that the length of hindcast_psl and hindcast_tas are the same for the rest of the members")

                # Extract the years for this member
                years2_tas = member_tas.time.dt.year.values

                # Limit years to those below 2020
                years2_tas = years2_tas[years2_tas < 2020]

                # Assert that this doesn't have any duplicate values
                assert len(years2_tas) == len(set(years2_tas)), \
                    "The years in the hindcast data for the {} model are not unique".format(
                        model)

                # Assert that there are no gaps of more than one year between the years
                assert np.all(np.diff(years2_tas) <= 1), \
                    "There is a gap of more than one year in the hindcast data for the {} model".format(
                        model)

                # Assert that there are at least 10 years in the hindcast data
                assert len(years2_tas) >= 10, \
                    "There are less than 10 years in the hindcast data for the {} model".format(
                        model)

                # if the model is BCC-CSM2-MR
                # then check years2_tas has years 1966 to 2019
                if model == 'BCC-CSM2-MR':
                    print("checking that the years for BCC-CSM2-MR are 1966 to 2019")
                    assert np.array_equal(years2_tas, np.arange(1966, 2020)), \
                        "The years for BCC-CSM2-MR are not 1966 to 2019"

                else:
                    # Check that years 1965 to 2019 are in years2_tas
                    print(
                        "checking that the years for {} are 1965 to 2019".format(model))
                    assert np.array_equal(years2_tas, np.arange(1965, 2020)), \
                        "The years for {} are not 1965 to 2019".format(model)

        # print("years checking complete for the {} model".format(model))
        # continue

        # Ensure that the observations and the hindcast have the same time axis
        # Extract the years for the observations
        years_obs = obs_psl.time.dt.year.values

        # Assert that this doesn't have any duplicate values
        assert len(years_obs) == len(set(years_obs)), \
            "The years in the observations are not unique"

        # Assert that there are no gaps of more than one year between the years
        assert np.all(np.diff(years_obs) <= 1), \
            "There is a gap of more than one year in the observations"

        # Assert that there are at least 10 years in the observations
        assert len(years_obs) >= 10, \
            "There are less than 10 years in the observations"

        # Assert that there are no NaNs in the observations
        assert np.all(np.isnan(obs_psl) == False), \
            "There are NaNs in the observations"

        # Create a temporary copy of the observations
        obs_tmp = obs_psl.copy()

        # Create a temporary copy of the observations
        obs_tmp_tas = obs_tas.copy()

        # if hindcast_tas is not None then assert that the length of hindcast_psl and hindcast_tas are the same
        if hindcast_tas is not None:
            print("comparing tas data to psl data and obs data")
            # print years1_tas and years1
            print("years1_tas first year: {}".format(years1_tas[0]))
            print("years1 first year: {}".format(years1[0]))
            print("years1_tas last year: {}".format(years1_tas[-1]))
            print("years1 last year: {}".format(years1[-1]))

            # assert that years1_tas and years1 are the same
            assert np.array_equal(years1_tas, years1), \
                "The years in the hindcast data for the {} model are not the same as the years in the hindcast tas data".format(
                    model)

        # If the values in years1 and years_obs are not the same then raise a value error
        if np.array_equal(years1, years_obs) == False:
            print("The years in the observations are not the same as the years in the hindcast data for the {} model".format(model))
            print("Obs first year: {}".format(years_obs[0]))
            print("Hindcast first year: {}".format(years1[0]))
            print("Obs last year: {}".format(years_obs[-1]))
            print("Hindcast last year: {}".format(years1[-1]))

            # assert that the obs is longer than the hindcast
            assert len(years_obs) > len(years1), \
                "The observations are shorter than the hindcast data for the {} model".format(
                    model)

            # Extract only the hindcast years from the observations
            obs_tmp = obs_tmp.sel(time=obs_tmp.time.dt.year.isin(years1))

        # if hindcast_tas is not None
        if hindcast_tas is not None:
            if np.array_equal(years1, obs_tmp_tas.time.dt.year.values) == False:
                print("The years in the observations are not the same as the years in the hindcast data for the {} model".format(model))
                print("Obs first year: {}".format(years_obs[0]))
                print("Hindcast first year: {}".format(years1[0]))
                print("Obs last year: {}".format(years_obs[-1]))
                print("Hindcast last year: {}".format(years1[-1]))

                # assert that the obs is longer than the hindcast
                assert len(obs_tmp_tas.time.dt.year.values) > len(years1), \
                    "The observations are shorter than the hindcast data for the {} model".format(
                        model)

                # Extract only the hindcast years from the observations
                obs_tmp_tas = obs_tmp_tas.sel(
                    time=obs_tmp_tas.time.dt.year.isin(years1))

        # Assert that year 1 of the observations is the same as year 1 of the hindcast
        assert obs_tmp.time.dt.year.values[0] == years1[0], \
            "The first year of the observations is not the same as the first year of the hindcast data for the {} model".format(
                model)

        # Assert that year -1 of the observations is the same as year -1 of the hindcast
        assert obs_tmp.time.dt.year.values[-1] == years1[-1], \
            "The last year of the observations is not the same as the last year of the hindcast data for the {} model".format(
                model)

        # Assert that the length of the observations is the same as the length of the hindcast
        assert len(obs_tmp.time.dt.year.values) == len(years1), \
            "The length of the observations is not the same as the length of the hindcast data for the {} model".format(
                model)

        print("years checking complete for the observations and the {} model".format(model))

        # Append the years to the dictionary
        nao_stats_dict[model]['years'] = years1

        # Create the years for the short period
        years_short = np.arange(years1[0], short_period[1] + 1)

        # Append the years for the short period to the dictionary
        nao_stats_dict[model]['years_short'] = years_short

        # Append the short period to the dictionary
        nao_stats_dict[model]['short_period'] = short_period

        # Append the long period to the dictionary
        nao_stats_dict[model]['long_period'] = (years1[0], years1[-1])

        # Append the short period with the lag applied to the dictionary
        nao_stats_dict[model]['short_period_lag'] = (
            years1[0] + lag - 1, short_period[1])

        # Append the long period with the lag applied to the dictionary
        nao_stats_dict[model]['long_period_lag'] = (
            years1[0] + lag - 1, years1[-1])

        # Create the years lag for the dictionary
        years_lag = np.arange(years1[0] + lag - 1, years1[-1] + 1)

        # Append the years lag to the dictionary
        nao_stats_dict[model]['years_lag'] = years_lag

        # Create the years lag for the dictionary
        years_lag_short = np.arange(years1[0] + lag - 1, short_period[1] + 1)

        # Append the years lag to the dictionary
        nao_stats_dict[model]['years_lag_short'] = years_lag_short

        # Append the nens to the dictionary
        nao_stats_dict[model]['nens'] = len(hindcast_list)

        # Calculate the observed NAO index
        obs_nao = calculate_obs_nao(obs_anomaly=obs_tmp,
                                    south_grid=azores_grid,
                                    north_grid=iceland_grid)

        # Form an array of years for the short period
        years_short = np.arange(years1[0], short_period[1] + 1)
        print("years_short: {}".format(years_short))

        # Constrain to the short period
        obs_nao_short = obs_nao.sel(
            time=obs_nao.time.dt.year.isin(years_short))

        # Convert the observed NAO index to a numpy array
        obs_nao = obs_nao.values

        # Convert the observed NAO index for the short period to a numpy array
        obs_nao_short = obs_nao_short.values

        # Append the observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts'] = obs_nao

        # Append the observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts_short'] = obs_nao_short

        # Create an empty array to store the NAO index for each member
        nao_members = np.zeros((len(hindcast_list), len(years1)))

        # Create an empty array to store the NAO index for each member
        # for the short period
        nao_members_short = np.zeros((len(hindcast_list), len(years_short)))

        # Set up the number of ensemble members
        nens = len(hindcast_list)

        # Set up the number of lagged members
        nens_lag = len(hindcast_list) * lag

        # Append the nens_lag to the dictionary
        nao_stats_dict[model]['nens_lag'] = nens_lag

        # Set up the lagged ensemble
        nao_members_lag = np.zeros([nens_lag, len(years1)])

        # Create the NAO members lag for the short period
        nao_members_lag_short = np.zeros([nens_lag, len(years_short)])

        # If hindcast_tas is not None
        if hindcast_tas is not None:

            # Append the nens to the dictionary
            nao_stats_dict[model]['tas_nens'] = len(hindcast_list_tas)

            # TODO: function for calculating SPNA index
            # Calculate the observed SPNA index
            obs_spna = calculate_spna_index(t_anom=obs_tmp_tas)

            # Constrain to the short period
            obs_spna_short = obs_spna.sel(
                time=obs_spna.time.dt.year.isin(years_short))

            # Convert the observed SPNA index to a numpy array
            obs_spna = obs_spna.values
            obs_spna_short = obs_spna_short.values

            # Append the observed SPNA index to the dictionary
            nao_stats_dict[model]['obs_spna'] = obs_spna

            # Append the observed SPNA index to the dictionary
            nao_stats_dict[model]['obs_spna_short'] = obs_spna_short

            # Create an empty array to store the SPNA index for each member
            spna_members = np.zeros((len(hindcast_list_tas), len(years1)))

            # Create an empty array to store the SPNA index for each member
            # for the short period
            spna_members_short = np.zeros(
                (len(hindcast_list_tas), len(years_short)))

        # Loop over the hindcast members to calculate the NAO index
        for i, member in enumerate(hindcast_list):

            # Calculate the NAO index for this member
            nao_member = calculate_obs_nao(obs_anomaly=member,
                                           south_grid=azores_grid,
                                           north_grid=iceland_grid)

            # Constrain to the long period
            nao_member = nao_member.sel(
                time=nao_member.time.dt.year.isin(years1))

            # Constrain to the short period
            nao_member_short = nao_member.sel(
                time=nao_member.time.dt.year.isin(years_short))

            # Loop over the years
            for year in range(len(years1)):
                # if the year index is less than the lag index
                if year < lag - 1:
                    # Set the lagged ensemble member to NaN
                    nao_members_lag[i, year] = np.nan

                    # Also set the lagged ensemble member to NaN
                    for lag_index in range(lag):
                        nao_members_lag[i + (lag_index * nens), year] = np.nan

                # Otherwise
                else:
                    # Loop over the lag indices
                    for lag_index in range(lag):
                        # Calculate the lagged ensemble member
                        nao_members_lag[i + (lag_index * nens),
                                        year] = nao_member[year - lag_index]

            # For the short period
            for year in range(len(years_short)):
                # If the year index is less than the lag index
                if year < lag - 1:
                    # Set the lagged ensemble member to NaN
                    nao_members_lag_short[i, year] = np.nan

                    # Also set the lagged ensemble member to NaN
                    for lag_index in range(lag):
                        nao_members_lag_short[i +
                                              (lag_index * nens), year] = np.nan

                # Otherwise
                else:
                    # Loop over the lag indices
                    for lag_index in range(lag):
                        # Calculate the lagged ensemble member
                        nao_members_lag_short[i + (lag_index * nens),
                                              year] = nao_member_short[year - lag_index]

            # # Now remove the first lag - 1 years from the NAO index
            # nao_members_lag[i, lag - 1:] = np.nan
            # Logging the shapes of the arrays
            print("nao_members shape: {}".format(nao_members.shape))
            print("nao_members_short shape: {}".format(
                nao_members_short.shape))
            print("nao_member shape: {}".format(nao_member.shape))
            print("nao_member_short shape: {}".format(nao_member_short.shape))

            # Append the NAO index to the members array
            nao_members[i, :] = nao_member

            # Append the NAO index to the members array
            nao_members_short[i, :] = nao_member_short

        # If hindcast_tas is not None
        if hindcast_tas is not None:
            print("Looping over the hindcast tas members to calculate the SPNA index")
            # Loop over the hindcast members to calculate the SPNA index
            for i, member in enumerate(hindcast_list_tas):

                # Calculate the SPNA index for this member
                spna_member = calculate_spna_index(t_anom=member)

                # Constrain to the long period
                spna_member = spna_member.sel(
                    time=spna_member.time.dt.year.isin(years1))

                # Constrain to the short period
                spna_member_short = spna_member.sel(
                    time=spna_member.time.dt.year.isin(years_short))

                # Append the SPNA index to the members array
                spna_members[i, :] = spna_member

                # Append the SPNA index to the members array
                spna_members_short[i, :] = spna_member_short

        # Remove the first lag - 1 years from the NAO index
        nao_members_lag = nao_members_lag[:, lag - 1:]

        # Form the lagged obs
        obs_nao_lag = obs_nao[lag - 1:]

        # Form the lagged obs for the short period
        obs_nao_lag_short = obs_nao_short[lag - 1:]

        # Remove the first lag - 1 years from the NAO index
        nao_members_lag_short = nao_members_lag_short[:, lag - 1:]

        # Calculate the ensemble mean NAO index
        nao_mean = np.mean(nao_members, axis=0)

        # Calculate the ensemble mean NAO index for the short period
        nao_mean_short = np.mean(nao_members_short, axis=0)

        # if hindcast_tas is not None
        if hindcast_tas is not None:
            print("Calculating the ensemble mean SPNA index")
            # Calculate the ensemble mean SPNA index
            spna_mean = np.mean(spna_members, axis=0)

            # Calculate the ensemble mean SPNA index for the short period
            spna_mean_short = np.mean(spna_members_short, axis=0)

            # Calculate the 5% and 95% min and max of the SPNA index
            spna_min = np.percentile(spna_members, 5, axis=0)
            spna_max = np.percentile(spna_members, 95, axis=0)

            # Calculate the 5% and 95% min and max of the SPNA index short
            spna_short_min = np.percentile(spna_members_short, 5, axis=0)
            spna_short_max = np.percentile(spna_members_short, 95, axis=0)

            # Print the shapes of the SPNA members
            print("years1.shape: {}".format(years1.shape))
            print("spna_members.shape: {}".format(spna_members.shape))
            print("spna_members_short.shape: {}".format(
                spna_members_short.shape))
            print("spna_mean.shape: {}".format(spna_mean.shape))
            print("spna_mean_short.shape: {}".format(spna_mean_short.shape))
            print("spna_min.shape: {}".format(spna_min.shape))
            print("spna_max.shape: {}".format(spna_max.shape))
            print("spna_short_min.shape: {}".format(spna_short_min.shape))
            print("spna_short_max.shape: {}".format(spna_short_max.shape))
            print("obs_spna.shape: {}".format(obs_spna.shape))
            print("obs_spna_short.shape: {}".format(obs_spna_short.shape))

            # Calculate the correlation between the SPNA index and the observed SPNA index
            corr1_spna, p1_spna = pearsonr(spna_mean, obs_spna)

            # Calculate the correlation between the SPNA index and the observed SPNA index
            corr1_spna_short, p1_spna_short = pearsonr(
                spna_mean_short, obs_spna_short)

            # Calculate the RPC between the SPNA index and the observed SPNA index
            rpc1_spna = corr1_spna / (np.std(spna_mean) / np.std(spna_members))

            # Calculate the RPC between the SPNA index and the observed SPNA index
            rpc1_spna_short = corr1_spna_short / \
                (np.std(spna_mean_short) / np.std(spna_members_short))

            # Calculate the correlation between the SPNA index and the NAO index
            # for the observations
            corr_spna_nao_obs, p_spna_nao_obs = pearsonr(obs_spna, obs_nao)

            # Calculate the correlation between the SPNA index and the NAO index
            # for the observations
            corr_spna_nao_short_obs, p_spna_nao_short_obs = pearsonr(
                obs_spna_short, obs_nao_short)

            # Calculate the correlation between the SPNA index and the NAO index
            # for the model
            corr_spna_nao_model, p_spna_nao_model = pearsonr(
                spna_mean, nao_mean)

            # Calculate the correlation between the SPNA index and the NAO index
            # for the model
            corr_spna_nao_short_model, p_spna_nao_short_model = pearsonr(
                spna_mean_short, nao_mean_short)

            # Append the SPNA members to the dictionary
            nao_stats_dict[model]['model_spna_members'] = spna_members

            # Append the SPNA members to the dictionary
            nao_stats_dict[model]['model_spna_members_short'] = spna_members_short

            # Append the SPNA index to the dictionary
            nao_stats_dict[model]['model_spna'] = spna_mean

            # Append the SPNA index to the dictionary
            nao_stats_dict[model]['model_spna_short'] = spna_mean_short

            # Append the SPNA index to the dictionaryn min and max
            nao_stats_dict[model]['model_spna_min'] = spna_min

            # Append the SPNA index to the dictionaryn min and max
            nao_stats_dict[model]['model_spna_max'] = spna_max

            # Append the SPNA index to the dictionaryn min and max short
            nao_stats_dict[model]['model_spna_short_min'] = spna_short_min

            # Append the SPNA index to the dictionaryn min and max short
            nao_stats_dict[model]['model_spna_short_max'] = spna_short_max

            # Append the correlation between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['corr1_spna'] = corr1_spna

            # Append the correlation between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['corr1_spna_short'] = corr1_spna_short

            # Append the correlation between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['p1_spna'] = p1_spna

            # Append the correlation between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['p1_spna_short'] = p1_spna_short

            # Append the RPC between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['RPC1_spna'] = rpc1_spna

            # Append the RPC between the SPNA index and the observed SPNA index
            nao_stats_dict[model]['RPC1_spna_short'] = rpc1_spna_short

            # Append the correlation between the SPNA index and the NAO index
            # for the observations
            nao_stats_dict[model]['corr_spna_nao_obs'] = corr_spna_nao_obs

            # Append the correlation between the SPNA index and the NAO index
            # for the observations
            nao_stats_dict[model]['corr_spna_nao_short_obs'] = corr_spna_nao_short_obs

            # Append the correlation between the SPNA index and the NAO index
            # for the model
            nao_stats_dict[model]['corr_spna_nao_model'] = corr_spna_nao_model

            # Append the correlation between the SPNA index and the NAO index
            # for the model
            nao_stats_dict[model]['corr_spna_nao_short_model'] = corr_spna_nao_short_model

            # Append the correlation between the SPNA index and the NAO index
            # for the observations
            nao_stats_dict[model]['p_spna_nao_obs'] = p_spna_nao_obs

            # Append the correlation between the SPNA index and the NAO index
            # for the observations
            nao_stats_dict[model]['p_spna_nao_short_obs'] = p_spna_nao_short_obs

            # Append the correlation between the SPNA index and the NAO index
            # for the model
            nao_stats_dict[model]['p_spna_nao_model'] = p_spna_nao_model

            # Append the correlation between the SPNA index and the NAO index
            # for the model
            nao_stats_dict[model]['p_spna_nao_short_model'] = p_spna_nao_short_model
        
        # Calculate the ensemble mean NAO index for the lagged ensemble
        nao_mean_lag = np.mean(nao_members_lag, axis=0)

        # Calculate the ensemble mean NAO index for the lagged ensemble
        nao_mean_lag_short = np.mean(nao_members_lag_short, axis=0)

        # Add the observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts'] = obs_nao

        # Add the observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts_short'] = obs_nao_short

        # Add the lagged observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts_lag'] = obs_nao_lag

        # Add the lagged observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts_lag_short'] = obs_nao_lag_short

        # Append the ensemble mean NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts'] = nao_mean

        # Append the ensemble mean NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_short'] = nao_mean_short

        # Append the ensemble members NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_members'] = nao_members

        # Append the ensemble members NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_members_short'] = nao_members_short

        # Extract the 5th and 95th percentiles of the NAO index
        model_nao_ts_min = np.percentile(nao_members, 5, axis=0)

        # and the 95th percentile
        model_nao_ts_max = np.percentile(nao_members, 95, axis=0)

        # Extract the 5th and 95th percentiles of the NAO index
        model_nao_ts_short_min = np.percentile(nao_members_short, 5, axis=0)

        # and the 95th percentile
        model_nao_ts_short_max = np.percentile(nao_members_short, 95, axis=0)

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_min'] = model_nao_ts_min

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_max'] = model_nao_ts_max

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_short_min'] = model_nao_ts_short_min

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_short_max'] = model_nao_ts_short_max

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1 = pearsonr(nao_mean, obs_nao)[0]

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_lag = pearsonr(nao_mean_lag, obs_nao_lag)[0]

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_short = pearsonr(nao_mean_short, obs_nao_short)[0]

        # And for the lagged NAO index
        corr1_lag_short = pearsonr(nao_mean_lag_short, obs_nao_lag_short)[0]

        # Calculate the standard deviation of the model NAO index
        nao_std = np.std(nao_mean)
        nao_lag_std = np.std(nao_mean_lag)

        # Calculate the standard deviation of the model NAO index for the short period
        nao_std_short = np.std(nao_mean_short)
        nao_lag_std_short = np.std(nao_mean_lag_short)

        # Calculate the rpc between the model NAO index and the observed NAO index
        rpc1 = corr1 / (nao_std / np.std(nao_members))

        # Calculate the rpc between the model NAO index and the observed NAO index
        rpc1_lag = corr1_lag / (nao_lag_std / np.std(nao_members_lag))

        # Calculate the rpc between the model NAO index and the observed NAO index
        # for the short period
        rpc1_short = corr1_short / (nao_std_short / np.std(nao_members_short))

        # Calculate the rpc between the model NAO index and the observed NAO index
        # for the short period
        rpc1_lag_short = corr1_lag_short / \
            (nao_lag_std_short / np.std(nao_members_lag_short))

        # Calculate the ratio of predictable signals (RPS)
        rps1 = rpc1 * (np.std(obs_nao) / np.std(nao_members))

        # Calculate the ratio of predictable signals (RPS) for the lag
        rps1_lag = rpc1_lag * (np.std(obs_nao_lag) / np.std(nao_members_lag))

        # Calculate the ratio of predictable signals (RPS) for the short period
        rps1_short = rpc1_short * \
            (np.std(obs_nao_short) / np.std(nao_members_short))

        # Calculate the ratio of predictable signals (RPS) for the short period
        rps1_lag_short = rpc1_lag_short * \
            (np.std(obs_nao_lag_short) / np.std(nao_members_lag_short))

        # Adjust the variance of the model NAO index
        nao_var_adjust = nao_mean * rps1
        nao_var_adjust_lag = nao_mean_lag * rps1_lag

        # Adjust the variance of the model NAO index for the short period
        nao_var_adjust_short = nao_mean_short * rps1_short

        # Adjust the variance of the model NAO index for the short period
        nao_var_adjust_lag_short = nao_mean_lag_short * rps1_lag_short

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_lag_var_adjust = pearsonr(nao_var_adjust_lag, obs_nao_lag)[0]

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_lag_var_adjust_short = pearsonr(
            nao_var_adjust_lag_short, obs_nao_lag_short)[0]

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1_lag_var_adjust = pearsonr(nao_var_adjust_lag, obs_nao_lag)[1]

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1_lag_var_adjust_short = pearsonr(
            nao_var_adjust_lag_short, obs_nao_lag_short)[1]

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1_lag_var_adjust'] = corr1_lag_var_adjust

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1_lag_var_adjust_short'] = corr1_lag_var_adjust_short

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1_lag_var_adjust'] = p1_lag_var_adjust

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1_lag_var_adjust_short'] = p1_lag_var_adjust_short

        # Calculate the 5th and 95th percentiles of the lagged NAO index
        nao_var_adjust_lag_min = np.percentile(
            nao_members_lag * rps1_lag, 5, axis=0)

        # Calculate the 5th and 95th percentiles of the lagged NAO index
        nao_var_adjust_lag_max = np.percentile(
            nao_members_lag * rps1_lag, 95, axis=0)

        # And for the short period
        nao_var_adjust_lag_min_short = np.percentile(
            nao_members_lag_short * rps1_lag_short, 5, axis=0)

        # And for the short period
        nao_var_adjust_lag_max_short = np.percentile(
            nao_members_lag_short * rps1_lag_short, 95, axis=0)

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust_min'] = nao_var_adjust_lag_min

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust_max'] = nao_var_adjust_lag_max

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust_min_short'] = nao_var_adjust_lag_min_short

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust_max_short'] = nao_var_adjust_lag_max_short

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_var_adjust'] = nao_var_adjust

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_var_adjust_short'] = nao_var_adjust_short

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust'] = nao_var_adjust_lag

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_var_adjust_short'] = nao_var_adjust_lag_short

        # Append the ensemble members NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_members'] = nao_members_lag

        # Append the ensemble members NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_lag_members_short'] = nao_members_lag_short

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1'] = corr1

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1_short'] = corr1_short

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1'] = rpc1

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1_short'] = rpc1_short

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1_lag'] = rpc1_lag

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1_lag_short'] = rpc1_lag_short

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1 = pearsonr(nao_mean, obs_nao)[1]

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1_short = pearsonr(nao_mean_short, obs_nao_short)[1]

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1'] = p1

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1_short'] = p1_short

        # Append the rps to the dictionary
        nao_stats_dict[model]['RPS1'] = rps1

        # Append the rps to the dictionary
        nao_stats_dict[model]['RPS1_short'] = rps1_short

        # Append the rps to the dictionary
        nao_stats_dict[model]['RPS1_lag'] = rps1_lag

        # Append the rps to the dictionary
        nao_stats_dict[model]['RPS1_lag_short'] = rps1_lag_short

        print("NAO index calculated for the {} model".format(model))

    print("NAO stats dictionary created")

    # Return the dictionary
    return nao_stats_dict

# Define a plotting function for creating the 2 columns x 6 rows plot


def plot_subplots_ind_models(nao_stats_dict: dict,
                             models_list: List[str],
                             short_period: bool = False,
                             lag_and_var_adjust: bool = False
                             ) -> None:
    """
    Creates a series of subplots for the NAO index for each model for the
    different models during the winter season (DJFM). The skill is assessed
    using the correlation, p-value and RPC.

    Plots can be determined by a series of boolean flags.

    Inputs:
    -------

    nao_stats_dict: dict[dict]
        A dictionary containing the NAO stats for each model. The keys are
        the model names and the values are a dictionary containing the NAO
        stats for that model.

    models_list: List[str]
        A list of the model names to be plotted.

    short_period: bool
        If True then the short period is plotted. Default is False.

    lag_and_var_adjust: bool
        If True then the lag and variance adjusted NAO index is plotted.
        Default is False.

    Outputs:
    --------
    None

    """

    # Print statements indicating the boolean flags
    if short_period == True:
        print("Plotting the short period")
    else:
        print("Plotting the long period")

    if lag_and_var_adjust == True:
        print("Plotting the lag and variance adjusted NAO index")
    else:
        print("Plotting the raw NAO index")

    # Set up the figure
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 12),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate over the models
    for i, model in enumerate(models_list):
        print("Plotting the {} model".format(model))
        print("At index {} of the axes".format(i))

        # Set up the axes
        ax = axes[i]

        # Extract the NAO stats for this model
        nao_stats_model = nao_stats_dict[model]

        # Set up the boolean flags
        if short_period == True and lag_and_var_adjust == False:
            print("Plotting the short period and the raw NAO index")

            # Loop over the members
            for i in range(nao_stats_model['nens']):
                print("Plotting member {}".format(i))

                # Extract the NAO index for this member
                nao_member_short = nao_stats_model['model_nao_ts_members_short'][i, :]
                print("NAO index extracted for member {}".format(i))

                # Plot this member
                ax.plot(nao_stats_model['years_short'] - 5, nao_member_short / 100,
                        color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_short'] - 5, nao_stats_model['model_nao_ts_short'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            ax.fill_between(nao_stats_model['years_short'] - 5, nao_stats_model['model_nao_ts_short_min'] / 100,
                            nao_stats_model['model_nao_ts_short_max'] / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_short'] - 5, nao_stats_model['obs_nao_ts_short'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_short']:.2f} "
                         f"(p = {nao_stats_model['p1_short']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_short']:.2f}, "
                         f"N = {nao_stats_model['nens']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == False and lag_and_var_adjust == False:

            print("Plotting the long period and the raw NAO index")

            # Loop over the members
            for i in range(nao_stats_model['nens']):
                print("Plotting member {}".format(i))

                # Extract the NAO index for this member
                nao_member = nao_stats_model['model_nao_ts_members'][i, :]
                print("NAO index extracted for member {}".format(i))

                # Plot this member
                ax.plot(nao_stats_model['years'] - 5, nao_member / 100,
                        color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years'] - 5, nao_stats_model['model_nao_ts'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            ax.fill_between(nao_stats_model['years'] - 5, nao_stats_model['model_nao_ts_min'] / 100,
                            nao_stats_model['model_nao_ts_max'] / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years'] - 5, nao_stats_model['obs_nao_ts'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1']:.2f} "
                         f"(p = {nao_stats_model['p1']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1']:.2f}, "
                         f"N = {nao_stats_model['nens']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == True and lag_and_var_adjust == True:
            print("Plotting the short period and the lag and variance adjusted NAO index")

            # Loop over the members
            # for i in range(nao_stats_model['nens_lag']):
            #     print("Plotting member {}".format(i))

            #     # Extract the NAO index for this member
            #     nao_member_short = nao_stats_model['model_nao_ts_lag_members_short'][i, :]
            #     print("NAO index extracted for member {}".format(i))

            #     # Plot this member
            #     ax.plot(nao_stats_model['years_lag_short'] - 5, nao_member_short / 100,
            #             color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_lag_short'] - 5,
                    nao_stats_model['model_nao_ts_lag_var_adjust_short'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            # TODO: Compute RMSE confidence intervals here
            # RMSE between the ensemble mean and observations
            rmse = np.sqrt(np.mean((nao_stats_model['model_nao_ts_lag_var_adjust_short']
                                    - nao_stats_model['obs_nao_ts_lag_short'])**2))

            # Calculate the upper and lower confidence intervals
            ci_lower = nao_stats_model['model_nao_ts_lag_var_adjust_short'] - (
                rmse)
            ci_upper = nao_stats_model['model_nao_ts_lag_var_adjust_short'] + (
                rmse)

            # Plot the confidence intervals
            ax.fill_between(nao_stats_model['years_lag_short'] - 5, ci_lower / 100,
                            ci_upper / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_lag_short'] - 5, nao_stats_model['obs_nao_ts_lag_short'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_lag_var_adjust_short']:.2f} "
                         f"(p = {nao_stats_model['p1_lag_var_adjust_short']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_lag_short']:.2f}, "
                         f"N = {nao_stats_model['nens_lag']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == False and lag_and_var_adjust == True:
            print("Plotting the long period and the lag and variance adjusted NAO index")

            # Loop over the members
            # for i in range(nao_stats_model['nens_lag']):
            #     print("Plotting member {}".format(i))

            #     # Extract the NAO index for this member
            #     nao_member = nao_stats_model['model_nao_ts_lag_members'][i, :]
            #     print("NAO index extracted for member {}".format(i))

            #     # Plot this member
            #     ax.plot(nao_stats_model['years_lag'] - 5, nao_member / 100,
            #             color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_lag'] - 5, nao_stats_model['model_nao_ts_lag_var_adjust'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            # RMSE between the ensemble mean and observations
            rmse = np.sqrt(np.mean((nao_stats_model['model_nao_ts_lag_var_adjust']
                                    - nao_stats_model['obs_nao_ts_lag'])**2))

            # Calculate the upper and lower confidence intervals
            ci_lower = nao_stats_model['model_nao_ts_lag_var_adjust'] - (rmse)
            ci_upper = nao_stats_model['model_nao_ts_lag_var_adjust'] + (rmse)

            # Plot the confidence intervals
            ax.fill_between(nao_stats_model['years_lag'] - 5, ci_lower / 100,
                            ci_upper / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_lag'] - 5, nao_stats_model['obs_nao_ts_lag'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_lag_var_adjust']:.2f} "
                         f"(p = {nao_stats_model['p1_lag_var_adjust']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_lag']:.2f}, "
                         f"N = {nao_stats_model['nens_lag']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        else:
            raise ValueError("The boolean flags are not set up correctly")

        # Set the axhline
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        ax.set_ylim([-10, 10])

    # Set the y-axis label for the left column
    for ax in axes[0::2]:
        ax.set_ylabel('NAO (hPa)')

    # Set the x-axis label for the bottom row
    if i >= 10:
        ax.set_xlabel('initialisation year')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join("/gws/nopw/j04/canari/users/benhutch/plots/NAO_skill",
                f"NAO_skill_short_period_{short_period}_lag_and_var_adjust_{lag_and_var_adjust}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"),
                dpi=300)

    # Show the figure
    plt.show()

# Function for the plots including the SPNA index
def plot_subplots_ind_models_spna(nao_stats_dict: dict,
                                  models_list: List[str],
                                  short_period: bool = False,
                                  lag_and_var_adjust: bool = False
                                  ) -> None:
    """
    Creates a series of subplots for the NAO index and SPNA index for each model for the
    different models during the winter season (DJFM). The skill is assessed
    using the correlation, p-value and RPC.

    Plots can be determined by a series of boolean flags.

    Inputs:
    -------

    nao_stats_dict: dict[dict]
        A dictionary containing the NAO and SPNA stats for each model. The keys are
        the model names and the values are a dictionary containing the NAO
        stats for that model.

    models_list: List[str]
        A list of the model names to be plotted.

    short_period: bool
        If True then the short period is plotted. Default is False.

    lag_and_var_adjust: bool
        If True then the lag and variance adjusted NAO index is plotted.
        Default is False.

    Outputs:
    --------
    None

    """

    # Print statements indicating the boolean flags
    if short_period == True:
        print("Plotting the short period")
    else:
        print("Plotting the long period")

    if lag_and_var_adjust == True:
        raise ValueError("The lag and variance adjusted NAO index is not available for the SPNA index")
    else:
        print("Plotting the raw NAO index")

    # Set up the figure
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 12),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate over the models
    for i, model in enumerate(models_list):
        print("Plotting the {} model".format(model))
        print("At index {} of the axes".format(i))

        # Set up the axes
        ax = axes[i]

        # Extract the NAO stats for this model
        nao_stats_model = nao_stats_dict[model]

        # Set up the boolean flags
        if short_period == True and lag_and_var_adjust == False:
            print("Plotting the short period and the raw NAO index")
            print("Plotting the short period and the raw SPNA index")

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_short'] - 5, nao_stats_model['model_nao_ts_short'] / 100,
                    color='red', label='dcppA NAO')

            # Plot the 5th and 95th percentiles
            ax.fill_between(nao_stats_model['years_short'] - 5, nao_stats_model['model_nao_ts_short_min'] / 100,
                            nao_stats_model['model_nao_ts_short_max'] / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_short'] - 5, nao_stats_model['obs_nao_ts_short'] / 100,
                    color='black', label='ERA5')
            
            # create a twin axes
            ax2 = ax.twinx()

            # Plot the ensemble mean
            ax2.plot(nao_stats_model['years_short'] - 5, -nao_stats_model['model_spna_short'],
                    color='red', linestyle='--')

            # Plot the 5th and 95th percentiles
            # ax2.fill_between(nao_stats_model['years_short'] - 5, nao_stats_model['model_spna_ts_short_min'],
            #                 nao_stats_model['model_spna_ts_short_max'], color='blue', alpha=0.2)
            
            # Plot the observed NAO index
            ax2.plot(nao_stats_model['years_short'] - 5, -nao_stats_model['obs_spna_short'],
                    color='black', linestyle='--')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_short']:.2f} ({nao_stats_model['corr1_spna_short']:.2f}) "
                         f"(p = {nao_stats_model['p1_short']:.2f}, {nao_stats_model['p1_spna_short']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_short']:.2f} ({nao_stats_model['RPC1_spna_short']:.2f}), "
                         f"N = {nao_stats_model['nens']} ({nao_stats_model['tas_nens']})")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == False and lag_and_var_adjust == False:
            print("Plotting the long period and the raw NAO index")
            print("Plotting the long period and the raw SPNA index")

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years'] - 5, nao_stats_model['model_nao_ts'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            ax.fill_between(nao_stats_model['years'] - 5, nao_stats_model['model_nao_ts_min'] / 100,
                            nao_stats_model['model_nao_ts_max'] / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years'] - 5, nao_stats_model['obs_nao_ts'] / 100,
                    color='black', label='ERA5')
            
            # create a twin axes
            ax2 = ax.twinx()

            # Plot the ensemble mean
            ax2.plot(nao_stats_model['years'] - 5, -nao_stats_model['model_spna'],
                    color='red', linestyle='--')
            
            # Plot the observed NAO index
            ax2.plot(nao_stats_model['years'] - 5, -nao_stats_model['obs_spna'],
                    color='black', linestyle='--')
            
            # Plot the 5th and 95th percentiles
            # ax2.fill_between(nao_stats_model['years'] - 5, nao_stats_model['model_spna_ts_min'],
            #                 nao_stats_model['model_spna_ts_max'], color='blue', alpha=0.2)

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1']:.2f} ({nao_stats_model['corr1_spna']:.2f}) "
                         f"(p = {nao_stats_model['p1']:.2f}, {nao_stats_model['p1_spna']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1']:.2f} ({nao_stats_model['RPC1_spna']:.2f}), "
                         f"N = {nao_stats_model['nens']} ({nao_stats_model['tas_nens']})")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == True and lag_and_var_adjust == True:
            print("Plotting the short period and the lag and variance adjusted NAO index")

            # Loop over the members
            # for i in range(nao_stats_model['nens_lag']):
            #     print("Plotting member {}".format(i))

            #     # Extract the NAO index for this member
            #     nao_member_short = nao_stats_model['model_nao_ts_lag_members_short'][i, :]
            #     print("NAO index extracted for member {}".format(i))

            #     # Plot this member
            #     ax.plot(nao_stats_model['years_lag_short'] - 5, nao_member_short / 100,
            #             color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_lag_short'] - 5,
                    nao_stats_model['model_nao_ts_lag_var_adjust_short'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            # TODO: Compute RMSE confidence intervals here
            # RMSE between the ensemble mean and observations
            rmse = np.sqrt(np.mean((nao_stats_model['model_nao_ts_lag_var_adjust_short']
                                    - nao_stats_model['obs_nao_ts_lag_short'])**2))

            # Calculate the upper and lower confidence intervals
            ci_lower = nao_stats_model['model_nao_ts_lag_var_adjust_short'] - (
                rmse)
            ci_upper = nao_stats_model['model_nao_ts_lag_var_adjust_short'] + (
                rmse)

            # Plot the confidence intervals
            ax.fill_between(nao_stats_model['years_lag_short'] - 5, ci_lower / 100,
                            ci_upper / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_lag_short'] - 5, nao_stats_model['obs_nao_ts_lag_short'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_lag_var_adjust_short']:.2f} "
                         f"(p = {nao_stats_model['p1_lag_var_adjust_short']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_lag_short']:.2f}, "
                         f"N = {nao_stats_model['nens_lag']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        elif short_period == False and lag_and_var_adjust == True:
            print("Plotting the long period and the lag and variance adjusted NAO index")

            # Loop over the members
            # for i in range(nao_stats_model['nens_lag']):
            #     print("Plotting member {}".format(i))

            #     # Extract the NAO index for this member
            #     nao_member = nao_stats_model['model_nao_ts_lag_members'][i, :]
            #     print("NAO index extracted for member {}".format(i))

            #     # Plot this member
            #     ax.plot(nao_stats_model['years_lag'] - 5, nao_member / 100,
            #             color='grey', alpha=0.2)

            # Plot the ensemble mean
            ax.plot(nao_stats_model['years_lag'] - 5, nao_stats_model['model_nao_ts_lag_var_adjust'] / 100,
                    color='red', label='dcppA')

            # Plot the 5th and 95th percentiles
            # RMSE between the ensemble mean and observations
            rmse = np.sqrt(np.mean((nao_stats_model['model_nao_ts_lag_var_adjust']
                                    - nao_stats_model['obs_nao_ts_lag'])**2))

            # Calculate the upper and lower confidence intervals
            ci_lower = nao_stats_model['model_nao_ts_lag_var_adjust'] - (rmse)
            ci_upper = nao_stats_model['model_nao_ts_lag_var_adjust'] + (rmse)

            # Plot the confidence intervals
            ax.fill_between(nao_stats_model['years_lag'] - 5, ci_lower / 100,
                            ci_upper / 100, color='red', alpha=0.2)

            # Plot the observed NAO index
            ax.plot(nao_stats_model['years_lag'] - 5, nao_stats_model['obs_nao_ts_lag'] / 100,
                    color='black', label='ERA5')

            # Set the title with the ACC and RPC scores
            ax.set_title(f"ACC = {nao_stats_model['corr1_lag_var_adjust']:.2f} "
                         f"(p = {nao_stats_model['p1_lag_var_adjust']:.2f}), "
                         f"RPC = {nao_stats_model['RPC1_lag']:.2f}, "
                         f"N = {nao_stats_model['nens_lag']}")

            # Format the model name in the top left of the figure
            ax.text(0.05, 0.95, f"{model}", transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

            # Add the legend in the bottom right corner
            ax.legend(loc='lower right')

        else:
            raise ValueError("The boolean flags are not set up correctly")

        # Set the axhline
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

        ax.set_ylim([-10, 10])

    # Set the y-axis label for the left column
    for ax in axes[0::2]:
        ax.set_ylabel('NAO (hPa)')

    # Set the x-axis label for the bottom row
    if i >= 10:
        ax.set_xlabel('initialisation year')

    # Set the y-axis label for the right column
    for ax in axes[1::2]:
        ax.set_ylabel('SPNA index (-Celsius)')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join("/gws/nopw/j04/canari/users/benhutch/plots/NAO_skill",
                f"NAO_skill_short_period_{short_period}_lag_and_var_adjust_{lag_and_var_adjust}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"),
                dpi=300)

    # Show the figure
    plt.show()



def plot_multi_model_mean(nao_stats_dict: dict,
                          models_list: List[str],
                          lag_and_var_adjust: bool = False
                          ) -> None:
    """
    Plots the multi-model mean NAO time series for the short period (left plot)
    and long period (right plot) during the winter season (DJFM).
    The skill is assessed using the correlation, p-value and RPC.

    Plots can be determined by a series of boolean flags.

    Inputs:
    -------

    nao_stats_dict: dict[dict]
        A dictionary containing the NAO stats for each model. The keys are
        the model names and the values are a dictionary containing the NAO
        stats for that model.

    models_list: List[str]
        A list of the model names which are used to calculate the multi-model
        mean.

    lag_and_var_adjust: bool
        If True then the lag and variance adjusted NAO index is plotted.
        Default is False.

    Outputs:
    --------
    None

    """

    # Set up the length of the time series for raw and lagged
    nyears_short = len(np.arange(1966, 2010 + 1))
    nyears_long = len(np.arange(1966, 2019 + 1))

    # lagged time series
    nyears_short_lag = len(np.arange(1969, 2010 + 1))
    nyears_long_lag = len(np.arange(1969, 2019 + 1))

    # Print statements indicating the boolean flags
    if lag_and_var_adjust is True:
        print("Plotting the lag and variance adjusted NAO index")

        # Set up a counter for the number of lagged ensemble members
        total_lagged_nens = 0

        # Loop over the models
        for model in models_list:
            # Extract the NAO stats for this model
            nao_stats_model = nao_stats_dict[model]

            # Add the number of lagged ensemble members to the counter
            total_lagged_nens += nao_stats_model['nens_lag']

        # Create an empty array to store the lagged ensemble members
        lagged_nao_members = np.zeros([total_lagged_nens, nyears_long_lag])

        # Create an empty array to store the lagged ensemble members
        lagged_nao_members_short = np.zeros(
            [total_lagged_nens, nyears_short_lag])

    else:
        print("Plotting the raw NAO index")

        # Set up a counter for the number of ensemble members
        total_nens = 0

        # Loop over the models
        for model in models_list:
            # Extract the NAO stats for this model
            nao_stats_model = nao_stats_dict[model]

            # Add the number of ensemble members to the counter
            total_nens += nao_stats_model['nens']

        # Create an empty array to store the ensemble members
        nao_members = np.zeros([total_nens, nyears_long])

        # Create an empty array to store the ensemble members
        nao_members_short = np.zeros([total_nens, nyears_short])

    # Set up the figure
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4),
                             sharex=True, sharey=True)

    # Set up the axes
    ax1 = axes[0]
    ax2 = axes[1]

    # Initialise the counter
    current_index = 0

    # Iterate over the models
    for i, model in enumerate(models_list):
        print("Extracting ensemble members from the {} model".format(model))

        # Extract the NAO stats for this model
        nao_stats_model = nao_stats_dict[model]

        # Set up the boolean flags
        if lag_and_var_adjust is False:
            print("Extracting members for the raw NAO index")

            # Loop over the members
            for i in range(nao_stats_model['nens']):
                print("Extracting member {}".format(i))

                # Extract the NAO index for this member
                nao_member = nao_stats_model['model_nao_ts_members'][i, :]
                print("NAO index extracted for member {}".format(i))

                # extract the short period
                nao_member_short = nao_stats_model['model_nao_ts_members_short'][i, :]
                print("NAO index extracted for short period for member {}".format(i))

                # If the model is not BCC-CSM2-MR
                # then we need to skip over the 0th time index
                if model != "BCC-CSM2-MR":
                    # Append this member to the array
                    nao_members[current_index, :] = nao_member[1:]

                    # Append this member to the array
                    nao_members_short[current_index, :] = nao_member_short[1:]
                else:
                    # Append this member to the array
                    nao_members[current_index, :] = nao_member

                    # Append this member to the array
                    nao_members_short[current_index, :] = nao_member_short

                # Increment the counter
                current_index += 1

        elif lag_and_var_adjust is True:
            print("Extracting members for the lag and variance adjusted NAO index")

            # Loop over the members
            for i in range(nao_stats_model['nens_lag']):
                print("Extracting member {}".format(i))

                # Extract the NAO index for this member
                nao_member = nao_stats_model['model_nao_ts_lag_members'][i, :]
                print("NAO index extracted for member {}".format(i))

                # extract the short period
                nao_member_short = nao_stats_model['model_nao_ts_lag_members_short'][i, :]
                print("NAO index extracted for short period for member {}".format(i))

                # If the model is not BCC-CSM2-MR
                # then we need to skip over the 0th time index
                if model != "BCC-CSM2-MR":
                    # Append this member to the array
                    lagged_nao_members[current_index, :] = nao_member[1:]

                    # Append this member to the array
                    lagged_nao_members_short[current_index,
                                             :] = nao_member_short[1:]
                else:
                    # Append this member to the array
                    lagged_nao_members[current_index, :] = nao_member

                    # Append this member to the array
                    lagged_nao_members_short[current_index,
                                             :] = nao_member_short

                # Increment the counter
                current_index += 1

        else:
            raise ValueError("The boolean flags are not set up correctly")

    # Now for the plotting
    # Set up the boolean flags
    if lag_and_var_adjust is False:
        print("Plotting the raw NAO index")

        # count the number of ensemble members
        total_nens = nao_members.shape[0]

        # Calculate the ensemble mean
        nao_mean = np.mean(nao_members, axis=0)

        # Calculate the ensemble mean for the short period
        nao_mean_short = np.mean(nao_members_short, axis=0)

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1, p1 = pearsonr(nao_mean,
                             nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts'])

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_short, p1_short = pearsonr(nao_mean_short,
                                         nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_short'])

        # Calculate the RPC between the model NAO index and the observed NAO index
        rpc1 = corr1 / (np.std(nao_mean) / np.std(nao_members))

        # Calculate the RPC between the model NAO index and the observed NAO index
        rpc1_short = corr1_short / \
            (np.std(nao_mean_short) / np.std(nao_members_short))

        # Calculate the 5th and 95th percentiles
        nao_mean_min = np.percentile(nao_members, 5, axis=0)
        nao_mean_max = np.percentile(nao_members, 95, axis=0)

        # Calculate the 5th and 95th percentiles
        nao_mean_short_min = np.percentile(nao_members_short, 5, axis=0)
        nao_mean_short_max = np.percentile(nao_members_short, 95, axis=0)

        # Plot the ensemble mean
        ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years'] - 5, nao_mean / 100,
                 color='red', label='dcppA')

        # Plot the observed NAO index - time valid for BCC-CSM2-MR
        model = "BCC-CSM2-MR"

        # Extract the NAO stats for this model
        ax2.plot(nao_stats_dict[model]['years'] - 5, nao_stats_dict[model]['obs_nao_ts'] / 100,
                 color='black', label='ERA5')

        # Plot the 5th and 95th percentiles
        ax2.fill_between(nao_stats_dict['BCC-CSM2-MR']['years'] - 5, nao_mean_min / 100,
                         nao_mean_max / 100, color='red', alpha=0.2)

        # Plot the ensemble mean
        ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years_short'] - 5, nao_mean_short / 100,
                 color='red', label='dcppA')

        # Plot the observed NAO index - time valid for BCC-CSM2-MR
        ax1.plot(nao_stats_dict[model]['years_short'] - 5, nao_stats_dict[model]['obs_nao_ts_short'] / 100,
                 color='black', label='ERA5')

        # Plot the 5th and 95th percentiles
        ax1.fill_between(nao_stats_dict['BCC-CSM2-MR']['years_short'] - 5, nao_mean_short_min / 100,
                         nao_mean_short_max / 100, color='red', alpha=0.2)

    elif lag_and_var_adjust is True:
        print("Plotting the lag and variance adjusted NAO index")

        # count the number of ensemble members
        total_nens = lagged_nao_members.shape[0]

        # Calculate the ensemble mean
        nao_mean = np.mean(lagged_nao_members, axis=0)

        # Calculate the ensemble mean for the short period
        nao_mean_short = np.mean(lagged_nao_members_short, axis=0)

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1, p1 = pearsonr(nao_mean,
                             nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_short, p1_short = pearsonr(nao_mean_short,
                                         nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag_short'])

        # Calculate the RPC between the model NAO index and the observed NAO index
        rpc1 = corr1 / (np.std(nao_mean) / np.std(lagged_nao_members))

        # Calculate the RPC between the model NAO index and the observed NAO index
        rpc1_short = corr1_short / \
            (np.std(nao_mean_short) / np.std(lagged_nao_members_short))

        # Calculate the RPS between the model NAO index and the observed NAO index
        rps1 = rpc1 * \
            (np.std(nao_stats_dict['BCC-CSM2-MR']
             ['obs_nao_ts_lag']) / np.std(lagged_nao_members))

        # Calculate the RPS between the model NAO index and the observed NAO index
        rps1_short = rpc1_short * \
            (np.std(nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag_short']
                    ) / np.std(lagged_nao_members_short))

        # Adjust the variance of the lagged NAO index
        nao_var_adjust = nao_mean * rps1

        # Adjust the variance of the lagged NAO index
        nao_var_adjust_short = nao_mean_short * rps1_short

        # Calculate the RMSE between the ensemble mean and observations
        rmse = np.sqrt(np.mean((nao_var_adjust
                                - nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])**2))

        # Calculate the rmse for the short period
        rmse_short = np.sqrt(np.mean((nao_var_adjust_short
                                      - nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag_short'])**2))

        # Calculate the upper and lower confidence intervals
        ci_lower = nao_var_adjust - (rmse)
        ci_upper = nao_var_adjust + (rmse)

        # Calculate the upper and lower confidence intervals
        ci_lower_short = nao_var_adjust_short - (rmse_short)
        ci_upper_short = nao_var_adjust_short + (rmse_short)

        # Plot the ensemble mean
        ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - 5, nao_var_adjust / 100,
                 color='red', label='dcppA')

        # Plot the observed NAO index - time valid for BCC-CSM2-MR
        model = "BCC-CSM2-MR"

        # Extract the NAO stats for this model
        ax2.plot(nao_stats_dict[model]['years_lag'] - 5, nao_stats_dict[model]['obs_nao_ts_lag'] / 100,
                 color='black', label='ERA5')

        # Plot the 5th and 95th percentiles
        ax2.fill_between(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - 5, ci_lower / 100,
                         ci_upper / 100, color='red', alpha=0.2)

        # Plot the ensemble mean
        ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag_short'] - 5, nao_var_adjust_short / 100,
                 color='red', label='dcppA')

        # Plot the observed NAO index - time valid for BCC-CSM2-MR
        ax1.plot(nao_stats_dict[model]['years_lag_short'] - 5,
                 nao_stats_dict[model]['obs_nao_ts_lag_short'] / 100,
                 color='black', label='ERA5')

        # Plot the 5th and 95th percentiles
        ax1.fill_between(nao_stats_dict['BCC-CSM2-MR']['years_lag_short'] - 5, ci_lower_short / 100,
                         ci_upper_short / 100, color='red', alpha=0.2)

    else:
        raise ValueError("The boolean flags are not set up correctly")

        # Set the title with the ACC and RPC scores
    ax1.set_title(f"ACC = {corr1:.2f} (p = {p1:.2f}), "
                  f"RPC = {rpc1:.2f}, "
                  f"N = {total_nens}")

    # Set the title with the ACC and RPC scores
    ax2.set_title(f"ACC = {corr1_short:.2f} (p = {p1_short:.2f}), "
                  f"RPC = {rpc1_short:.2f}, "
                  f"N = {total_nens}")

    # Format the initialisation year range in the top left of the figure
    ax1.text(0.05, 0.95, "1961-2005", transform=ax1.transAxes, ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    # Format the initialisation year range in the top left of the figure
    ax2.text(0.05, 0.95, "1961-2014", transform=ax2.transAxes, ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    # Add the legend in the bottom right corner
    ax1.legend(loc='lower right')

    # Add the legend in the bottom right corner
    ax2.legend(loc='lower right')

    # Set the axhline
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Set the axhline
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Set the y-axis label for the left column
    ax1.set_ylabel('NAO (hPa)')

    # Set the ylims
    ax1.set_ylim([-10, 10])

    # Set the x-axis label for the bottom row
    plt.xlabel('initialisation year')

    # Adjust the layout
    plt.tight_layout()

# Define a function to calculate the SPNA temperature index


def calculate_spna_index(t_anom,
                         gridbox=dic.spna_grid):
    """
    Calculates the SPNA index from a given temperature anomaly field.

    Inputs:
    -------

    t_anom: DataArray
        A 3D array of temperature anomalies with dimensions (time, lat, lon).

    gridbox: dict
        A dictionary containing the lat/lon bounds for the SPNA region.

    Outputs:
    --------

    spna_index: DataArray
        A 1D array of the SPNA index with dimensions (time).

    """

    # Extract the lat/lon bounds
    lon1, lon2 = gridbox['lon1'], gridbox['lon2']
    lat1, lat2 = gridbox['lat1'], gridbox['lat2']

    # Take the mean over the lat/lon bounds
    spna_index = t_anom.sel(lat=slice(lat1, lat2), lon=slice(
        lon1, lon2)).mean(dim=['lat', 'lon'])

    # Return the SPNA index
    return spna_index
