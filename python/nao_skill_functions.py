# Functions for exploring the NAO skill
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
import dictionaries as dic


# Define a function for the NAO stats
def nao_stats(obs: DataArray, 
            hindcast: Dict[str, List[DataArray]],
            models_list: List[str],
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
    
    obs[time, lat, lon]: DataArray
        Observations of the NAO index.
        
    hindcast: Dict[str, List[DataArray]]
        A dictionary containing the hindcasts for each model. The keys are
        the model names and the values are a list of DataArrays containing
        the hindcast data.
        
    models_list: List[str]
        A list of the model names

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

    # Hard code in the dictionaries containing
    # the grid boxes for the NAO index
    azores_grid = dic.azores_grid_corrected
    iceland_grid = dic.iceland_grid_corrected

    # Create a dictionary to store the NAO stats for each model
    nao_stats_dict = {}

    # Set up the missing data indicator
    mdi = -9999.0

    # Loop over the models
    for model in models_list:
        print("Setting up the NAO stats for the {} model".format(model))

        # Create a dictionary for the NAO stats for this model
        nao_stats_dict[model] = {

            'years': [], 'years_lag': [], 'obs_nao_ts': [], 'model_nao_ts': [],

            'model_nao_ts_members': [], 

            'model_nao_ts_min': [], 'model_nao_ts_max': [], 

            'model_nao_ts_var_adjust': [], 'model_nao_ts_var_adjust_short': [],
             
            'model_nao_ts_lag_var_adjust': [],

            'model_nao_ts_lag_var_adjust_min': [],

            'model_nao_ts_lag_var_adjust_max': [], 'corr1': mdi, 

            'corr1_short': mdi, 'corr1_lag': mdi, 'corr1_lag_short': mdi,

            'p1': mdi, 'p1_short': mdi, 'p1_lag': mdi, 'p1_lag_short': mdi,

            'RPC1': mdi, 'RPC1_short': mdi, 'RPC1_lag': mdi,

            'RPC1_lag_short': mdi, 'short_period': short_period,

            'long_period': (), 'short_period_lag': (), 'long_period_lag': (),

            'nens': mdi, 'nens_lag': mdi

        }

        # Extract the list of hindcast DataArrays for this model
        hindcast_list = hindcast[model]

        # Ensure that each of the data arrays has the same time axis
        # Extract the years for the first member
        years1 = hindcast_list[0].time.dt.year.values

        # Assert that this doesn't have any duplicate values
        assert len(years1) == len(set(years1)), \
            "The years in the hindcast data for the {} model are not unique".format(model)
        
        # Assert that there are no gaps of more than one year between the years
        assert np.all(np.diff(years1) <= 1), \
            "There is a gap of more than one year in the hindcast data for the {} model".format(model)
        
        # Assert that there are at least 10 years in the hindcast data
        assert len(years1) >= 10, \
            "There are less than 10 years in the hindcast data for the {} model".format(model)

        # Loop over the remaining members
        for member in hindcast_list[1:]:

            # Extract the years for this member
            years2 = member.time.dt.year.values

            # Assert that this doesn't have any duplicate values
            assert len(years2) == len(set(years2)), \
                "The years in the hindcast data for the {} model are not unique".format(model)
            
            # Assert that there are no gaps of more than one year between the years
            assert np.all(np.diff(years2) <= 1), \
                "There is a gap of more than one year in the hindcast data for the {} model".format(model)

            # Assert that there are at least 10 years in the hindcast data
            assert len(years2) >= 10, \
                "There are less than 10 years in the hindcast data for the {} model".format(model)

            # If years1 and years2 are not the same then raise a value error
            assert np.all(years1 == years2), \
                "The years in the hindcast data for the {} model are not the same".format(model)
            
        print("years checking complete for the {} model".format(model))

        # Ensure that the observations and the hindcast have the same time axis
        # Extract the years for the observations
        years_obs = obs.time.dt.year.values

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
        assert np.all(np.isnan(obs) == False), \
            "There are NaNs in the observations"
        
        # Create a temporary copy of the observations
        obs_tmp = obs.copy()
        
        # If the values in years1 and years_obs are not the same then raise a value error
        if np.array_equal(years1, years_obs) == False:
            print("The years in the observations are not the same as the years in the hindcast data for the {} model".format(model))
            print("Obs first year: {}".format(years_obs[0]))
            print("Hindcast first year: {}".format(years1[0]))
            print("Obs last year: {}".format(years_obs[-1]))
            print("Hindcast last year: {}".format(years1[-1]))

            # assert that the obs is longer than the hindcast
            assert len(years_obs) > len(years1), \
                "The observations are shorter than the hindcast data for the {} model".format(model)

            # TODO: obs getting shorter
            # Extract only the hindcast years from the observations
            obs_tmp = obs_tmp.sel(time=obs_tmp.time.dt.year.isin(years1))

        # Assert that year 1 of the observations is the same as year 1 of the hindcast
        assert obs_tmp.time.dt.year.values[0] == years1[0], \
            "The first year of the observations is not the same as the first year of the hindcast data for the {} model".format(model)

        # Assert that year -1 of the observations is the same as year -1 of the hindcast
        assert obs_tmp.time.dt.year.values[-1] == years1[-1], \
            "The last year of the observations is not the same as the last year of the hindcast data for the {} model".format(model)

        # Assert that the length of the observations is the same as the length of the hindcast
        assert len(obs_tmp.time.dt.year.values) == len(years1), \
            "The length of the observations is not the same as the length of the hindcast data for the {} model".format(model)
        
        print("years checking complete for the observations and the {} model".format(model))

        # Append the years to the dictionary
        nao_stats_dict[model]['years'] = years1

        # Append the short period to the dictionary
        nao_stats_dict[model]['short_period'] = short_period

        # Append the long period to the dictionary
        nao_stats_dict[model]['long_period'] = (years1[0], years1[-1])

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
        obs_nao_short = obs_nao.sel(time=obs_nao.time.dt.year.isin(years_short))
        
        # Convert the observed NAO index to a numpy array
        obs_nao = obs_nao.values

        # Convert the observed NAO index for the short period to a numpy array
        obs_nao_short = obs_nao_short.values

        # Append the observed NAO index to the dictionary
        nao_stats_dict[model]['obs_nao_ts'] = obs_nao

        # Create an empty array to store the NAO index for each member
        nao_members = np.zeros((len(hindcast_list), len(years1)))

        # Create an empty array to store the NAO index for each member
        # for the short period
        nao_members_short = np.zeros((len(hindcast_list), len(years_short)))

        # Set up the number of ensemble members
        nens = len(hindcast_list)

        # Set up the number of lagged members
        nens_lag = len(hindcast_list) * lag

        # Set up the lagged ensemble
        nao_members_lag = np.zeros([nens_lag, len(years1)])

        # Loop over the hindcast members to calculate the NAO index
        for i, member in enumerate(hindcast_list):

            # Calculate the NAO index for this member
            nao_member = calculate_obs_nao(obs_anomaly=member,
                                    south_grid=azores_grid,
                                    north_grid=iceland_grid)
            
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
                        nao_members_lag[i + (lag_index * nens), year] = nao_member[year - lag_index]

            # Now remove the first lag - 1 years from the NAO index
            nao_members_lag[i, lag - 1:] = np.nan

            # Form the lagged obs
            obs_nao_lag = obs_nao[lag - 1:]

            # Constrain to the short period
            nao_member_short = nao_member.sel(time=nao_member.time.dt.year.isin(years_short))

            # Append the NAO index to the members array
            nao_members[i, :] = nao_member

            # Append the NAO index to the members array
            nao_members_short[i, :] = nao_member_short

        # Calculate the ensemble mean NAO index
        nao_mean = np.mean(nao_members, axis=0)

        # Calculate the ensemble mean NAO index for the short period
        nao_mean_short = np.mean(nao_members_short, axis=0)

        # Append the ensemble mean NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts'] = nao_mean

        # Append the ensemble members NAO index to the dictionary
        nao_stats_dict[model]['model_nao_ts_members'] = nao_members

        # Extract the 5th and 95th percentiles of the NAO index
        model_nao_ts_min = np.percentile(nao_members, 5, axis=0)

        # and the 95th percentile
        model_nao_ts_max = np.percentile(nao_members, 95, axis=0)

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_min'] = model_nao_ts_min

        # Append the 5th and 95th percentiles to the dictionary
        nao_stats_dict[model]['model_nao_ts_max'] = model_nao_ts_max

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1 = pearsonr(nao_mean, obs_nao)[0]

        # Calculate the correlation between the model NAO index and the observed NAO index
        corr1_short = pearsonr(nao_mean_short, obs_nao_short)[0]

        # Calculate the standard deviation of the model NAO index
        nao_std = np.std(nao_mean)

        # Calculate the standard deviation of the model NAO index for the short period
        nao_std_short = np.std(nao_mean_short)

        # Calculate the rpc between the model NAO index and the observed NAO index
        rpc1 = corr1 / (nao_std / np.std(nao_members, axis=0))

        # Calculate the rpc between the model NAO index and the observed NAO index
        # for the short period
        rpc1_short = corr1_short / (nao_std_short / np.std(nao_members_short, axis=0))

        # Calculate the ratio of predictable signals (RPS)
        rps1 = rpc1 * (np.std(obs_nao) / np.std(nao_members, axis=0))

        # Calculate the ratio of predictable signals (RPS) for the short period
        rps1_short = rpc1_short * (np.std(obs_nao_short) / np.std(nao_members_short, axis=0))

        # Adjust the variance of the model NAO index
        nao_var_adjust = nao_mean * rps1

        # Adjust the variance of the model NAO index for the short period
        nao_var_adjust_short = nao_mean_short * rps1_short

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_var_adjust'] = nao_var_adjust

        # Append the adjusted variance to the dictionary
        nao_stats_dict[model]['model_nao_ts_var_adjust_short'] = nao_var_adjust_short

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1'] = corr1

        # Append the correlation to the dictionary
        nao_stats_dict[model]['corr1_short'] = corr1_short

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1'] = rpc1

        # Append the rpc to the dictionary
        nao_stats_dict[model]['RPC1_short'] = rpc1_short

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1 = pearsonr(nao_mean, obs_nao)[1]

        # Calculate the p-value for the correlation between the model NAO index and the observed NAO index
        p1_short = pearsonr(nao_mean_short, obs_nao_short)[1]

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1'] = p1

        # Append the p-value to the dictionary
        nao_stats_dict[model]['p1_short'] = p1_short
        
        print("NAO index calculated for the {} model".format(model))

    print("NAO stats dictionary created")