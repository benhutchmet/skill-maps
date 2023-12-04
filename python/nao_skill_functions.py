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


# Define a function for the NAO stats
def nao_stats(obs: DataArray, 
            hindcast: Dict[str, List[DataArray]],
            models_list: List[str],
            lag: int = 3,
            short_period: tuple = (1965, 2010)):

    """
    Assess and compare the skill of the NAO index between different models
    and observations.
    
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
        
    Outputs:
    --------

    nao_stats: dict[dict]

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

    # Set up the missing data indicator
    mdi = -9999.0

    # Loop over the models
    for model in models_list:
        print("Setting up the NAO stats for the {} model".format(model))

        # Create a dictionary for the NAO stats for this model
        nao_stats[model] = {
            
            'years': [], 'years_lag': [], 'obs_nao_ts': [], 'model_nao_ts': [],

            'model_nao_ts_min': [], 'model_nao_ts_max': [], 

            'model_nao_ts_var_adjust': [], 'model_nao_ts_lag_var_adjust': [],

            'model_nao_ts_lag_var_adjust_min': [],

            'model_nao_ts_lag_var_adjust_max': [], 'corr1': mdi, 

            'corr1_short': mdi, 'corr1_lag': mdi, 'corr1_lag_short': mdi,

            'p1': mdi, 'p1_short': mdi, 'p1_lag': mdi, 'p1_lag_short': mdi,

            'RPC1': mdi, 'RPC1_short': mdi, 'RPC1_lag': mdi,

            'RPC1_lag_short': mdi, 'short_period': short_period,

            'long_period': (), 'short_period_lag': (), 'long_period_lag': (),

            'nens': mdi, 'nens_lag': mdi

        }