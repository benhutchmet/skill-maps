"""
Functions for exploring the position of the jet stream and the SNAO in the observations and models.
"""

# Local imports
import os
import sys
import glob
import random

# Third-party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import os.path
from pathlib import Path

# Import dictionaries
sys.path.append('/home/users/benhutch/skill-maps/')
import dictionaries as dicts

# Write a function which will plot the climatology of winds
# at a certain level for a specified region and time period
def calculate_climatology(
        data_path: str,
        variable: str,
        season: str,
        region: dict,
        level: int,
        years: list = None
):
    """
    Function which calculates the climatology for a specified variable, season, region and level.
    
    Args:
        data_path (str): Path to the data.
        variable (str): Variable to calculate the climatology for.
        season (str): Season to calculate the climatology for.
        region (dict): Dictionary containing the region to calculate the climatology for.
        level (int): Level to calculate the climatology for. (Mb)
        years (list): List of years to calculate the climatology for. If None, the climatology will be calculated for all years in the dataset.
        
    Returns:
            climatology (xarray.DataArray): Climatology of the specified variable for the specified region and level.        
    """

    # Load the data
    data = xr.open_dataset(data_path, chunks={'time': 100, 'lat': 100, 'lon': 100})

    # Select the level
    data = data.sel(level=level)

    # Extract the region
    lon1, lon2 = region['lon1'], region['lon2']
    lat1, lat2 = region['lat1'], region['lat2']

    # If the lats for the data are in descending order, reverse the lats
    if data['latitude'].values[0] > data['latitude'].values[-1]:
        lat1, lat2 = lat2, lat1

    # Select the region
    data = data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))

    # Select the season
    months ={
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    # Extract the months
    months = months[season]

    # Select the months
    data = data.sel(time=data['time.month'].isin(months))

    # If years are specified, select the years
    if years is not None:
        data = data.sel(time=data['time.year'].isin(years))

    # Calculate the climatology
    climatology = data.mean(dim='time')

    return climatology

# Write a function for diagnosing the SWJ index
def diagnose_swj_index(data_path: str,
                       season: str = 'JJA',
                       variable: str = 'u',
                       lon1: float = 320.0,
                       lon2: float = 340.0,
                       lat1: float = 90.0,
                       lat2: float = 0.0,
                       level: int = 300):
    """
    Diagnoses the Summer Westerly Jet (SWJ) position index for a specified variable, level and region. This SWJ position index
    is defined as the latitude of peak magnitudes of 300-hPa zonal
    winds averaged over 320°E–340°E.
    
    Parameters:
    ------------
    
    data_path: str
        Path to the data.
        
    variable: str
        Variable to calculate the SWJ index for.
        
    lon1: float
        Western boundary of the region to calculate the SWJ index for.
        
    lon2: float
        Eastern boundary of the region to calculate the SWJ index for.
        
    level: int
        Level to calculate the SWJ index for. (Mb)
        
    Returns:
    ------------
    
    swj_index: xarray.DataArray
        SWJ index for the specified variable, level and region.
    """

    # Load the data
    data = xr.open_dataset(data_path, chunks={'time': 100, 'lat': 100, 'lon': 100})

    # Assert that the months are in the data
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    # Select the season
    months = seasons[season]

    # Assert that only the specified months are in the data
    assert all(month in data['time.month'].values for month in months), 'The specified months are not in the data.'

    # Select the level
    data = data.sel(level=level)

    # Select the region
    data = data.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))

    # Calculate the annual mean
    data = data.groupby('time.year').mean(dim='time')

    # Calculate the SWJ index
    # Find the latitude of peak magnitudes of 300-hPa zonal winds averaged over 320°E–340°E
    swj_index = data.mean(dim='lon').max(dim='lat')

    return swj_index




