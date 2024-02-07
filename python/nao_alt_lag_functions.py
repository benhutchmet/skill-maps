"""

nao_alt_lag_functions.py

This file contains functions for loading the processed (either bootstrap or .npy) data and then calculating the NAO index
for this. The goal is to plot the NAO index (raw) alongside the alt-lagged (or lagged) NAO index for 
years 1, years 2, years 2-3, years 2-5 and years 2-9, for both the winter NAO and the summer NAO.
"""

# Import relevant libraries
import os
import sys
import glob
import re

# Third-party libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Import local modules
import functions as func
import nao_skill_functions as nao_func

# Import dictionaries
sys.path.append('/home/users/benhutch/skill-maps')
import dictionaries as dict

# Import the functions from the rose-suite-matching repository
sys.path.append('/home/users/benhutch/skill-maps/rose-suite-matching')
import nao_matching_seasons as nms_func

# Set up a function for loading the processed data
def load_data(season: str,
            forecast_range: str,
            start_year: int,
            end_year: int,
            lag: int,
            alt_lag: bool = False,
            region: str = "global",
            variable: str = "psl",
            data_dir: str = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data",):
    """
    Loads the data for the given season, forecast range, start year, end year, lag and region.

    Parameters:
    -----------

    season: str
        The season for which the data is being loaded (either "winter" or "summer").

    forecast_range: str
        The forecast range for which the data is being loaded (either "years1", "years2", "years2-3", "years2-5" or "years2-9").

    start_year: int
        The start year for which the data is being loaded.

    end_year: int
        The end year for which the data is being loaded.

    lag: int
        The lag for which the data is being loaded.

    alt_lag: bool
        Whether the alternate lag data is being loaded (default is False).
        True = alt_lag data
        False = raw data

    region: str
        The region for which the data is being loaded (either "global", "atlantic" or "pacific").
        Default is "global".

    variable: str
        The variable for which the data is being loaded (default is "psl").

    data_dir: str
        The directory in which the data is stored (default is "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data").

    Returns:
    --------

    data: xarray.Dataset
        The data for the given season, forecast range, start year, end year, lag and region.
    """

    # Set up the years for extraction
    if alt_lag:
        if lag is not None:
            first_year = int(start_year) + int(lag) - 1

        # Set up the end year
        last_year = int(end_year)

        # Set up the file path
        filename = f"{variable}_{season}_{region}_{first_year}_{last_year}_{forecast_range}_{lag}_*alternate_lag.npy"

        # find files matching the filename
        files = glob.glob(data_dir + "/" + filename)

        # Assert that files is not empty
        assert files, f"No files found for {filename}"

        # If there is more than one file
        if len(files) > 1:
            print("More than one file found")

            # If the psl_DJFM_global_1962_1980_2-9_2_1706281292.628301_alternate_lag.npy
            # 1706281292.628301 is the datetime
            # Extract the datetimes
            datetimes = [file.split("_")[7] for file in files]

            # Remove the .npy from the datetimes
            datetimes = [datetime.split(".")[0] for datetime in datetimes]

            # Convert the datasetimes to datetimes using pandas
            datetimes = [pd.to_datetime(datetime, unit="s") for datetime in datetimes]

            # Find the latest datetime
            latest_datetime = max(datetimes)

            # Find the index of the latest datetime
            latest_datetime_index = datetimes.index(latest_datetime)

            # Print that we are using the latest datetime file
            print("Using the latest datetime file:", files[latest_datetime_index])

            # Load the file
            alt_lag_data = np.load(files[latest_datetime_index])
        else:
            # Load the file
            alt_lag_data = np.load(files[0])

        # print the shape of the data
        print("Shape of the data:", alt_lag_data.shape)

        # Return the data
        return alt_lag_data
    else:
        # Set up the file path
        filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}*.npy"

        # find files matching the filename
        files = glob.glob(data_dir + "/" + filename)

        # Assert that files is not empty
        assert files, f"No files found for {filename}"

        # If there is more than one file
        if len(files) > 1:
            print("More than one file found")

            # If the psl_DJFM_global_1962_1980_2-9_2.npy
            # 1706281292.628301 is the datetime
            # Extract the datetimes
            datetimes = [file.split("_")[7] for file in files]

            # Remove the .npy from the datetimes
            datetimes = [datetime.split(".")[0] for datetime in datetimes]

            # Convert the datasetimes to datetimes using pandas
            datetimes = [pd.to_datetime(datetime, unit="s") for datetime in datetimes]

            # Find the latest datetime
            latest_datetime = max(datetimes)

            # Find the index of the latest datetime
            latest_datetime_index = datetimes.index(latest_datetime)

            # Print that we are using the latest datetime file
            print("Using the latest datetime file:", files[latest_datetime_index])

            # Load the file
            data = np.load(files[latest_datetime_index])
        else:
            # Load the file
            data = np.load(files[0])

        # print the shape of the data
        print("Shape of the data:", data.shape)

        # Return the data
        return data
    
# Set up a function for calculating the NAO index
def calc_nao_stats(data: np.ndarray,
                season: str,
                forecast_range: str,
                start_year: int,
                end_year: int,
                lag: int,
                alt_lag: bool = False,
                region: str = "global",
                variable: str = "psl",
):
        """
        Calculates the NAO index for the given data.
        
        Parameters:
        -----------
        
        data: np.ndarray
            The data for which the NAO index is being calculated.
            
        season: str
            The season for which the NAO index is being calculated.

        forecast_range: str
            The forecast range for which the NAO index is being calculated.

        start_year: int
            The start year for which the NAO index is being calculated.

        end_year: int
            The end year for which the NAO index is being calculated.

        lag: int
            The lag for which the NAO index is being calculated.

        region: str
            The region for which the NAO index is being calculated (either "global", "atlantic" or "pacific").
            Default is "global".

        variable: str
            The variable for which the NAO index is being calculated (default is "psl").

        Returns:

        nao_stats: dict
            A dictionary containing the NAO index for the given data.
        """

        # Set up the mdi
        mdi = -9999.0

        # Form the dictionary
        nao_stats = {
            'obs_nao': [],
            'model_nao_mean': [],
            'model_nao_members': [],
            'years': [],
            'corr1': mdi,
            'p1': mdi,
            'rpc1': mdi,
            'rps1': mdi,
            'nens': mdi
        }

        # Set up the lats and lons
        lons = np.arange(-180, 180, 2.5) ; lats = np.arange(-90, 90, 2.5)

        # Depending on the season select the NAO gridboxes
        if season in ["DJFM", "DJF", "ONDJFM", "MAM"]:
            print("Using standard NAO definition")
            # Hardcoded for now
            south_grid = dict.azores_grid_corrected
            north_grid = dict.iceland_grid_corrected
        else:
            print("Using summer NAO definition")
            # Hardcoded for now
            south_grid = dict.snao_south_grid
            north_grid = dict.snao_north_grid

        # Extract the lats and lons for the south grid
        s_lon1, s_lon2 = south_grid['lon1'], south_grid['lon2']
        s_lat1, s_lat2 = south_grid['lat1'], south_grid['lat2']

        # Extract the lats and lons for the north grid
        n_lon1, n_lon2 = north_grid['lon1'], north_grid['lon2']
        n_lat1, n_lat2 = north_grid['lat1'], north_grid['lat2']

        # First process the obs NAO
        # FIXME: make sure this read_obs function works for years 1 and years 2
        obs_psl_anom = func.read_obs(variable=variable,
                                     region=region,
                                     forecast_range=forecast_range,
                                     season=season,
                                     observations_path=nms_func.find_obs_path(variable=variable,),
                                     start_year=start_year,
                                     end_year=end_year,
        )

        # TODO: Make sure this lines up with the model data
        # Set up the first and last years accordingly
        if len(data) == 5:
            print("Processing the raw data")

            # If forecast range is 2-9
            if forecast_range == "2-9":
                # Set up the raw first and last years
                raw_first_year = int(start_year) + 5
                raw_last_year = int(end_year) + 5
            elif forecast_range == "2-5":
                # Set up the raw first and last years
                raw_first_year = int(start_year) + 3
                raw_last_year = int(end_year) + 3
            elif forecast_range == "2-3":
                # Set up the raw first and last years
                raw_first_year = int(start_year) + 2
                raw_last_year = int(end_year) + 2
            elif forecast_range == "2":
                # Set up the raw first and last years
                raw_first_year = int(start_year) + 1
                raw_last_year = int(end_year) + 1
            elif forecast_range == "1":
                # Set up the raw first and last years
                raw_first_year = int(start_year)
                raw_last_year = int(end_year)
            else:
                print("Forecast range not recognised")

            # If the season is not DJFM
            if season not in ["DJFM", "DJF", "ONDJFM"]:
                # Add 1 to the raw first and last years
                raw_first_year += 1
                raw_last_year += 1

            # Set up the common years accordingly
            common_years = np.arange(raw_first_year, raw_last_year + 1)

            # Constrain the obs_psl_anom to the common years
            obs_psl_anom = obs_psl_anom.sel(time=slice(f"{raw_first_year}-01-01",
                                                       f"{raw_last_year}-12-31"))

            # extract the data for the south grid
            obs_psl_anom_south = obs_psl_anom.sel(lat=slice(s_lat1, s_lat2),
                                                  lon=slice(s_lon1, s_lon2).mean(dim=["lat", "lon"]))

            # extract the data for the north grid
            obs_psl_anom_north = obs_psl_anom.sel(lat=slice(n_lat1, n_lat2),
                                                  lon=slice(n_lon1, n_lon2).mean(dim=["lat", "lon"]))

            # Calculate the NAO index
            obs_nao = obs_psl_anom_south - obs_psl_anom_north

            # Print the shape of the obs_nao
            print("Shape of the obs_nao:", obs_nao.shape)

            # Loop over the years
            for year in obs_nao.time.dt.year.values:
                # Extract the data for the year
                years_obs_nao = obs_nao.sel(time=f"{year}")

                # If there are any nans, raise an error
                if np.isnan(years_obs_nao).any():
                    print("Nans found in obs for year:", year)
                    if np.isnan(years_obs_nao).all():
                        print("All values are nan")
                        print("Removing year:", year, "from obs")
                        obs_nao = obs_nao.sel(time=obs_nao.time.dt.year != year)

            # append the obs_nao to the dictionary
            nao_stats['obs_nao'] = obs_nao

            # Swap the axes of the data
            data = np.swapaxes(data, 0, 1)

            # Print the shape of the data
            print("Shape of the data:", data.shape)

            # If the third axis has size > 1
            if data.shape[2] > 1:
                # Calculate the mean of the data
                # Extract the second number in forecast_range
                forecast_range_number = int(forecast_range.split("-")[1])

                # Calculate the mean of the data
                data = data[:, :, :forecast_range_number - 1, :, :].mean(axis=2)
            elif data.shape[2] == 1:
                # Squeeze the data
                data = np.squeeze(data)

            # Print the shape of the data
            print("Shape of the data:", data.shape)

            # Assert that the shape of lats is the same as the shape of the data third axis
            assert data.shape[2] == len(lats), "Data lats shape not equal to lats shape"

            # Assert that the shape of lons is the same as the shape of the data fourth axis
            assert data.shape[3] == len(lons), "Data lons shape not equal to lons shape"

            



        elif len(data) == 4:
            print("Processing the alt-lag data")

        else:
            print("Data length not recognised")