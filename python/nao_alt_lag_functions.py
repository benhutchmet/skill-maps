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
from scipy.stats import pearsonr

# Import local modules
import functions as func
import nao_skill_functions as nao_func

# Import dictionaries
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts

# Import the functions from the rose-suite-matching repository
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
import nao_matching_seasons as nms_func


# Set up a function for loading the processed data
def load_data(
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    lag: int,
    method: str = None,
    region: str = "global",
    variable: str = "psl",
    data_dir: str = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data",
):
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
    if method == "alt_lag":
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

    elif method == "nao_matched":
        print("Loading the nao matched data")

        # Set up the last year
        last_year = int(end_year)

        # Set up the file path
        filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_*nao_matched*.npy"

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
            datetimes = [file.split("_")[8] for file in files]

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


# Define a function to load the saved historical data
# psl in this case
def load_hist_data(
    season: str,
    start_year: int = 1961,
    end_year: int = 2023,
    forecast_range: str = "2-9",
    lag_period: bool = False,
    variable: str = "psl",
    experiment: str = "historical_ssp245",
    lagged_data: bool = False,
    data_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/saved_files/arrays/",
    lag: int = 0,
):
    """
    Loads the historical data.

    Parameters:

    season: str
        The season for which the historical data is being loaded.

    start_year: int
        The start year for which the historical data is being loaded.

    end_year: int
        The end year for which the historical data is being loaded.

    forecast_range
        The forecast range for the data.
        Default is year 2-9.

    lag_period: bool
        Whether the data is for a lag period (default is False).

    variable: str
        The variable for which the historical data is being loaded (default is "psl").

    experiment: str
        The experiment for which the historical data is being loaded (default is "historical_ssp245").

    lagged_data: bool
        Whether the data is over the lagged (shorter) period.
        Default is false.

    data_dir: str
        The directory in which the historical data is stored (default is "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/saved_files/arrays/").

    Returns:

    hist_data: np.ndarray
        The historical data.
    """

    # assert that the data_dir exists
    assert os.path.exists(data_dir), f"{data_dir} does not exist"

    # Set up the dir path for the saved data
    dir_path = f"{data_dir}/{variable}/{season}/{forecast_range}/"

    # assert that this dir path is not empty
    assert os.path.exists(dir_path), f"{dir_path} does not exist"

    # Set up the filename
    if lag_period:
        if lagged_data:
            # psl_ONDJFM_2-9_1961-2023_lag_4_historical_ssp245_lag.npy
            filename = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_lag_{lag}_{experiment}_lag.npy"
        else:
            filename = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_{experiment}_lag.npy"
    else:
        if lagged_data:
            # psl_ONDJFM_2-9_1961-2023_lag_4_historical_ssp245_raw.npy
            filename = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_lag_{lag}_{experiment}_raw.npy"
        else:
            filename = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_{experiment}_raw.npy"

    # Set up the file path
    file_path = f"{dir_path}{filename}"

    # if the file path does not exist
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist")

        # exit with an error
        sys.exit(1)

    # Load the data
    hist_data = np.load(file_path)

    # print the shape of the data
    print("Shape of the historical data:", hist_data.shape)

    # Return the historical data
    return hist_data


# Set up a function for calculating the NAO index
def calc_nao_stats(
    data: np.ndarray,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    lag: int,
    alt_lag: bool = False,
    nao_matched: bool = False,
    region: str = "global",
    variable: str = "psl",
    winter_nao_n_grid: dict = dicts.iceland_grid_corrected,
    winter_nao_s_grid: dict = dicts.azores_grid_corrected,
    hist_data: np.ndarray = None,
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

    winter_nao_n_grid: dict
        The dictionary containing the gridboxes for the winter NAO north region.

    winter_nao_s_grid: dict
        The dictionary containing the gridboxes for the winter NAO south region.

    hist_data: bool
        Whether the data is historical data (default is False).

    Returns:

    nao_stats: dict
        A dictionary containing the NAO index for the given data.
    """

    # Set up the mdi
    mdi = -9999.0

    # Form the dictionary
    nao_stats = {
        "obs_nao": [],
        "model_nao_mean": [],
        "model_nao_members": [],
        "model_nao_members_min": [],
        "model_nao_members_max": [],
        "init_years": [],
        "valid_years": [],
        "corr1": mdi,
        "p1": mdi,
        "rpc1": mdi,
        "rps1": mdi,
        "nens": mdi,
    }

    if alt_lag:
        # Set up the years
        years = np.arange(start_year + lag - 1, end_year + 1)
    elif forecast_range == "2-9" and season not in ["DJFM", "DJF", "ONDJFM"]:
        years = np.arange(start_year, end_year)
    else:
        # Set up the years
        years = np.arange(start_year, end_year + 1)

    # Append the years to the dictionary
    nao_stats["init_years"] = years

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Depending on the season select the NAO gridboxes
    if season in ["DJFM", "DJF", "ONDJFM", "MAM"]:
        print("Using standard NAO definition")
        # Hardcoded for now
        south_grid = winter_nao_s_grid
        north_grid = winter_nao_n_grid
    else:
        print("Using summer NAO definition")
        # Hardcoded for now
        south_grid = dicts.snao_south_grid
        north_grid = dicts.snao_north_grid

    # Extract the lats and lons for the south grid
    s_lon1, s_lon2 = south_grid["lon1"], south_grid["lon2"]
    s_lat1, s_lat2 = south_grid["lat1"], south_grid["lat2"]

    # Extract the lats and lons for the north grid
    n_lon1, n_lon2 = north_grid["lon1"], north_grid["lon2"]
    n_lat1, n_lat2 = north_grid["lat1"], north_grid["lat2"]

    # If the forecast range is a single digit
    if "-" not in forecast_range:
        forecast_range_obs = "1"
    else:
        forecast_range_obs = forecast_range

    # Print the forecast range obs
    print("Forecast range obs:", forecast_range_obs)

    # First process the obs NAO
    # FIXME: make sure this read_obs function works for years 1 and years 2
    obs_psl_anom = func.read_obs(
        variable=variable,
        region=region,
        forecast_range=forecast_range_obs,
        season=season,
        observations_path=nms_func.find_obs_path(
            match_var=variable,
        ),
        start_year=1960,
        end_year=2023,
    )

    # Set up the lats
    obs_lats = obs_psl_anom.lat.values

    # Set up the lons
    obs_lons = obs_psl_anom.lon.values

    # Assert that the lats and lons are equivalent to the obs
    assert np.array_equal(lats, obs_lats), "Lats not equal to obs lats"

    # Assert that the lons are equivalent to the obs
    assert np.array_equal(lons, obs_lons), "Lons not equal to obs lons"

    # Find the indices which correspond
    s_lat1_idx, s_lat2_idx = np.argmin(np.abs(lats - s_lat1)), np.argmin(
        np.abs(lats - s_lat2)
    )

    # Find the indices which correspond
    s_lon1_idx, s_lon2_idx = np.argmin(np.abs(lons - s_lon1)), np.argmin(
        np.abs(lons - s_lon2)
    )

    # Find the indices which correspond
    n_lat1_idx, n_lat2_idx = np.argmin(np.abs(lats - n_lat1)), np.argmin(
        np.abs(lats - n_lat2)
    )

    # Find the indices which correspond
    n_lon1_idx, n_lon2_idx = np.argmin(np.abs(lons - n_lon1)), np.argmin(
        np.abs(lons - n_lon2)
    )

    # Print the shape of the data
    print("data shape", np.shape(data))

    # Print the len of the data
    print("len data", len(data))

    # if hist_data is none
    if hist_data is None:
        # Set up an array of nans with shape (100, 10)
        hist_data = np.full((100, 10), np.nan)

    # TODO: Make sure this lines up with the model data
    # Set up the first and last years accordingly
    if data.ndim == 5 or len(hist_data[1]) == 54:
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

        # Print the raw first year and last year
        print("Raw first year:", raw_first_year)
        print("Raw last year:", raw_last_year)

        # Set up the common years accordingly
        common_years = np.arange(raw_first_year, raw_last_year + 1)

        # Append the valid years
        nao_stats["valid_years"] = common_years

        # Constrain the obs_psl_anom to the common years
        obs_psl_anom = obs_psl_anom.sel(
            time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31")
        )

        # extract the data for the south grid
        obs_psl_anom_south = obs_psl_anom.sel(
            lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)
        ).mean(dim=["lat", "lon"])

        # extract the data for the north grid
        obs_psl_anom_north = obs_psl_anom.sel(
            lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index: azores - iceland
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

        # Extract obs_nao as its values
        obs_nao = obs_nao.values

        # append the obs_nao to the dictionary
        nao_stats["obs_nao"] = obs_nao

        # Swap the axes of the data
        data = np.swapaxes(data, 0, 1)

        # Print the shape of the data
        print("Shape of the data:", np.shape(data))
        # print the shape of the 2nd axis
        print("Shape of the 2nd axis:", data.shape[2])

        # if hist_data has 4 dimensions
        if hist_data.ndim == 4:
            # Set this as the data
            data = hist_data
        else:
            # If the third axis has size > 1
            if data.shape[2] > 6:
                # Calculate the mean of the data
                # Extract the second number in forecast_range
                forecast_range_number = int(forecast_range.split("-")[1])

                # Calculate the mean of the data
                data = data[:, :, : forecast_range_number - 1, :, :].mean(axis=2)
            elif data.shape[2] == 1:
                # Squeeze the data
                data = np.squeeze(data)
            elif data.shape[2] in [2, 3, 4, 5, 6]:
                # Take the year 2 index (year 2)
                data = data[:, :, 0, :, :]
            else:
                print("Data shape not recognised")
                AssertionError("Data shape not recognised")

            # If years 2-9
            if forecast_range == "2-9" and season not in ["DJFM", "DJF", "ONDJFM"]:
                # Remove the final time step
                data = data[:, :-1, :, :]

        print("Shape of the data:", np.shape(data))

        # Assert that the shape of lats is the same as the shape of the data third axis
        assert data.shape[2] == len(lats), "Data lats shape not equal to lats shape"

        # Assert that the shape of lons is the same as the shape of the data fourth axis
        assert data.shape[3] == len(lons), "Data lons shape not equal to lons shape"

        # Print the shape of the data
        print("Shape of the model data:", np.shape(data))

        # Print the shape of the obs_nao
        print("shape of the observed data:", np.shape(obs_nao))

        # Extract the data for the NAO gridboxes
        n_lat_box_model = data[
            :, :, n_lat1_idx : n_lat2_idx + 1, n_lon1_idx : n_lon2_idx + 1
        ].mean(axis=(2, 3))

        # Extract the data for the NAO gridboxes
        s_lat_box_model = data[
            :, :, s_lat1_idx : s_lat2_idx + 1, s_lon1_idx : s_lon2_idx + 1
        ].mean(axis=(2, 3))

        # Calculate the NAO index for the model
        # Azores - iceland
        model_nao = s_lat_box_model - n_lat_box_model

        # Print the shape of the model nao
        print("Shape of the model nao:", model_nao.shape)

        # Append the model nao to the dictionary
        nao_stats["model_nao_members"] = model_nao

        # Calculate the 5% lower interval
        model_nao_min = np.percentile(model_nao, 5, axis=0)

        # Calculate the 95% upper interval
        model_nao_max = np.percentile(model_nao, 95, axis=0)

        # Append the model_nao_min to the dictionary
        nao_stats["model_nao_members_min"] = model_nao_min

        # Append the model_nao_max to the dictionary
        nao_stats["model_nao_members_max"] = model_nao_max

        # Calculate the mean of the model nao
        nao_stats["model_nao_mean"] = model_nao.mean(axis=0)

        # Calculate the corr1
        corr1, p1 = pearsonr(obs_nao, model_nao.mean(axis=0))

        # Append the corr1 and p1 to the dictionary
        nao_stats["corr1"] = corr1
        nao_stats["p1"] = p1

        # calculate the standard deviation of the forecast
        sig_f1 = np.std(model_nao.mean(axis=0))

        # Calculate the rpc1
        rpc1 = corr1 / (sig_f1 / np.std(model_nao))

        # Calculate the rps1
        rps1 = rpc1 * (np.std(obs_nao) / np.std(model_nao))

        # Append the rpc1 to the dictionary
        nao_stats["rpc1"] = np.abs(rpc1)

        # Append the rps1 to the dictionary
        nao_stats["rps1"] = np.abs(rps1)

        # Append the nens to the dictionary
        nao_stats["nens"] = len(model_nao)

    elif data.ndim == 4 or len(hist_data[1]) == 51:
        print("Processing the alt-lag data")

        # Set up the alt_lag_first_year
        alt_lag_first_year = start_year + lag - 1

        # Set up the first and last years accordingly
        # If forecast range is 2-9
        if forecast_range == "2-9":
            # Set up the raw first and last years
            alt_lag_first_year = int(alt_lag_first_year) + 5
            alt_lag_last_year = int(end_year) + 5
        elif forecast_range == "2-5":
            # Set up the raw first and last years
            alt_lag_first_year = int(alt_lag_first_year) + 3
            alt_lag_last_year = int(end_year) + 3
        elif forecast_range == "2-3":
            # Set up the raw first and last years
            alt_lag_first_year = int(alt_lag_first_year) + 2
            alt_lag_last_year = int(end_year) + 2
        elif forecast_range == "2":
            # Set up the raw first and last years
            alt_lag_first_year = int(alt_lag_first_year) + 1
            alt_lag_last_year = int(end_year) + 1
        elif forecast_range == "1":
            # Set up the raw first and last years
            alt_lag_first_year = int(alt_lag_first_year)
            alt_lag_last_year = int(end_year)
        else:
            print("Forecast range not recognised")
            AssertionError("Forecast range not recognised")

        # # If the season is not DJFM
        # if season not in ["DJFM", "DJF", "ONDJFM"]:
        #     # Add 1 to the raw first and last years
        #     alt_lag_first_year += 1
        #     alt_lag_last_year += 1

        # Print the alt_lag_first year and last year
        print("Alt-lag first year:", alt_lag_first_year)
        print("Alt-lag last year:", alt_lag_last_year)

        # Set up the common years accordingly
        common_years = np.arange(alt_lag_first_year, alt_lag_last_year + 1)

        # set up the init_years
        nao_stats["init_years"] = np.arange(start_year + lag - 1, end_year + 1)

        # Append the valid years
        nao_stats["valid_years"] = common_years

        # Constrain the obs_psl_anom to the common years
        obs_psl_anom = obs_psl_anom.sel(
            time=slice(f"{alt_lag_first_year}-01-01", f"{alt_lag_last_year}-12-31")
        )

        # Print the first and last years of the obs_psl_anom
        print("First year of obs_psl_anom:", obs_psl_anom.time.dt.year.values[0])
        print("Last year of obs_psl_anom:", obs_psl_anom.time.dt.year.values[-1])

        # extract the data for the south grid
        obs_psl_anom_south = obs_psl_anom.sel(
            lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)
        ).mean(dim=["lat", "lon"])

        # extract the data for the north grid
        obs_psl_anom_north = obs_psl_anom.sel(
            lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)
        ).mean(dim=["lat", "lon"])

        # Calculate the NAO index: azores - iceland
        obs_nao = obs_psl_anom_south - obs_psl_anom_north

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

        # Extract obs_nao as its values
        obs_nao = obs_nao.values

        # append the obs_nao to the dictionary
        nao_stats["obs_nao"] = obs_nao

        # if nao_matched is false
        if not nao_matched:
            # Swap the axes of the data
            data = np.swapaxes(data, 0, 1)

        # Print the shape of the data
        print("Shape of the model data:", np.shape(data))
        print("Shape of the obs_nao:", np.shape(obs_nao))

        if hist_data.ndim == 4:
            # Set this as the data
            data = hist_data

        # Assert that the shape of lats is the same as the shape of the data third axis
        assert data.shape[2] == len(lats), "Data lats shape not equal to lats shape"

        # Assert that the shape of lons is the same as the shape of the data fourth axis
        assert data.shape[3] == len(lons), "Data lons shape not equal to lons shape"

        # Extract the data for the NAO gridboxes
        n_lat_box_model = data[
            :, :, n_lat1_idx : n_lat2_idx + 1, n_lon1_idx : n_lon2_idx + 1
        ].mean(axis=(2, 3))

        # Extract the data for the NAO gridboxes
        s_lat_box_model = data[
            :, :, s_lat1_idx : s_lat2_idx + 1, s_lon1_idx : s_lon2_idx + 1
        ].mean(axis=(2, 3))

        # Calculate the NAO index for the model
        # Azores - iceland
        model_nao = s_lat_box_model - n_lat_box_model

        # Print the shape of the model nao
        print("Shape of the model nao:", model_nao.shape)

        # Append the model nao to the dictionary
        nao_stats["model_nao_members"] = model_nao

        # Calculate the ensemble mean of the model nao
        model_nao_mean = model_nao.mean(axis=0)

        # Calculate the corr1
        corr1, p1 = pearsonr(obs_nao, model_nao_mean)

        # Calculate the standard deviation of the forecast
        sig_f1 = np.std(model_nao_mean)

        # Calculate the RPC1
        rpc1 = corr1 / (sig_f1 / np.std(model_nao))

        # Calculate the RPS1
        rps1 = rpc1 * (np.std(obs_nao) / np.std(model_nao))

        # Append the rpc1 and rps1 to the dictionary
        nao_stats["rpc1"] = np.abs(rpc1)
        nao_stats["rps1"] = np.abs(rps1)

        # Append the nens to the dictionary
        nao_stats["nens"] = len(model_nao)

        # Scale the ensemble mean by RPS1
        model_nao_mean = model_nao_mean * np.abs(rps1)

        # Append the model_nao_mean to the dictionary
        nao_stats["model_nao_mean"] = model_nao_mean

        # Calculate the corr2
        corr2, p2 = pearsonr(obs_nao, model_nao_mean)

        # Append the corr2 and p2 to the dictionary
        nao_stats["corr1"] = corr2
        nao_stats["p1"] = p2

        # Calculate the rmse
        rmse = np.sqrt(np.mean((model_nao_mean - obs_nao) ** 2))

        # Calculate the ci_lower and ci_upper using rmse
        ci_lower = model_nao_mean - rmse
        ci_upper = model_nao_mean + rmse

        # Append the ci_lower and ci_upper to the dictionary
        nao_stats["model_nao_members_min"] = ci_lower
        nao_stats["model_nao_members_max"] = ci_upper

    else:
        print("Data length not recognised")
        AssertionError("Data length not recognised")

    return nao_stats


# Write a function which creates a dataframe from the nao_stats
# and saves this
def create_nao_stats_df(
    nao_stats: dict,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    lag: int,
    alt_lag: bool = False,
    region: str = "global",
    variable: str = "psl",
    nao_type: str = "nao_default",
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/nao_stats_df",
):
    """
    Creates a dataframe from the NAO stats and saves this.

    Parameters:

    nao_stats: dict
        The dictionary containing the NAO stats.

    season: str
        The season for which the NAO stats are being saved.

    forecast_range: str
        The forecast range for which the NAO stats are being saved.

    start_year: int
        The start year for which the NAO stats are being saved.

    end_year: int
        The end year for which the NAO stats are being saved.

    lag: int
        The lag for which the NAO stats are being saved.

    alt_lag: bool
        Whether the alternate lag data is being saved (default is False).
        True = alt_lag data
        False = raw data

    region: str
        The region for which the NAO stats are being saved (either "global", "atlantic" or "pacific").
        Default is "global".

    variable: str
        The variable for which the NAO stats are being saved (default is "psl").

    nao_type: str
        The type of NAO index being saved (default is "nao_default").

    output_dir: str
        The directory in which the NAO stats dataframe is being saved (default is "/gws/nopw/j04/canari/users/benhutch/nao_stats_df").

    Returns:

    None
    """

    # If the output_dir does not exist
    if not os.path.exists(output_dir):
        # Make the output_dir
        os.makedirs(output_dir)

    print(
        f"""
    Length of init_years: {len(nao_stats['init_years'])}
    Length of valid_years: {len(nao_stats['valid_years'])}
    Length of obs_nao: {len(nao_stats['obs_nao'])}
    Length of model_nao_mean: {len(nao_stats['model_nao_mean'])}
    Length of model_nao_members_min: {len(nao_stats['model_nao_members_min'])}
    Length of model_nao_members_max: {len(nao_stats['model_nao_members_max'])}
    """
    )

    # Set up the dataframe
    nao_stats_df = pd.DataFrame(
        {
            "init_time": nao_stats["init_years"],
            "valid_time": nao_stats["valid_years"],
            "obs_nao": nao_stats["obs_nao"],
            "model_nao_mean": nao_stats["model_nao_mean"],
            "model_nao_members_min": nao_stats["model_nao_members_min"],
            "model_nao_members_max": nao_stats["model_nao_members_max"],
        }
    )

    # Set up the filename
    filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{nao_type}.csv"

    # Set up the file path
    file_path = os.path.join(output_dir, filename)

    # Save the dataframe
    nao_stats_df.to_csv(file_path, index=False)

    # Print that the dataframe has been saved
    print(f"Dataframe saved to {file_path}")

    # return the dataframe
    return nao_stats_df


def plot_nao(
    nao_stats: dict,
    season: str,
    forecast_range: str,
    lag: int,
    alt_lag: bool = False,
    figsize_x: int = 12,
    figsize_y: int = 8,
    ylim_min: int = -10,
    ylim_max: int = 10,
    title: str = None,
    ylabel: str = None,
    label: str = None,
    fontsize: int = 12,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
):
    """
    Plots the NAO index from nao_stats.

    Parameters:
    ===========

    nao_stats: dict
        The dictionary containing the NAO index.

    season: str
        The season for which the NAO index is being plotted.

    forecast_range: str
        The forecast range for which the NAO index is being plotted.

    lag: int
        The lag for which the NAO index is being plotted.

    alt_lag: bool
        Whether the alternate lag data is being plotted (default is False).
        True = alt_lag data
        False = raw data

    figsize_x: int
        The x-dimension of the figure (default is 12).

    figsize_y: int
        The y-dimension of the figure (default is 8).

    ylim_min: int
        The minimum value for the y-axis (default is -10).

    ylim_max: int
        The maximum value for the y-axis (default is 10).

    title: str
        The title of the plot (default is None).

    ylabel: str
        The y-label of the plot (default is "NAO index").

    label: str
        The label for the plot (e.g. a, b, c etc.)

    fontsize: int
        The fontsize of the text in the plot (default is 12).

    save_dir: str
        The directory in which the plots are being saved (default is "/gws/nopw/j04/canari/users/benhutch/plots").

    Returns:
    ========

    None
    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Plot the 5% lower interval
    ax.fill_between(
        nao_stats["init_years"],
        nao_stats["model_nao_members_min"] / 100,
        nao_stats["model_nao_members_max"] / 100,
        color="red",
        alpha=0.2,
    )

    # Plot the observed NAO index
    ax.plot(
        nao_stats["init_years"], nao_stats["obs_nao"] / 100, label="ERA5", color="black"
    )

    # Plot the ensemble mean NAO index
    ax.plot(
        nao_stats["init_years"],
        nao_stats["model_nao_mean"] / 100,
        label="DCPP",
        color="red",
    )

    # if "-" in forecast_range and season in ["DJFM", "DJF", "ONDJFM"]:
    #     # Set the y lim
    #     ax.set_ylim(-7.5, 7.5)
    # elif "-" not in forecast_range and season in ["DJFM", "DJF", "ONDJFM"]:
    #     # Set the y lim
    #     ax.set_ylim(-15, 15)
    # elif season not in ["DJFM", "DJF", "ONDJFM"]:
    #     # Set the y lim
    #     ax.set_ylim(-10, 10)

    # Set the ylmits
    ax.set_ylim(ylim_min, ylim_max)

    # Set up the horizontal line
    ax.axhline(y=0, color="black", linestyle="--")

    # # Include the legend in the lower bottom right corner
    # ax.legend(loc="lower right")

    # Set up the experiment
    if alt_lag:
        experiment = "Lagged"

        # # Format a textbox in the top left with the experiment and lag
        # ax.text(0.05, 0.95, f"{experiment} ({lag})",
        #         transform=ax.transAxes, fontsize=10,
        #         verticalalignment='top')

    else:
        experiment = "Raw"

        # # Format a textbox in the top left with the experiment and lag
        # ax.text(0.05, 0.95, f"{experiment}",
        #         transform=ax.transAxes, fontsize=10,
        #         verticalalignment='top',
        #         horizontalalignment='left')

    first_year = nao_stats["init_years"][0]
    last_year = nao_stats["init_years"][-1]

    # if alt_lag:
    #     ax.set_title(
    #         f"ACC = {nao_stats['corr1']:.2f} "
    #         f"(p = {nao_stats['p1']:.2f}), "
    #         f"RPC = {nao_stats['rpc1']:.2f}, "
    #         f"N = {nao_stats['nens']}, "
    #         f"{experiment} "
    #         f"({lag}), "
    #         f"{season}, "
    #         f"{forecast_range}, "
    #         f"{first_year}-{last_year}"
    #     )
    # else:
    #     # Set up the title
    #     ax.set_title(
    #         f"ACC = {nao_stats['corr1']:.2f} "
    #         f"(p = {nao_stats['p1']:.2f}), "
    #         f"RPC = {nao_stats['rpc1']:.2f}, "
    #         f"N = {nao_stats['nens']}, "
    #         f"{experiment}, "
    #         f"{season}, "
    #         f"{forecast_range}, "
    #         f"s{first_year}-s{last_year}"
    #     )

    # Inluce the correlation, p-value, RPC and N
    # In the top lef hand corner
    ax.text(
        0.05,
        0.95,
        (
            f"ACC = {nao_stats['corr1']:.2f} "
            f"(P = {nao_stats['p1']:.2f}), "
            f"RPC = {nao_stats['rpc1']:.2f}, "
            f"N = {nao_stats['nens']}"
        ),
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # if label is not none
    # insert a textbox in the bottom right hand corner
    if label is not None:
        ax.text(
            0.95,
            0.05,
            label,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=fontsize,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Set up another textbox in the top right with the season and forecast range
    # ax.text(0.95, 0.95, f"{season}\nYears {forecast_range}\n{nao_stats['init_years'][0]}-{nao_stats['init_years'][-1]}",
    #     transform=ax.transAxes, fontsize=10,
    #     verticalalignment='top',
    #     horizontalalignment='right')

    # Set up the x label
    ax.set_xlabel("Initialisation year")

    # if ylabel is not none
    if ylabel is not None:
        # Set the y label
        ax.set_ylabel(ylabel)

    # if title is not none
    if title is not None:
        # Set the title
        ax.set_title(title, fontweight="bold")

    # Set up the current time
    current_time = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S")

    # Set up the plot_name
    plot_name = (
        f"{experiment}_{season}_{forecast_range}_{lag}_nao_index_{current_time}.pdf"
    )

    # Save the plot
    plt.savefig(os.path.join(save_dir, plot_name), dpi=600)

    # Show the plot
    plt.show()


# Define a function to plot the NAO as subplots
def plot_nao_subplots(
    nao_stats_1: dict,
    nao_stats_2: dict,
    method_1: str,
    method_2: str,
    season: str,
    forecast_range: str,
    lag: int,
    figsize_x_px: int = 1800,
    figsize_y_px: int = 900,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
    format: str = "pdf",
    dpi: int = 600,
    fontsize: int = 10,
    save_dpi: int = 600,
    fig_labels: list = ["a", "b"],
):
    """
    Plots the NAO index from nao_stats as subplots.

    Parameters:
    ===========

    nao_stats_1: dict
        The dictionary containing the NAO index for the first method.

    nao_stats_2: dict
        The dictionary containing the NAO index for the second method.

    method_1: str
        The method for the first NAO index.

    method_2: str
        The method for the second NAO index.

    season: str
        The season for which the NAO index is being plotted.

    forecast_range: str
        The forecast range for which the NAO index is being plotted.

    lag: int
        The lag for which the NAO index is being plotted.

    figsize_x: int
        The x-dimension of the figure (default is 12).

    figsize_y: int
        The y-dimension of the figure (default is 8).

    save_dir: str
        The directory in which the plots are being saved (default is "/gws/nopw/j04/canari/users/benhutch/plots").

    format: str
        The format in which the plots are being saved (default is "pdf").

    dpi: int
        The dpi of the plots being saved (default is 600).

    fontsize: int
        The fontsize of the text in the plot (default is 10).

    save_dpi: int
        The dpi of the plots being saved (default is 600).

    fig_labels: list
        The labels for the subplots (default is ["a", "b"]).

    Returns:
    ========

    None
    """
    # print the dpi
    print("dpi:", dpi)

    # Set the rcParams figure dpi
    # to be 100.0
    plt.rcParams["figure.dpi"] = dpi

    # Set the px
    px = 1 / plt.rcParams["figure.dpi"]

    # print the px
    print("px:", px)

    # Calculate the figure size in inches
    figsize_x = figsize_x_px * px

    # Calculate the figure size in inches
    figsize_y = figsize_y_px * px

    # print the figsize_x and figsize_y
    print("figsize_x:", figsize_x)
    print("figsize_y:", figsize_y)

    # print the total size
    print("Total size:", figsize_x * figsize_y)

    # Set up the figure
    fig, ax = plt.subplots(
        figsize=(figsize_x_px * px, figsize_y_px * px), nrows=1, ncols=2, sharey=True
    )

    # Plot the 5% lower interval
    ax[0].fill_between(
        nao_stats_1["init_years"],
        nao_stats_1["model_nao_members_min"] / 100,
        nao_stats_1["model_nao_members_max"] / 100,
        color="red",
        alpha=0.2,
    )

    # Set up the horizontal line
    ax[0].axhline(y=0, color="black", linestyle="-")

    # Plot the observed NAO index
    ax[0].plot(
        nao_stats_1["init_years"],
        nao_stats_1["obs_nao"] / 100,
        label="ERA5",
        color="black",
    )

    # Plot the ensemble mean NAO index
    ax[0].plot(
        nao_stats_1["init_years"],
        nao_stats_1["model_nao_mean"] / 100,
        label="dcppA-hindcast",
        color="red",
    )

    # Set the xlabel
    ax[0].set_xlabel("Start of 8-year period")

    # Include the legend in the bottom right corner
    ax[0].legend(loc="upper right")

    # Include the first fig label in the bottom right corner
    ax[0].text(
        0.95,
        0.05,
        f"{fig_labels[0]}",
        transform=ax[0].transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Include the method as a textbox in the lower left corner
    ax[0].text(
        0.05,
        0.05,
        f"{method_1}",
        transform=ax[0].transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Inluce the correlation, p-value, RPC and N
    # In the top lef hand corner
    ax[0].text(
        0.05,
        0.95,
        (
            f"ACC = {nao_stats_1['corr1']:.2f} "
            f"(P = {nao_stats_1['p1']:.2f}), "
            f"RPC = {nao_stats_1['rpc1']:.2f}, "
            f"N = {nao_stats_1['nens']}"
        ),
        transform=ax[0].transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Set up the second subplot
    # Plot the 5% lower interval
    ax[1].fill_between(
        nao_stats_2["init_years"],
        nao_stats_2["model_nao_members_min"] / 100,
        nao_stats_2["model_nao_members_max"] / 100,
        color="red",
        alpha=0.2,
    )

    # Set up the horizontal line
    ax[1].axhline(y=0, color="black", linestyle="-")

    # Plot the observed NAO index
    ax[1].plot(
        nao_stats_2["init_years"],
        nao_stats_2["obs_nao"] / 100,
        label="ERA5",
        color="black",
    )

    # Plot the ensemble mean NAO index
    ax[1].plot(
        nao_stats_2["init_years"],
        nao_stats_2["model_nao_mean"] / 100,
        label="dcppA-hindcast",
        color="red",
    )

    # Set the xlabel
    ax[1].set_xlabel("Start of 8-year period")

    # Include the legend in the bottom right corner
    ax[1].legend(loc="upper right")

    # Include the second fig label in the bottom right corner
    ax[1].text(
        0.95,
        0.05,
        f"{fig_labels[1]}",
        transform=ax[1].transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Include the method as a textbox in the lower left corner
    ax[1].text(
        0.05,
        0.05,
        f"{method_2}",
        transform=ax[1].transAxes,
        fontsize=fontsize,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Inluce the correlation, p-value, RPC and N
    # In the top lef hand corner
    ax[1].text(
        0.05,
        0.95,
        (
            f"ACC = {nao_stats_2['corr1']:.2f} "
            f"(P = {nao_stats_2['p1']:.2f}), "
            f"RPC = {nao_stats_2['rpc1']:.2f}, "
            f"N = {nao_stats_2['nens']}"
        ),
        transform=ax[1].transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    for subplot in ax:
        if "-" in forecast_range and season in ["DJFM", "DJF"]:
            # Set the y lim
            subplot.set_ylim(-10, 10)
            # Set the y ticks
            subplot.set_yticks(np.linspace(-10, 10, num=7))
        elif "-" in forecast_range and season in ["ONDJFM"]:
            # Set the y lim
            subplot.set_ylim(-7.5, 7.5)
            # Set the y ticks
            subplot.set_yticks(np.linspace(-7.5, 7.5, num=7))
        elif "-" not in forecast_range and season in ["DJFM", "DJF", "ONDJFM"]:
            # Set the y lim
            subplot.set_ylim(-15, 15)
            # Set the y ticks
            subplot.set_yticks(np.linspace(-15, 15, num=7))
        elif season not in ["DJFM", "DJF", "ONDJFM"]:
            # Set the y lim
            subplot.set_ylim(-10, 10)
            # Set the y ticks
            subplot.set_yticks(np.linspace(-10, 10, num=7))
        else:
            raise ValueError("Season not recognised")

        # Your existing code
        subplot.set_xlim(1960, 2020)

        # Get the x-tick labels
        labels = subplot.get_xticks().tolist()

        # Replace the label for 2020 with an empty string
        labels[-1] = ""

        # Make sure these labels are int values
        labels = [int(label) for label in labels if label != ""]

        # Set the new labels
        subplot.set_xticklabels(labels)

        # Set the x ticks padding
        subplot.tick_params(axis="x", pad=8)

    # set up the y label
    ax[0].set_ylabel("NAO anomaly (hPa)")

    # # specify a tight layout
    plt.tight_layout()

    # Set up the current time
    current_time = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M")

    # Set up the plot_name
    plot_name = f"{method_1}_{method_2}_{season}_{forecast_range}_{lag}_nao_index_{current_time}.{format}"

    # Save the plot
    plt.savefig(os.path.join(save_dir, plot_name), dpi=save_dpi)

    # Show the plot
    plt.show()

    # Return none
    return None
