"""
Functions for use in paper1_plots.ipynb notesbook.
"""
# Local Imports
import os
import sys
import glob

# Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import pearsonr
import cartopy.feature as cfeature

# Local imports
sys.path.append("/home/users/benhutch/skill-maps")
import dictionaries as dicts

# # Import functions from skill-maps
sys.path.append("/home/users/benhutch/skill-maps/python")
# import functions as fnc
import plotting_functions as plt_fnc

# Import nao skill functions
import nao_skill_functions as nao_fnc

# Import functions
import functions as fnc

# Import functions from plot_init_benefit
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
import plot_init_benefit as pib_fnc

# Import the nao_matching_seasons functions
import nao_matching_seasons as nao_match_fnc

# Import the functions from process_bs_values
import process_bs_values as pbs_func


# Create a function to process the raw data for the full forecast period
# TODO: may also need to do this for lagged at some point as well
def forecast_stats_var(variables: list,
                       season: str,
                       forecast_range: str,
                       region: str = "global",
                       start_year: int = 1961,
                       end_year: int = 2023,
                       method: str = "raw",
                       no_bootstraps: int = 1,
                       base_dir: str = "/home/users/benhutch/skill-maps-processed-data"):
    """
    Wrapper function which processes and creates a forecast_stats dictionary
    object for each variable in the variables list.

    Inputs:
    -------

    variables: list
        List of variables to process.
        e.g. ["tas", "pr", "psl"]

    season: str
        Season to process.
        e.g. "DJF"

    forecast_range: str
        Forecast range to process.
        e.g. "2-9"

    region: str
        Region to process.
        e.g. default is "global"

    start_year: int
        Start year to process.
        e.g. default is 1961

    end_year: int
        End year to process.
        e.g. default is 2023

    method: str
        Method to process.
        e.g. default is "raw"

    no_bootstraps: int
        Number of bootstraps to process.
        e.g. default is 1

    base_dir: str
        Base directory to process.
        e.g. default is "/home/users/benhutch/skill-maps-processed-data"

    Outputs:
    --------
    forecast_stats_var: dict
        Dictionary containing forecast statistics for each variable.
    """

    # Create empty dictionary containing a key for each variable
    forecast_stats_var = {}

    # Translate the seasons
    if season == "DJF":
        model_season = "DJF"
    if season == "DJFM":
        model_season = "DJFM"
    elif season == "MAM":
        model_season = "MAY"
    elif season == "JJA":
        model_season = "ULG"
    elif season == "SON":
        model_season = "SON"
    else:
        raise ValueError(f"season {season} not recognised!")

    # Loop over each variable in the variables list
    for variable in variables:
        # Do some logging
        print(f"Processing {variable}...")

        # Assign the obs variable name
        obs_var_name = dicts.var_name_map[variable]

        # Print the obs variable name
        print(f"obs_var_name = {obs_var_name}")

        # Set up the dcpp models for the variable
        dcpp_models = nao_match_fnc.match_variable_models(match_var=variable)

        # Set up the obs path for the variable
        obs_path = nao_match_fnc.find_obs_path(match_var=variable)

        # Process the observations for this variable
        # Prrocess the observations
        obs = fnc.process_observations(variable=variable,
                                        region=region,
                                        region_grid=dicts.gridspec_global,
                                        forecast_range=forecast_range,
                                        season=season,
                                        observations_path=obs_path,
                                        obs_var_name=variable)
        
        # Load and process the dcpp model data
        dcpp_data = pbs_func.load_and_process_dcpp_data(base_dir=base_dir,
                                                        dcpp_models=dcpp_models,
                                                        variable=variable,
                                                        region=region,
                                                        forecast_range=forecast_range,
                                                        season=model_season)
        
        # Make sure that the individual models have the same valid years
        dcpp_data = fnc.constrain_years(model_data=dcpp_data,
                                        models=dcpp_models)
        
        # Align the obs and dcpp data
        obs, dcpp_data, _ = fnc.remove_years_with_nans_nao(observed_data=obs,
                                                           model_data=dcpp_data,
                                                           models=dcpp_models)
        
        # Extract an array from the obs
        obs_array = obs.values

        # Extract the dims from the obs data
        nyears = obs_array.shape[0]
        lats = obs_array.shape[1]
        lons = obs_array.shape[2]

        # Calculate the number of dcpp ensemble members
        dcpp_ensemble_members = np.sum([len(dcpp_data[model]) for model in dcpp_models])

        # Create an empty array to store the dcpp data
        dcpp_array = np.zeros([dcpp_ensemble_members, nyears, lats, lons])

        # Set up the member counter
        member_index = 0

        # Loop over the models
        for model in dcpp_models:
            dcpp_model_data = dcpp_data[model]

            # Loop over the ensemble members
            for member in dcpp_model_data:
                # Increment the member index
                member_index += 1

                # Extract the data
                data = member.values

                # If the data has four dimensions
                if len(data.shape) == 4:
                    # Squeeze the data
                    data = np.squeeze(data)

                # Assign the data to the forecast1 array
                dcpp_array[member_index-1, :, :, :] = data

        # Assert that obs and dcpp_array have the same shape
        assert obs_array.shape == dcpp_array[0, :, :, :].shape, "obs and dcpp_array have different shapes!"

        # Create an empty dictionary to store the forecast stats
        forecast_stats_var[variable] = {}

        # Calculate the forecast stats for the variable
        forecast_stats_var[variable] = fnc.forecast_stats(obs=obs_array,
                                                          forecast1=dcpp_array,
                                                          forecast2=dcpp_array, # use the same here as a placeholder for historical
                                                          no_boot=no_bootstraps)
        
        # Do some logging
        print(f"Finished processing {variable}!")

    # If psl is in the variables list
    if "psl" in variables:
        print("Calculating the NAO index...")

        # Set up the dcpp models for the variable
        dcpp_models = nao_match_fnc.match_variable_models(match_var="psl")

        # Set up the obs path for the variable
        obs_path = nao_match_fnc.find_obs_path(match_var="psl")

        # Process the observations for the NAO index
        obs_psl_anom = fnc.read_obs(variable="psl",
                                    region=region,
                                    forecast_range=forecast_range,
                                    season=season,
                                    observations_path=obs_path,
                                    start_year=1960,
                                    end_year=2023)
        
        # Load adn process the dcpp model data
        dcpp_data = pbs_func.load_and_process_dcpp_data(base_dir=base_dir,
                                                        dcpp_models=dcpp_models,
                                                        variable="psl",
                                                        region=region,
                                                        forecast_range=forecast_range,
                                                        season=model_season)

        # Remove the years with NaNs from the obs and dcpp data
        obs_psl_anom, \
        dcpp_data, \
        _ = fnc.remove_years_with_nans_nao(observed_data=obs_psl_anom,
                                            model_data=dcpp_data,
                                            models=dcpp_models,
                                            NAO_matched=False)
        
        # Extract the nao stats
        nao_stats_dict = nao_fnc.nao_stats(obs_psl=obs_psl_anom,
                                           hindcast_psl=dcpp_data,
                                           models_list=dcpp_models,
                                           lag=4,
                                           short_period=(1965,2010),
                                           season=season)

    else:
        print("Not calculating the NAO index...")
        nao_stats_dict = None

    # Return the forecast_stats_var dictionary
    return forecast_stats_var, nao_stats_dict

# Define a plotting function for this data
def plot_forecast_stats_var(forecast_stats_var_dic: dict,
                            nao_stats_dict: dict,
                            psl_models: list,
                            season: str,
                            forecast_range: str,
                            figsize_x: int = 10,
                            figsize_y: int = 12,
                            gridbox_corr: dict = None,
                            gridbox_plot: dict = None,
                            sig_threshold: float = 0.05):
    """
    Plots the correlation fields for each variable in the forecast_stats_var.
    
    Inputs:
    -------
    
    forecast_stats_var_dic: dict
        Dictionary containing forecast statistics for each variable.
        Dictionary keys are the variable names.
        e.g. forecast_stats_var["tas"] = {"corr": corr, "rmse": rmse, "bias": bias}

    nao_stats_dict: dict
        Dictionary containing the NAO stats.
        e.g. nao_stats_dict = {"corr": corr, "rmse": rmse, "bias": bias}

    psl_models: list
        List of models which are used to calculate the NAO index.

    season: str
        Season to process.

    forecast_range: str
        Forecast range to process.

    figsize_x: int
        Figure size in x direction.
        e.g. default is 10

    figsize_y: int
        Figure size in y direction.
        e.g. default is 12

    gridbox_corr: dict
        Dictionary containing the gridbox which is used to calculate the correlation.
        e.g. gridbox_corr = {"lat": lat, "lon": lon, "corr": corr}

    gridbox_plot: dict
        Dictionary containing the gridbox which is used to constrain
        the domain of the plot.
        e.g. gridbox_plot = {"lat": lat, "lon": lon, "corr": corr}

    sig_threshold: float
        Significance threshold for the correlation.
        e.g. default is 0.05

    Outputs:
    --------
    
    None

    """

    # First do the processing for the NAO index
    if forecast_range == "2-9" and season == "DJFM":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1966, 2019 + 1))
        nyears_long_lag = len(np.arange(1969, 2019 + 1))
    elif forecast_range == "2-9" and season == "JJA":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1967, 2019 + 1))
        nyears_long_lag = len(np.arange(1970, 2019 + 1))
    elif forecast_range == "2-5":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1964, 2017 + 1))
        nyears_long_lag = len(np.arange(1967, 2017 + 1))
    elif forecast_range == "2-3":
        # Set up the length of the time series for raw and lagged
        nyears_long = len(np.arange(1963, 2016 + 1))
        nyears_long_lag = len(np.arange(1966, 2016 + 1))
    else:
        raise ValueError("forecast_range must be either 2-9 or 2-5 or 2-3")

    # Set up the forecast ylims
    if season == "DJFM":
        ylims = [-10, 10]
    elif season == "JJA":
        ylims = [-3, 3]
    else:
        raise ValueError(f"season {season} not recognised!")

    # Set up the arrays for plotting the NAO index
    total_nens = 0 ; total_lagged_nens = 0

    # Loop over the models
    for model in psl_models:
        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Add the number of ensemble members to the total
        total_nens += nao_stats['nens']

        # And for the lagged ensemble
        total_lagged_nens += nao_stats['nens_lag']

    # Set up the arrays for the NAO index
    nao_members = np.zeros([total_nens, nyears_long])

    # Set up the lagged arrays for the NAO index
    nao_members_lag = np.zeros([total_lagged_nens, nyears_long_lag])

    # Set up the counter for the current index
    current_index = 0 ; current_index_lag = 0

    # Iterate over the models
    for i, model in enumerate(psl_models):
        print(f"Extracting ensemble members for model {model}...")

        # Extract the nao stats for this model
        nao_stats = nao_stats_dict[model]

        # Loop over the ensemble members
        for j in range(nao_stats['nens']):
            print(f"Extracting member {j} from model {model}...")

            # Extract the nao member
            nao_member = nao_stats['model_nao_ts_members'][j, :]

            # Set up the lnegth of the correct time series
            nyears_bcc = len(nao_stats_dict['BCC-CSM2-MR']['years'])

            # If the model is not BCC-CSM2-MR
            if model != "BCC-CSM2-MR":
                # Extract the len of the time series
                nyears = len(nao_stats['years'][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years'])                

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print("The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                    model))
                
                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears
                
                # Assert that the new len is correct
                assert len(nao_member[1:skip_years]) == nyears_bcc, "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members[current_index, :] = nao_member[1: skip_years]
                else:
                    nao_members[current_index, :] = nao_member[1:]
            else:
                # Append this member to the array
                nao_members[current_index, :] = nao_member

            # Increment the counter
            current_index += 1

        # Loop over the lagged ensemble members
        for j in range(nao_stats['nens_lag']):

            # Extract the NAO index for this member
            nao_member = nao_stats['model_nao_ts_lag_members'][j, :]
            print("NAO index extracted for member {}".format(j))

            # Set up the length of the correct time series
            nyears_bcc = len(nao_stats_dict['BCC-CSM2-MR']['years_lag'])

            if model != "BCC-CSM2-MR":
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years_lag'][1:])
            else:
                # Extract the length of the time series for this model
                nyears = len(nao_stats['years_lag'])

            # if these lens are not equal then we need to skip over the 0th time index
            if nyears != nyears_bcc:
                print("The length of the time series for {} is not equal to the length of the time series for BCC-CSM2-MR".format(
                    model))
                
                # Figure out how many years to skip over at the end
                skip_years = nyears_bcc - nyears
                
                # Assert that the new len is correct
                assert len(nao_member[1:skip_years]) == nyears_bcc, "Length of nao_member is not equal to nyears_bcc"
            else:
                skip_years = None

            # If the model is not BCC-CSM2-MR
            # then we need to skip over the 0th time index
            if model != "BCC-CSM2-MR":
                # If skip_years is not None
                if skip_years is not None:
                    # Append this member to the array
                    nao_members_lag[current_index_lag, :] = nao_member[1: skip_years]
                else:
                    nao_members_lag[current_index_lag, :] = nao_member[1:]
            else:
                # Append this member to the array
                nao_members_lag[current_index_lag, :] = nao_member

            # Increment the counter
            current_index_lag += 1

    # Set up the initialisation offset
    if forecast_range == "2-9":
        init_offset = 5
    elif forecast_range == "2-5":
        init_offset = 2
    elif forecast_range == "2-3":
        init_offset = 1
    else:
        raise ValueError("forecast_range must be either 2-9 or 2-5")


    # Set up the axis labels
    axis_labels = ["a", "b", "c", "d", "e", "f"]

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Count the number of keys in the forecast_stats_var_dic
    no_keys = len(forecast_stats_var_dic.keys())

    # Set up the nrows depending on whether the number of keys is even or odd
    if no_keys % 2 == 0:
        nrows = int(no_keys / 2) + 1 # Extra row for the NAO index
    else:
        nrows = int((no_keys + 1) / 2) + 1

    # Set up the figure
    fig, axs = plt.subplots(nrows=3,
                            ncols=2,
                            figsize=(figsize_x, figsize_y))
    
    # Update the params for mathtext default rcParams
    plt.rcParams.update({"mathtext.default": "regular"})

    # Set up the sup title
    sup_title = "Total correlation skill (r) for each variable"

    # Set up the sup title
    fig.suptitle(sup_title, fontsize=6, y=0.93)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5) ; lats = np.arange(-90, 90, 2.5)

    # If the gridbox_corr is not None
    if gridbox_corr is not None:
        # Extract the lats and lons from the gridbox_corr
        lon1_corr, lon2_corr = gridbox_corr['lon1'], gridbox_corr['lon2']
        lat1_corr, lat2_corr = gridbox_corr['lat1'], gridbox_corr['lat2']

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_corr = np.argmin(np.abs(lats - lat1_corr))
        lat2_idx_corr = np.argmin(np.abs(lats - lat2_corr))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_corr = np.argmin(np.abs(lons - lon1_corr))
        lon2_idx_corr = np.argmin(np.abs(lons - lon2_corr))

        # Constrain the lats and lons
        # lats_corr = lats[lat1_idx_corr:lat2_idx_corr]
        # lons_corr = lons[lon1_idx_corr:lon2_idx_corr]

    # Set up the extent
    # Using the gridbox plot here as this is the extent of the plot
    if gridbox_plot is not None:
        lon1_gb, lon2_gb = gridbox_plot['lon1'], gridbox_plot['lon2']
        lat1_gb, lat2_gb = gridbox_plot['lat1'], gridbox_plot['lat2']

        # find the indices of the lats which correspond to the gridbox
        lat1_idx_gb = np.argmin(np.abs(lats - lat1_gb))
        lat2_idx_gb = np.argmin(np.abs(lats - lat2_gb))

        # find the indices of the lons which correspond to the gridbox
        lon1_idx_gb = np.argmin(np.abs(lons - lon1_gb))
        lon2_idx_gb = np.argmin(np.abs(lons - lon2_gb))

        # Constrain the lats and lons
        lats = lats[lat1_idx_gb:lat2_idx_gb]
        lons = lons[lon1_idx_gb:lon2_idx_gb]

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up a list to store the contourf objects
    cf_list = []

    # Set up the axes
    axes = []

    # Loop over the keys and forecast_stats_var_dic
    for i, (key, forecast_stats) in enumerate(forecast_stats_var_dic.items()):
        # Logging
        print(f"Plotting variable {key}...") ; print(f"Plotting index {i}...")

        # Extract the correlation arrays from the forecast_stats dictionary
        corr = forecast_stats["corr1"]
        corr1_p = forecast_stats["corr1_p"]

        # Extract the time series
        fcst1_ts = forecast_stats["f1_ts"]
        obs_ts = forecast_stats["o_ts"]

        # Extract the values
        nens1 = forecast_stats["nens1"]
        start_year = 1966 - 5
        end_year = 2019 - 5

        # Print the start and end years
        print(f"start_year = {start_year}")
        print(f"end_year = {end_year}")
        print(f"nens1 = {nens1}")
        print(f"for variable {key}")

        # If grid_box_plot is not None
        if gridbox_plot is not None:
            # Print
            print("Constraining data to gridbox_plot...")
            print("As defined by gridbox_plot = ", gridbox_plot)

            # Constrain the data to the gridbox_plot
            corr = corr[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]
            corr1_p = corr1_p[lat1_idx_gb:lat2_idx_gb, lon1_idx_gb:lon2_idx_gb]

        # Set up the axes
        ax = plt.subplot(nrows, 2, i + 1, projection=proj)

        # Include coastlines
        ax.coastlines()

        # # Add borders (?)
        # ax.add_feature(cfeature.BORDERS)

        # Set up the cf object
        cf = ax.contourf(lons, lats, corr, clevs, transform=proj, cmap="RdBu_r")

        # If gridbox_corr is not None
        if gridbox_corr is not None:
            # Loggging
            print("Calculating the correlations with a specific gridbox...")
            print("As defined by gridbox_corr = ", gridbox_corr)

            # Constrain the ts to the gridbox_corr
            fcst1_ts = fcst1_ts[:, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]
            obs_ts = obs_ts[:, lat1_idx_corr:lat2_idx_corr, lon1_idx_corr:lon2_idx_corr]

            # Calculate the mean of both time series
            fcst1_ts_mean = np.mean(fcst1_ts, axis=(1, 2))
            obs_ts_mean = np.mean(obs_ts, axis=(1, 2))

            # Calculate the correlation between the two time series
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Show these values on the plot
            ax.text(0.05, 0.05, f"r = {r:.2f}, p = {p:.2f}", transform=ax.transAxes,
                    va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                    fontsize=8)
            
            # Add the gridbox to the plot
            ax.plot([lon1_corr, lon2_corr, lon2_corr, lon1_corr, lon1_corr],
                    [lat1_corr, lat1_corr, lat2_corr, lat2_corr, lat1_corr],
                    color="green", linewidth=2, transform=proj)
    
        # If any of the corr1 values are NaNs
        # then set the p values to NaNs at the same locations
        corr1_p[np.isnan(corr)] = np.nan

        # If any of the corr1_p values are greater than the sig_threshold
        # then set the corr1 values to NaNs at the same locations
        # WHat if the sig threshold is two sided?
        # We wan't to set the corr1_p values to NaNs
        # Where corr1_p<sig_threshold and corr1_p>1-sig_threshold
        corr1_p[(corr1_p > sig_threshold) & (corr1_p < 1-sig_threshold)] = np.nan

        # plot the p-values
        ax.contourf(lons, lats, corr1_p, hatches=["...."], alpha=0., transform=proj)

        # Add a text box with the axis label
        ax.text(0.95, 0.05, f"{axis_labels[i]}", transform=ax.transAxes,
                va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Add a textboc with the variable name in the top left
        ax.text(0.05, 0.95, f"{key}", transform=ax.transAxes,
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Include the number of ensemble members in the top right of the figure
        ax.text(0.95, 0.95, f"n = {nens1}", transform=ax.transAxes,
                va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=8)
        
        # Add a text box with the season in the top right
        # ax.text(0.95, 0.95, f"{season}", transform=ax.transAxes,
        #         va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
        #         fontsize=6)

        # Add the contourf object to the list
        cf_list.append(cf)

        # Add the axes to the list
        axes.append(ax)

    # Add a colorbar
    cbar = fig.colorbar(cf_list[0], ax=axes, orientation="horizontal", pad=0.05,
                        shrink=0.8)
    cbar.set_label("correlation coefficient", fontsize=10)

    # Now plot the NAO index
    print("Plotting the raw NAO index...")

    # Extract the total_nesn
    total_nens = nao_members.shape[0]

    # Calculate the mean of the nao_members
    nao_members_mean = np.mean(nao_members, axis=0)

    # Calculate the correlation between the nao_members_mean and the obs_ts
    corr1, p1 = pearsonr(nao_members_mean,
                            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts'])

    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1 = (corr1) / (np.std(nao_members_mean) /np.std(nao_members))

    # Calculate the 5th and 95th percentiles of the nao_members
    nao_members_mean_min = np.percentile(nao_members, 5, axis=0)
    nao_members_mean_max = np.percentile(nao_members, 95, axis=0)

    # Plot the ensemble mean
    ax1 = axs[2, 0]

    # Plot the ensemble mean
    ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
            nao_members_mean / 100, color="red", label="dcppA")

    # Plot the obs
    ax1.plot(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts'] / 100, color="black",
            label="ERA5")
    
    # Plot the 5th and 95th percentiles
    ax1.fill_between(nao_stats_dict['BCC-CSM2-MR']['years'] - init_offset,
                    nao_members_mean_min / 100,
                    nao_members_mean_max / 100,
                    color="red", alpha=0.2)

    # Add a text box with the axis label
    ax1.text(0.95, 0.05, f"{axis_labels[-2]}", transform=ax.transAxes,
            va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)
    
    # Include a textbox containing the total nens in the top right
    ax1.text(0.95, 0.95, f"n = {total_nens}", transform=ax.transAxes,
            va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)

    # Include a title which contains the correlations
    # p values and the rpccorr1_pbbbbeuibweiub
    ax1.set_title(f"ACC = {corr1:.2f} (p = {p1:.2f}), "
                 f"RPC = {rpc1:.2f}, "
                 f"N = {total_nens}", fontsize=10)

    # Include a legend
    ax1.legend(loc="upper left", fontsize=8)

    # Set a horizontal line at zero
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # Set the y limits
    ax1.set_ylim(ylims)

    # Set the y label
    ax1.set_ylabel("NAO (hPa)")

    # Set the x label
    ax1.set_xlabel("Initialisation year")

    # Print that we are plotting the lag and variance adjusted NAO index
    print("Plotting the lag and variance adjusted NAO index...")

    # Extract the total_nens
    total_nens_lag = nao_members_lag.shape[0]

    # Calculate the mean of the nao_members
    nao_members_mean_lag = np.mean(nao_members_lag, axis=0)

    # Calculate the correlation between the nao_members_mean and the obs_ts
    corr1_lag, p1_lag = pearsonr(nao_members_mean_lag,
                                    nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])
    
    # Calculate the RPC between the model NAO index and the obs NAO index
    rpc1_lag = (corr1_lag) / (np.std(nao_members_mean_lag) /np.std(nao_members_lag))

    # Calculate the rps between the model nao index and the obs nao index
    rps1 = rpc1_lag * (np.std(nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag']) / np.std(nao_members_lag))

    # Var adjust the NAO
    nao_var_adjust = nao_members_mean_lag * rps1

    # Calculate the RMSE between the ensemble mean and observations
    rmse = np.sqrt(np.mean((nao_var_adjust - nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'])**2))

    # Calculate the upper and lower bounds
    ci_lower = nao_var_adjust - rmse
    ci_upper = nao_var_adjust + rmse

    # Set up the axes
    ax2 = axs[2, 1]

    # Plot the ensemble mean
    ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
            nao_var_adjust / 100, color="red", label="dcppA")
    
    # Plot the obs
    ax2.plot(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
            nao_stats_dict['BCC-CSM2-MR']['obs_nao_ts_lag'] / 100, color="black",
            label="ERA5")

    # Plot the 5th and 95th percentiles
    ax2.fill_between(nao_stats_dict['BCC-CSM2-MR']['years_lag'] - init_offset,
                    ci_lower / 100,
                    ci_upper / 100,
                    color="red", alpha=0.2)

    # Add a text box with the axis label
    ax2.text(0.95, 0.05, f"{axis_labels[-1]}", transform=ax.transAxes,
            va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)
    
    # Include a textbox containing the total nens in the top right
    ax2.text(0.95, 0.95, f"n = {total_nens_lag}", transform=ax.transAxes,
            va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
            fontsize=8)

    # Include a title which contains the correlations
    # p values and rpc values
    ax2.set_title(f"ACC = {corr1_lag:.2f} (p = {p1_lag:.2f}), "
                 f"RPC = {rpc1_lag:.2f}, "
                 f"N = {total_nens_lag}", fontsize=10)
    
    # Set a horizontal line at zero
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # # Share the y axis with ax1
    # ax2.set_ylim([-10, 10])

    # Share the y-axis with ax1
    ax2.sharey(ax1)

    # Set up the pathname for saving the figure
    fig_name = f"different_variables_corr_{start_year}_{end_year}"

    # Set up the plots directory
    plots_dir = "/gws/nopw/j04/canari/users/benhutch/plots"

    # Set up the path to the figure
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # # Specify a tight layout
    # plt.tight_layout()

    # Show the figure
    plt.show()