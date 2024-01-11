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
                       end_year: int = 2018,
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
        e.g. default is 2018

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
                                                        season=season)
        
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

    # Return the forecast_stats_var dictionary
    return forecast_stats_var

# Define a plotting function for this data
def plot_forecast_stats_var(forecast_stats_var_dic: dict,
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

    # Set up the axis labels
    axis_labels = ["a", "b", "c", "d", "e", "f"]

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Count the number of keys in the forecast_stats_var_dic
    no_keys = len(forecast_stats_var_dic.keys())

    # Set up the nrows depending on whether the number of keys is even or odd
    if no_keys % 2 == 0:
        nrows = int(no_keys / 2)
    else:
        nrows = int((no_keys + 1) / 2)

    # Set up the figure
    fig, axs = plt.subplots(nrows=nrows,
                            ncols=2,
                            figsize=(figsize_x, figsize_y),
                            subplot_kw={"projection": proj},
                            gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    
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
        ax = axs.flat[i]

        # Include coastlines
        ax.coastlines()

        # Add borders (?)
        ax.add_feature(cfeature.BORDERS)

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
            ax.text(0.05, 0.05, f"r={r:.2f}, p={p:.2f}", transform=ax.transAxes,
                    va="bottom", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                    fontsize=6)
            
            # Add the gridbox to the plot
            ax.plot([lon1_corr, lon2_corr, lon2_corr, lon1_corr, lon1_corr],
                    [lat1_corr, lat1_corr, lat2_corr, lat2_corr, lat1_corr],
                    color="green", linewidth=2, transform=proj)
    
        # If any of the corr1 values are NaNs
        # then set the p values to NaNs at the same locations
        corr1_p[np.isnan(corr)] = np.nan

        # If any of the corr1_p values are greater than the sig_threshold
        # then set the corr1 values to NaNs at the same locations
        corr1_p[corr1_p > sig_threshold] = np.nan

        # plot the p-values
        ax.contourf(lons, lats, corr1_p, hatches=["...."], alpha=0., transform=proj)

        # Add a text box with the axis label
        ax.text(0.95, 0.05, f"{axis_labels[i]}", transform=ax.transAxes,
                va="bottom", ha="right", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=6)
        
        # Add a textboc with the variable name in the top left
        ax.text(0.05, 0.95, f"{key}", transform=ax.transAxes,
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.5),
                fontsize=6)
        
        # Add a text box with the season in the top right
        # ax.text(0.95, 0.95, f"{season}", transform=ax.transAxes,
        #         va="top", ha="right", bbox=dict(facecolor="white", alpha=0.5),
        #         fontsize=6)

        # Add the contourf object to the list
        cf_list.append(cf)

    # Add a colorbar
    cbar = fig.colorbar(cf_list[0], ax=axs, orientation="horizontal", pad=0.5,
                         aspect=50, shrink=0.8)
    cbar.set_label("correlation coefficient", fontsize=6)

    # Set up the pathname for saving the figure
    fig_name = f"different_variables_corr_{start_year}_{end_year}"

    # Set up the plots directory
    plots_dir = "/gws/nopw/j04/canari/users/benhutch/plots"

    # Set up the path to the figure
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()