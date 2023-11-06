"""
Functions to be used for creating plots from bootstrapped data.
"""
# Import general Python modules
import argparse, os, sys, glob, re

# Import additional modules
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import matplotlib as mpl

# # Use LaTeX for rendering
# mpl.rcParams['text.usetex'] = True

# Import functions from plot_init_benefit
# sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
# Modify this for working away from JASMIN
# Import functions from plot_init_benefit
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")
from plot_init_benefit import extract_values_from_txt, load_arrays_from_npy

# Define a function for plotting the values
def plot_raw_init_impact(corr1, corr1_p, init_impact, r_partial_p,
                        variable, season, forecast_range, method,
                        no_bootstraps, nens1, nens2, start_year, finish_year,
                        plots_dir):
    """
    Plots two subplots alongside (similar to Doug Smith's 2019 - Robust skill
    figure 3) each other. Left subplot is the correlation between the 
    initialized forecast and the obs, and the right subplot is the
    benefit of initialization relative to the uninitialized forecast. 

    The p-values are derived from the bootstrapped data.

    Args:
        corr1 (float): correlation between the initialized forecast and the
                        observations
        corr1_p (float): p-value for the correlation between the initialized
                            forecast and the observations
        init_impact (float): benefit of initialization relative to the
                                uninitialized forecast
        r_partial_p (float): p-value for the partial correlation between the
                                initialized forecast and the observations
                                after removing the influence of the
                                uninitialized forecast
        variable (str): variable to plot
        season (str): season to plot
        forecast_range (str): forecast range to plot
        method (str): method to plot
        no_bootstraps (int): number of bootstraps to plot
        nens1 (int): number of ensemble members in the first ensemble
                    (initialized)
        nens2 (int): number of ensemble members in the second ensemble
                    (uninitialized)
        start_year (int): start year of the forecast
        finish_year (int): end year of the forecast
        plots_dir (str): path to the directory to save the plots

    Returns:
        None
    """

    # Set up the axis labels
    ax_labels = ['A', 'B']

    # Set up the plot_names
    plot_names = ['Total skill', 'Residual Correlations']

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),
                            subplot_kw={'projection': proj}, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Set up the title
    title = 'Total skill and impact of initialization for ' + variable + ' in ' + season + ' for ' + forecast_range + 'between ' + str(start_year) + ' and ' + str(finish_year) + ' using ' + method + ' method' + 'no_bootstraps = ' + str(no_bootstraps)

    # set up the supertitle
    fig.suptitle(title, fontsize=10, y=0.90)

    # Set up the axes
    ax1 = axs[0]
    ax2 = axs[1]


    # set up the lons and lats
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # set up the contour levels for the raw correlation
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Plot the correlation between the initialized forecast 
    ax1.coastlines()

    # Plot the correlation between the initialized forecast 
    # and the observations
    cf1 = ax1.contourf(lons, lats, corr1, clevs, cmap='RdBu_r', transform=proj,
                        extend='both')

    # If any corr1 values are NaN
    if np.isnan(corr1).any():
        # Set the corr1_p value to NaN
        corr1_p[corr1 == np.nan] = np.nan

    # Set up the significance threshold as 0.05
    sig_threshold = 0.05

    # Set the corr1_p values to 1 if they are greater than the significance
    # threshold and 0 otherwise
    corr1_p[corr1_p > sig_threshold] = np.nan

    # Plot the p-values for the correlation between the initialized forecast
    # and the observations
    ax1.contourf(lons, lats, corr1_p, hatches=['....'],
                  alpha=0., transform=proj)
    
    # Add a textbox with the figure label
    ax1.text(0.95, 0.05, ax_labels[0], transform=ax1.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Add a textbox with the plot name
    ax1.text(0.05, 0.95, plot_names[0], transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5))

    # Plot the benefit of initialization relative to the uninitialized forecast
    ax2.coastlines()

    # Plot the benefit of initialization relative to the uninitialized forecast
    cf2 = ax2.contourf(lons, lats, init_impact, clevs, cmap='RdBu_r', 
                        transform=proj, extend='both')
    
    # If any r_partial_p values are NaN
    if np.isnan(init_impact).any():
        # Set the r_partial_p value to NaN
        r_partial_p[init_impact == np.nan] = np.nan

    # Set the corr1_p values to NaN if they are greater than the significance
    # threshold and 0 otherwise
    r_partial_p[r_partial_p > sig_threshold] = np.nan

    # Plot the p-values for the partial correlation between the initialized
    # forecast and the observations after removing the influence of the
    # uninitialized forecast
    ax2.contourf(lons, lats, r_partial_p, hatches=['....'], alpha=0.,
                 transform=proj)
    
    # Add a textbox with the figure label
    ax2.text(0.95, 0.05, ax_labels[1], transform=ax2.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Add a textbox with the plot name
    ax2.text(0.05, 0.95, plot_names[1], transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5))

    # Add a colorbar for the correlation
    cbar = fig.colorbar(cf1, ax=axs, orientation='horizontal', pad = 0.05,
                        aspect=50, shrink=0.8)
    cbar.set_label('correlation coefficient')

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/raw_init_impact_{variable}_{season}_" + \
f"{forecast_range}_{method}_{no_bootstraps}_{nens1}_{nens2}_{start_year}" + \
f"_{finish_year}.png"
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # show the figure
    plt.show()


# define a function which plots all of the seasons as subplots
# in a 4 x 2 grid of subplots
# with the total skill on the left 
# and the residual correlations on the right
# Rows are DJFM, MAM, JJA, SON, going downwards
def plot_raw_init_impact_subplots(arrays_list: list, values_list: list, variable: str, seasons_list: list,
                                  forecast_range: str, method: str, no_bootstraps: int,
                                  plots_dir: str) -> None:
    """
    Plots two subplots alongside (similar to Doug Smith's 2019 - Robust skill
    figure 3) each other. Left subplot is the correlation between the
    initialized forecast and the obs, and the right subplot is the residual
    correlation between the initialized forecast and the observations after
    removing the influence of the uninitialized forecast. 

    Subplots are for the seasons in the seasons_list.

    Args:
        arrays_list (list): list of dictionaries containing the arrays to plot.
                            Indexed by season.
                            Each dictionary contains the following keys:
                                corr1 (array): correlation between the initialized forecast and the observations
                                corr1_p (array): p-value for the correlation between the initialized forecast and the observations
                                partial_r (array): partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
                                partial_r_p (array): p-value for the partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
        values_list (list): list of dictionaries containing the values to plot.
                            Indexed by season.
                            Each dictionary contains the following keys:
                                nens1 (int): number of ensemble members in the first ensemble (initialized)
                                nens2 (int): number of ensemble members in the second ensemble (uninitialized)
                                start_year (int): start year of the forecast
                                end_year (int): end year of the forecast
        variable (str): variable to plot
        seasons_list (list): list of seasons to plot
        forecast_range (str): forecast range to plot
        method (str): method to plot
        no_bootstraps (int): number of bootstraps to plot
        plots_dir (str): path to the directory to save the plots

    Returns:
        None
    """

    # Set up the axis labels
    ax_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Set up the plot_names
    plot_names = ['total skill', 'residual corr']

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Set up the figure
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 12),
                            subplot_kw={'projection': proj}, 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Extract a start year and finish year from the values_list
    start_year = values_list[0]['start_year']
    finish_year = values_list[0]['end_year']
    
    # Set up the title
    title = 'Total skill and impact of initialization for ' + variable + ' for ' + forecast_range + \
        ' between ' + str(start_year) + ' and ' + str(finish_year) + ' using ' + method + \
        ' method' + 'no_bootstraps = ' + str(no_bootstraps)
    
    # set up the supertitle
    fig.suptitle(title, fontsize=8, y=0.90)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up the significance threshold as 0.05
    sig_threshold = 0.05

    # Create a list for the contourf objects
    cf_list = []

    # Loop over the seasons
    for i, season in enumerate(seasons_list):
        print("Plotting index:", i, "season:", season)

        # Extract the dictionaries for this season
        season_arrays = arrays_list[i]
        season_values = values_list[i]

        # From the dictionaries, extract the arrays
        corr1 = season_arrays['corr1']
        corr1_p = season_arrays['corr1_p']
        partial_r = season_arrays['partial_r']
        partial_r_p = season_arrays['partial_r_p']

        # From the dictionaries, extract the values
        nens1 = season_values['nens1']
        nens2 = season_values['nens2']
        start_year = season_values['start_year']
        end_year = season_values['end_year']

        # Set up the axes for the total skill
        ax1 = axs[i, 0]
        ax1.coastlines()
        cf = ax1.contourf(lons, lats, corr1, clevs, cmap='RdBu_r', transform=proj)
        
        # if any of the corr1 values are NaN
        if np.isnan(corr1).any():
            # set the corr1_p value to NaN at those points
            corr1_p[corr1 == np.nan] = np.nan

        # If any of the corr1_p values are greater than the significance
        # threshold - set them to NaN
        corr1_p[corr1_p > sig_threshold] = np.nan

        # Plot the p-values for the correlation between the initialized forecast
        # and the observations
        ax1.contourf(lons, lats, corr1_p, hatches=['....'], alpha=0.,
                        transform=proj)
        
        # Add a textbox with the figure label
        ax1.text(0.95, 0.05, ax_labels[2 * i], transform=ax1.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Add a textbox with the plot name
        ax1.text(0.05, 0.95, plot_names[0], transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Add a textbox with the method
        # to the top right of the plot
        ax1.text(0.95, 0.95, method, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # add a textbox with the season
        # to the bottom left of the plot
        ax1.text(0.05, 0.05, season, transform=ax1.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Add the contourf object to the list
        cf_list.append(cf)

        # Set up the axes for the residual correlation
        ax2 = axs[i, 1]
        ax2.coastlines()
        cf = ax2.contourf(lons, lats, partial_r, clevs, cmap='RdBu_r', 
                            transform=proj)

        # Append the contourf object to the list
        cf_list.append(cf)

        # If any of the partial_r values are NaN
        if np.isnan(partial_r).any():
            # Set the partial_r_p values to NaN at those points
            partial_r_p[partial_r == np.nan] = np.nan

        # If any of the partial_r_p values are greater than the significance
        # threshold - set them to NaN
        partial_r_p[partial_r_p > sig_threshold] = np.nan

        # Plot the p-values for the partial correlation between the initialized
        # forecast and the observations after removing the influence of the
        # uninitialized forecast
        ax2.contourf(lons, lats, partial_r_p, hatches=['....'], alpha=0.,
                        transform=proj)
        
        # Add a textbox with the figure label
        ax2.text(0.95, 0.05, ax_labels[(2*i)+1], transform=ax2.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Add a textbox with the plot name
        ax2.text(0.05, 0.95, plot_names[1], transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # Add a textbox with the method
        # to the top right of the plot
        ax2.text(0.95, 0.95, method, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))
        
        # add a textbox with the season
        # to the bottom left of the plot
        ax2.text(0.05, 0.05, season, transform=ax2.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))

    # Add a colorbar for the correlation
    cbar = fig.colorbar(cf_list[0], ax=axs, orientation='horizontal', pad = 0.05,
                        aspect=50, shrink=0.8)
    cbar.set_label('correlation coefficient')

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/raw_init_impact_{variable}_subplots_" + \
    f"{forecast_range}_{method}_{no_bootstraps}_{start_year}" + \
    f"_{finish_year}.png"
    
    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # show the figure
    plt.show()

# Create a function which plots a matrix of subplots
# with three rows and two columns
# The first column is the total skill
# the second column is the residual correlation
# The rows are for raw, lagged and nao-matched data
def plot_different_methods_same_season_var(arrays: list, values: list, 
                                            variable: str, season: str,
                                            forecast_range: str, method_list: list,
                                            no_bootstraps: int, plots_dir: str,
                                            gridbox: dict = None,
                                            figsize_x: int = 10, figsize_y: int = 12,
                                            plot_gridbox: dict = None,
                                            ts_arrays: list = None,
                                            region_name: str = None,
                                            seasons_list: list = None,
                                            plot_corr_diff: bool = False) -> None:
    """
    Plots a 3 x 2 matrix of subplots. The first column is the total correlation
    skill and the second column is the residual correlation. The rows are for
    raw, lagged and nao-matched data, as defined in methods list.
    
    Args:

        arrays (list): list of dicts containing the arrays to plot.
                        Indexed by the methods in method_list.
                        Each dictionary contains the following keys:
                            corr1 (array): correlation between the initialized forecast and the observations
                            corr1_p (array): p-value for the correlation between the initialized forecast and the observations
                            partial_r (array): partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
                            partial_r_p (array): p-value for the partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
                            corr_diff (array): difference between the initialized forecast and the uninitialized forecast
                            corr_diff_p (array): p-value for the difference between the initialized forecast and the uninitialized forecast

        values (list): list of dicts containing the values to plot.
                        Indexed by the methods in method_list.
                        Each dictionary contains the following keys:
                            nens1 (int): number of ensemble members in the first ensemble (initialized)
                            nens2 (int): number of ensemble members in the second ensemble (uninitialized)
                            start_year (int): start year of the forecast
                            end_year (int): end year of the forecast

        variable (str): variable to plot

        season (str): season to plot

        forecast_range (str): forecast range to plot

        method_list (list): list of methods to plot (e.g. raw, lagged, nao_matched)

        no_bootstraps (int): number of bootstraps to plot

        plots_dir (str): path to the directory to save the plots

        gridbox (dict): dictionary containing the gridbox to plot. Default is None.
                        Contains constrained gridbox with dimensions as follows:
                            'lon1': lower longitude bound
                            'lon2': upper longitude bound
                            'lat1': lower latitude bound
                            'lat2': upper latitude bound

        figsize_x (int): size of the figure in the x direction. Default is 10.

        figsize_y (int): size of the figure in the y direction. Default is 12.

        plot_gridbox (dict): dictionary containing the gridbox to constrain the
                            plot to. Default is None.
                            Contains constrained gridbox with dimensions as follows:
                                'lon1': lower longitude bound
                                'lon2': upper longitude bound
                                'lat1': lower latitude bound
                                'lat2': upper latitude bound

        ts_arrays (list): list of dicts containing the time series arrays to plot.
                            Indexed by the methods in method_list.
                            Each dictionary contains the following keys:
                            corr1 (array): correlation between the initialized forecast and the observations
                            corr1_p (array): p-value for the correlation between the initialized forecast and the observations
                            partial_r (array): partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
                            partial_r_p (array): p-value for the partial correlation between the initialized forecast and the observations after removing the influence of the uninitialized forecast
                            corr_diff (array): difference between the initialized forecast and the uninitialized forecast
                            corr_diff_p (array): p-value for the difference between the initialized forecast and the uninitialized forecast
                            fcst1_ts (array): time series of the initialized forecast
                            fcst2_ts (array): time series of the uninitialized forecast
                            obs_ts (array): time series of the observations
                            fcst1_em_resid (array): residuals of the initialized forecast
                            obs_resid (array): residuals of the observations

        region_name (str): name of the region to plot. Default is None.

        seasons_list (list): list of seasons to plot. Default is None.

        plot_corr_diff (bool): whether to plot the difference between the
                                initialized forecast and the uninitialized
                                forecast. Default is False. If set to true, 
                                plots three columns: total skill, residual
                                correlation and difference between the
                                initialized forecast and the uninitialized
                                forecast fields.

    Returns:

        None

    """

    # Set up the axis labels
    ax_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    # Set up the plot_names
    plot_names = ['total skill', 'residual corr', 'corr diff']

    # Set up the projection
    proj = ccrs.PlateCarree()

    # If seasons_list is not None
    if seasons_list is not None:
        # Set up the list
        row_list = seasons_list
        season = "DJFM-SON"
    else:
        # Set up the list
        row_list = method_list

    # Depending on the number of methods in the method_list
    # Set up the number of rows
    if len(row_list) == 2:
        nrows = 2
    elif len(row_list) == 3:
        nrows = 3
    elif len(row_list) == 1:
        nrows = 1
    elif len(row_list) == 4:
        nrows = 4
    else:
        print("Number of methods not supported")
        sys.exit()


    # If the plot_corr_diff is True
    if plot_corr_diff:
        # Set up the number of columns
        ncols = 3
    else:
        # Set up the number of columns
        ncols = 2

    # Set up the figure size
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_x, figsize_y),
                            subplot_kw={'projection': proj}, 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Update the params for mathtext default rcParams
    plt.rcParams.update({'mathtext.default': 'regular'})
    
    # Extract a start year and finish year from the values_list
    start_year = values[0]['start_year']
    finish_year = values[0]['end_year']

    # Set up the title
    title = 'Total skill and impact of initialization for ' + variable + ' in ' + season + ' for ' + forecast_range + \
        ' between ' + str(start_year) + ' and ' + str(finish_year) + \
        ' using different methods' + 'no_bootstraps = ' + str(no_bootstraps)

    # set up the supertitle
    fig.suptitle(title, fontsize=4, y=0.92)

    # If the gridbox is not None
    if gridbox is not None and 'south' not in gridbox and 'north' not in gridbox:
        # Set up the lats and lons for this gridbox
        lon1, lon2 = gridbox['lon1'], gridbox['lon2']
        lat1, lat2 = gridbox['lat1'], gridbox['lat2']
    elif 'south' and 'north' in gridbox:
        # Extract the south and north nao skill gridboxes
        south_gridbox = gridbox['south']
        north_gridbox = gridbox['north']

        # Set up the lats and lons for the south gridbox
        lon1_s, lon2_s = south_gridbox['lon1'], south_gridbox['lon2']
        lat1_s, lat2_s = south_gridbox['lat1'], south_gridbox['lat2']

        # Set up the lats and lons for the north gridbox
        lon1_n, lon2_n = north_gridbox['lon1'], north_gridbox['lon2']
        lat1_n, lat2_n = north_gridbox['lat1'], north_gridbox['lat2']

    # If the plot_gridbox is not None
    if plot_gridbox is not None:
        # Set up the lats and lons for this gridbox
        plot_lon1, plot_lon2 = plot_gridbox['lon1'], plot_gridbox['lon2']
        plot_lat1, plot_lat2 = plot_gridbox['lat1'], plot_gridbox['lat2']

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Set up the contour levels
    clevs = np.arange(-1.0, 1.1, 0.1)

    # Set up the significance threshold as 0.05
    sig_threshold = 0.05

    # Create a list for the contourf objects
    cf_list = []

    # Loop over the methods
    for i, method in enumerate(row_list):
        # print("plotting index: ", i, " for method: ", method)

        # Extract the dictionaries for this method
        method_arrays = arrays[i]
        method_arrays_ts = ts_arrays[i]
        method_values = values[i]

        # From the dictionaries, extract the arrays
        corr1 = method_arrays['corr1']
        corr1_p = method_arrays['corr1_p']
        partial_r = method_arrays['partial_r']
        partial_r_p = method_arrays['partial_r_p']

        # Extract the arrays for the corr_diff
        corr_diff = method_arrays['corr_diff']
        corr_diff_p = method_arrays['corr_diff_p']

        # TODO: Once cylc suite has processed for 1 bootstrap case
        # Extract the time series arrays to calculate correlations
        fcst1_ts = method_arrays_ts['fcst1_ts']
        obs_ts = method_arrays_ts['obs_ts']
        fcst1_em_residual = method_arrays_ts['fcst1_em_resid']
        obs_resid = method_arrays_ts['obs_resid']

        # From the dictionaries, extract the values
        nens1 = method_values['nens1']
        nens2 = method_values['nens2']
        start_year = method_values['start_year']
        end_year = method_values['end_year']

        # If the plot_gridbox is not None
        if plot_gridbox is not None:
            # Find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats - plot_lat1))
            lat2_idx = np.argmin(np.abs(lats - plot_lat2))

            # Find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons - plot_lon1))
            lon2_idx = np.argmin(np.abs(lons - plot_lon2))

            # Constrain the lats and lon arrays to the gridbox
            lats_cs = lats[lat1_idx:lat2_idx]
            lons_cs = lons[lon1_idx:lon2_idx]

            # Constrain the corr1 array to the gridbox
            corr1_cs = corr1[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the corr1_p array to the gridbox
            corr1_p_cs = corr1_p[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the partial_r array to the gridbox
            partial_r_cs = partial_r[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the partial_r_p array to the gridbox
            partial_r_p_cs = partial_r_p[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the corr_diff array to the gridbox
            corr_diff_cs = corr_diff[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the corr_diff_p array to the gridbox
            corr_diff_p_cs = corr_diff_p[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constraint the fcst1_ts array to the gridbox
            fcst1_ts_cs = fcst1_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constraint the obs_ts array to the gridbox
            obs_ts_cs = obs_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constraint the fcst1_em_residual array to the gridbox
            fcst1_em_residual_cs = fcst1_em_residual[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constraint the obs_resid array to the gridbox
            obs_resid_cs = obs_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

        # Set up the axes for the total skill
        ax1 = axs[i, 0]
        ax1.coastlines()

        # If the region_grid is China, add borders
        if region_name == 'china':
            ax1.add_feature(cfeature.BORDERS, linestyle=':')

        if plot_gridbox is not None:
            cf = ax1.contourf(lons_cs, lats_cs, corr1_cs, clevs, cmap='RdBu_r', transform=proj)
        else:
            cf = ax1.contourf(lons, lats, corr1, clevs, cmap='RdBu_r', transform=proj)

        # If the gridbox is not None
        if gridbox is not None and 'south' and 'north' not in gridbox and plot_gridbox is None:
            print("Global plot, gridbox specified, but doesn't contain south and north")
            # Add green lines outlining the gridbox
            ax1.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
                    color='green', linewidth=2, transform=proj)

            # Constrain the corr1 array to the gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats - lat1))
            lat2_idx = np.argmin(np.abs(lats - lat2))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons - lon1))
            lon2_idx = np.argmin(np.abs(lons - lon2))

            # # Constrain both fcst1_ts and obs_ts to the gridbox
            fcst1_ts_gridbox = fcst1_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
            obs_ts_gridbox = obs_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean = np.nanmean(fcst1_ts_gridbox, axis=(1, 2))
            obs_ts_mean = np.nanmean(obs_ts_gridbox, axis=(1, 2))

            # Calculate the correlation between the two
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Show these values e.g. r = 0.50, p = 0.01 in the lower left textbox
            ax1.text(0.05, 0.05, f"r = {r:.2f}, p = {p:.2f}", transform=ax1.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
            
        elif 'south' and 'north' in gridbox and plot_gridbox is None:
            print("Global plot, gridbox specified which contains south and north")
            # Add green lines outlining the north gridbox
            ax1.plot([lon1_n, lon2_n, lon2_n, lon1_n, lon1_n], [lat1_n, lat1_n, lat2_n, lat2_n, lat1_n],
                    color='green', linewidth=2, transform=proj)

            # Add green lines outlining the south gridbox
            ax1.plot([lon1_s, lon2_s, lon2_s, lon1_s, lon1_s], [lat1_s, lat1_s, lat2_s, lat2_s, lat1_s],
                    color='green', linewidth=2, transform=proj)       

            # Constrain the corr1 array to the north gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_n = np.argmin(np.abs(lats - lat1_n))
            lat2_idx_n = np.argmin(np.abs(lats - lat2_n))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_n = np.argmin(np.abs(lons - lon1_n))
            lon2_idx_n = np.argmin(np.abs(lons - lon2_n))

            # Find the indices of the lats which correspond to the south gridbox
            lat1_idx_s = np.argmin(np.abs(lats - lat1_s))
            lat2_idx_s = np.argmin(np.abs(lats - lat2_s))

            # find the indices of the lons which correspond to the south gridbox
            lon1_idx_s = np.argmin(np.abs(lons - lon1_s))
            lon2_idx_s = np.argmin(np.abs(lons - lon2_s))

            # Constrain both fcst1_ts and obs_ts to the north gridbox
            fcst1_ts_gridbox_n = fcst1_ts[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]
            obs_ts_gridbox_n = obs_ts[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            # Constrain both fcst1_ts and obs_ts to the south gridbox
            fcst1_ts_gridbox_s = fcst1_ts[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]
            obs_ts_gridbox_s = obs_ts[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean_n = np.nanmean(fcst1_ts_gridbox_n, axis=(1, 2))
            obs_ts_mean_n = np.nanmean(obs_ts_gridbox_n, axis=(1, 2))

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean_s = np.nanmean(fcst1_ts_gridbox_s, axis=(1, 2))
            obs_ts_mean_s = np.nanmean(obs_ts_gridbox_s, axis=(1, 2))

            # Calculate the correlation between the two
            r_n, p_n = pearsonr(fcst1_ts_mean_n, obs_ts_mean_n)
            r_s, p_s = pearsonr(fcst1_ts_mean_s, obs_ts_mean_s)

            # Show these values e.g. r_{n} = 0.50, p_{n} = 0.01 in the lower left textbox
            # and r_{s} = 0.50, p_{s} = 0.01 in the same textbox, but on the next line
            ax1.text(0.05, 0.05, 
                    f"r$_{{n}}$ = {r_n:.2f}, p$_{{n}}$ = {p_n:.2f}\n" + \
                    f"r$_{{s}}$ = {r_s:.2f}, p$_{{s}}$ = {p_s:.2f}",
                    transform=ax1.transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5),
                    fontsize=10)
        elif gridbox is not None and 'south' not in gridbox and 'north' not in gridbox and plot_gridbox is not None:
            print("Plot constrained to plot_gridbox, gridbox specified, but doesn't contain south and north")
            # Add green lines outlining the gridbox
            ax1.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
                    color='green', linewidth=2, transform=proj)

            # Constrain the corr1 array to the gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats_cs - lat1))
            lat2_idx = np.argmin(np.abs(lats_cs - lat2))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons_cs - lon1))
            lon2_idx = np.argmin(np.abs(lons_cs - lon2))

            # Constrain both fcst1_ts and obs_ts to the gridbox
            fcst1_ts_gridbox = fcst1_ts_cs[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
            obs_ts_gridbox = obs_ts_cs[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean = np.nanmean(fcst1_ts_gridbox, axis=(1, 2))
            obs_ts_mean = np.nanmean(obs_ts_gridbox, axis=(1, 2))

            # Calculate the correlation between the two
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Show these values e.g. r = 0.50, p = 0.01 in the lower left textbox
            ax1.text(0.05, 0.05, f"r = {r:.2f}, p = {p:.2f}", transform=ax1.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',   
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        elif 'south'and 'north' in gridbox and plot_gridbox is not None:
            print("Plot constrained to plot_gridbox, gridbox specified which contains south and north")
            # Add green lines outlining the north gridbox
            ax1.plot([lon1_n, lon2_n, lon2_n, lon1_n, lon1_n], [lat1_n, lat1_n, lat2_n, lat2_n, lat1_n],
                    color='green', linewidth=2, transform=proj)
            
            # Add green lines outlining the south gridbox
            ax1.plot([lon1_s, lon2_s, lon2_s, lon1_s, lon1_s], [lat1_s, lat1_s, lat2_s, lat2_s, lat1_s],
                    color='green', linewidth=2, transform=proj)
            
            # Constrain the corr1 array to the north gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_n = np.argmin(np.abs(lats_cs - lat1_n))
            lat2_idx_n = np.argmin(np.abs(lats_cs - lat2_n))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_n = np.argmin(np.abs(lons_cs - lon1_n))
            lon2_idx_n = np.argmin(np.abs(lons_cs - lon2_n))

            # Constrain the corr1 array to the south gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_s = np.argmin(np.abs(lats_cs - lat1_s))
            lat2_idx_s = np.argmin(np.abs(lats_cs - lat2_s))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_s = np.argmin(np.abs(lons_cs - lon1_s))
            lon2_idx_s = np.argmin(np.abs(lons_cs - lon2_s))

            # Constrain both fcst1_ts and obs_ts to the north gridbox
            fcst1_ts_gridbox_n = fcst1_ts_cs[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]
            obs_ts_gridbox_n = obs_ts_cs[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            # Constrain both fcst1_ts and obs_ts to the south gridbox
            fcst1_ts_gridbox_s = fcst1_ts_cs[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]
            obs_ts_gridbox_s = obs_ts_cs[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean_n = np.nanmean(fcst1_ts_gridbox_n, axis=(1, 2))
            obs_ts_mean_n = np.nanmean(obs_ts_gridbox_n, axis=(1, 2))

            # Calculate the gridbox mean of both fcst1_ts and obs_ts
            fcst1_ts_mean_s = np.nanmean(fcst1_ts_gridbox_s, axis=(1, 2))
            obs_ts_mean_s = np.nanmean(obs_ts_gridbox_s, axis=(1, 2))

            # Calculate the correlation between the two
            r_n, p_n = pearsonr(fcst1_ts_mean_n, obs_ts_mean_n)

            # Calculate the correlation between the two
            r_s, p_s = pearsonr(fcst1_ts_mean_s, obs_ts_mean_s)

            # Show these values e.g. r_{n} = 0.50, p_{n} = 0.01 in the lower left textbox
            # and r_{s} = 0.50, p_{s} = 0.01 in the same textbox, but on the next line
            # Only if plot_corr_diff is False
            if plot_corr_diff is False:
                ax1.text(0.05, 0.05, 
                        f"r$_{{n}}$ = {r_n:.2f}, p$_{{n}}$ = {p_n:.2f}\n" + \
                        f"r$_{{s}}$ = {r_s:.2f}, p$_{{s}}$ = {p_s:.2f}",
                        transform=ax1.transAxes,
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5),
                        fontsize=10)
        
        if plot_gridbox is not None:
            if np.isnan(corr1_cs).any():
                print("corr1_cs contains NaNs")
                # set the corr1_p value to NaN at those points
                corr1_p_cs[corr1_cs == np.nan] = np.nan

            # If any of the corr1_p values are greater than the significance
            # threshold - set them to NaN
            corr1_p_cs[corr1_p_cs > sig_threshold] = np.nan

            # Create a masked array for the corr1_p values
            corr1_p_cs_masked = np.ma.masked_where(np.isnan(corr1_p_cs), corr1_p_cs)

            # Plot the p-values for the correlation between the initialized forecast
            # and the observations
            ax1.contourf(lons_cs, lats_cs, corr1_p_cs_masked, hatches=['....'], alpha=0.,
                            transform=proj)
        else:
            # if any of the corr1 values are NaN
            if np.isnan(corr1).any():
                print("corr1 contains NaNs")
                # set the corr1_p value to NaN at those points
                corr1_p[corr1 == np.nan] = np.nan

            # If any of the corr1_p values are greater than the significance
            # threshold - set them to NaN
            corr1_p[corr1_p > sig_threshold] = np.nan

            # Plot the p-values for the correlation between the initialized forecast
            # and the observations
            ax1.contourf(lons, lats, corr1_p, hatches=['....'], alpha=0.,
                            transform=proj)

        # Add a textbox with the figure label
        ax1.text(0.95, 0.05, ax_labels[ncols * i], transform=ax1.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the plot name
        ax1.text(0.05, 0.95, plot_names[0], transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the method
        # to the top right of the plot
        ax1.text(0.95, 0.95, method, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # # add a textbox with the season
        # # to the bottom left of the plot
        # ax1.text(0.05, 0.05, season, transform=ax1.transAxes,
        #             verticalalignment='bottom', horizontalalignment='left',
        #             bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add the contourf object to the list
        cf_list.append(cf)

        # Set up the axes for the residual correlation
        ax2 = axs[i, 1]
        ax2.coastlines()

        # if the region_grid is China, add borders
        if region_name == 'china':
            ax2.add_feature(cfeature.BORDERS, linestyle=':')

        if plot_gridbox is not None:
            cf = ax2.contourf(lons_cs, lats_cs, partial_r_cs, clevs, cmap='RdBu_r',
                                transform=proj)
        else:
            cf = ax2.contourf(lons, lats, partial_r, clevs, cmap='RdBu_r', 
                                transform=proj)
        
        # if the gridbox is not None
        if gridbox is not None and 'south' and 'north' not in gridbox and plot_gridbox is None:
            print("Global plot, gridbox specified, but doesn't contain south and north")
            # Add green lines outlining the gridbox
            ax2.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
                    color='green', linewidth=2, transform=proj)
            
            # Constrain the partia_r array to the gridbox
            # find the indices of the lats which correspond to the gridbox
            # Constrain the corr1 array to the gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats - lat1))
            lat2_idx = np.argmin(np.abs(lats - lat2))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons - lon1))
            lon2_idx = np.argmin(np.abs(lons - lon2))

            # Constrain both fcst1_em_residual and obs_resid to the gridbox
            fcst1_em_residual_gridbox = fcst1_em_residual[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
            obs_resid_gridbox = obs_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean = np.nanmean(fcst1_em_residual_gridbox, axis=(1, 2))
            obs_resid_mean = np.nanmean(obs_resid_gridbox, axis=(1, 2))

            # Calculate the correlation between the two
            r, p = pearsonr(fcst1_em_residual_mean, obs_resid_mean)

            # Show these values e.g. r = 0.50, p = 0.01 in the lower left textbox
            ax2.text(0.05, 0.05, f"r' = {r:.2f}, p = {p:.2f}", transform=ax2.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        elif 'south' and 'north' in gridbox and plot_gridbox is None:
            print("Global plot, gridbox specified which contains south and north")
            # Add green lines outlining the north gridbox
            ax2.plot([lon1_n, lon2_n, lon2_n, lon1_n, lon1_n], [lat1_n, lat1_n, lat2_n, lat2_n, lat1_n],
                    color='green', linewidth=2, transform=proj)
            
            # Add green lines outlining the south gridbox
            ax2.plot([lon1_s, lon2_s, lon2_s, lon1_s, lon1_s], [lat1_s, lat1_s, lat2_s, lat2_s, lat1_s],
                    color='green', linewidth=2, transform=proj)
            
            # Constrain the partia_r array to the north gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_n = np.argmin(np.abs(lats - lat1_n))
            lat2_idx_n = np.argmin(np.abs(lats - lat2_n))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_n = np.argmin(np.abs(lons - lon1_n))
            lon2_idx_n = np.argmin(np.abs(lons - lon2_n))

            # Constrain the partia_r array to the south gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_s = np.argmin(np.abs(lats - lat1_s))
            lat2_idx_s = np.argmin(np.abs(lats - lat2_s))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_s = np.argmin(np.abs(lons - lon1_s))
            lon2_idx_s = np.argmin(np.abs(lons - lon2_s))

            # Constrain both fcst1_em_residual and obs_resid to the north gridbox
            fcst1_em_residual_gridbox_n = fcst1_em_residual[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]
            obs_resid_gridbox_n = obs_resid[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            # Constrain both fcst1_em_residual and obs_resid to the south gridbox
            fcst1_em_residual_gridbox_s = fcst1_em_residual[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]
            obs_resid_gridbox_s = obs_resid[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean_n = np.nanmean(fcst1_em_residual_gridbox_n, axis=(1, 2))
            obs_resid_mean_n = np.nanmean(obs_resid_gridbox_n, axis=(1, 2))

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean_s = np.nanmean(fcst1_em_residual_gridbox_s, axis=(1, 2))
            obs_resid_mean_s = np.nanmean(obs_resid_gridbox_s, axis=(1, 2))

            # Calculate the correlation between the two
            r_n, p_n = pearsonr(fcst1_em_residual_mean_n, obs_resid_mean_n)
            
            # Calculate the correlation between the two
            r_s, p_s = pearsonr(fcst1_em_residual_mean_s, obs_resid_mean_s)

            # Show these values e.g. r_{n} = 0.50, p_{n} = 0.01 in the lower left textbox
            # and r_{s} = 0.50, p_{s} = 0.01 in the same textbox, but on the next line
            ax2.text(0.05, 0.05, 
                        f"r$'_{{n}}$ = {r_n:.2f}, p$_{{n}}$ = {p_n:.2f}\n" + \
                        f"r$'_{{s}}$ = {r_s:.2f}, p$_{{s}}$ = {p_s:.2f}",
                        transform=ax2.transAxes,
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5),
                        fontsize=10)
        elif gridbox is not None and 'south' not in gridbox and 'north' not in gridbox and plot_gridbox is not None:
            print("Plot constrained to plot_gridbox, gridbox specified, but doesn't contain south and north")
            # Add green lines outlining the gridbox
            ax2.plot([lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
                    color='green', linewidth=2, transform=proj)
            
            # Constrain the partia_r array to the gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats_cs - lat1))
            lat2_idx = np.argmin(np.abs(lats_cs - lat2))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons_cs - lon1))
            lon2_idx = np.argmin(np.abs(lons_cs - lon2))

            # Constrain both fcst1_em_residual and obs_resid to the gridbox
            fcst1_em_residual_gridbox = fcst1_em_residual_cs[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
            obs_resid_gridbox = obs_resid_cs[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean = np.nanmean(fcst1_em_residual_gridbox, axis=(1, 2))
            obs_resid_mean = np.nanmean(obs_resid_gridbox, axis=(1, 2))

            # Print the gridbox mean of both fcst1_em_residual and obs_resid
            print("fcst1_em_residual_mean = ", fcst1_em_residual_mean)
            print("obs_resid_mean = ", obs_resid_mean)

            # Calculate the correlation between the two
            r, p = pearsonr(fcst1_em_residual_mean, obs_resid_mean)

            # Show these values e.g. r = 0.50, p = 0.01 in the lower left textbox
            ax2.text(0.05, 0.05, f"r' = {r:.2f}, p = {p:.2f}", transform=ax2.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
            
        elif 'south'and 'north' in gridbox and plot_gridbox is not None:
            print("Plot constrained to plot_gridbox, gridbox specified which contains south and north")
                        # Add green lines outlining the north gridbox
            ax2.plot([lon1_n, lon2_n, lon2_n, lon1_n, lon1_n], [lat1_n, lat1_n, lat2_n, lat2_n, lat1_n],
                    color='green', linewidth=2, transform=proj)
            
            # Add green lines outlining the south gridbox
            ax2.plot([lon1_s, lon2_s, lon2_s, lon1_s, lon1_s], [lat1_s, lat1_s, lat2_s, lat2_s, lat1_s],
                    color='green', linewidth=2, transform=proj)
            
            # Constrain the partia_r array to the north gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_n = np.argmin(np.abs(lats_cs - lat1_n))
            lat2_idx_n = np.argmin(np.abs(lats_cs - lat2_n))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_n = np.argmin(np.abs(lons_cs - lon1_n))
            lon2_idx_n = np.argmin(np.abs(lons_cs - lon2_n))

            # Constrain the partia_r array to the south gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx_s = np.argmin(np.abs(lats_cs - lat1_s))
            lat2_idx_s = np.argmin(np.abs(lats_cs - lat2_s))

            # find the indices of the lons which correspond to the gridbox
            lon1_idx_s = np.argmin(np.abs(lons_cs - lon1_s))
            lon2_idx_s = np.argmin(np.abs(lons_cs - lon2_s))

            # Constrain both fcst1_em_residual and obs_resid to the north gridbox
            fcst1_em_residual_gridbox_n = fcst1_em_residual_cs[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]
            obs_resid_gridbox_n = obs_resid_cs[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            # Constrain both fcst1_em_residual and obs_resid to the south gridbox
            fcst1_em_residual_gridbox_s = fcst1_em_residual_cs[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]
            obs_resid_gridbox_s = obs_resid_cs[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean_n = np.nanmean(fcst1_em_residual_gridbox_n, axis=(1, 2))
            obs_resid_mean_n = np.nanmean(obs_resid_gridbox_n, axis=(1, 2))

            # Calculate the gridbox mean of both fcst1_em_residual and obs_resid
            fcst1_em_residual_mean_s = np.nanmean(fcst1_em_residual_gridbox_s, axis=(1, 2))
            obs_resid_mean_s = np.nanmean(obs_resid_gridbox_s, axis=(1, 2))

            # Calculate the correlation between the two
            r_n, p_n = pearsonr(fcst1_em_residual_mean_n, obs_resid_mean_n)
            
            # Calculate the correlation between the two
            r_s, p_s = pearsonr(fcst1_em_residual_mean_s, obs_resid_mean_s)

            # Show these values e.g. r_{n} = 0.50, p_{n} = 0.01 in the lower left textbox
            # and r_{s} = 0.50, p_{s} = 0.01 in the same textbox, but on the next line
            # Only if plot_corr_diff is False
            if plot_corr_diff is False:
                ax2.text(0.05, 0.05, 
                            f"r$'_{{n}}$ = {r_n:.2f}, p$_{{n}}$ = {p_n:.2f}\n" + \
                            f"r$'_{{s}}$ = {r_s:.2f}, p$_{{s}}$ = {p_s:.2f}",
                            transform=ax2.transAxes,
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.5),
                            fontsize=10)        # Append the contourf object to the list
        cf_list.append(cf)

        if plot_gridbox is not None:
            if np.isnan(partial_r_cs).any():
                print("there are NaNs in the partial_r_cs array")
                # Set the partial_r_p values to NaN at those points
                partial_r_p_cs[partial_r_cs == np.nan] = np.nan

            # If any of the partial_r_p values are greater than the significance
            # threshold - set them to NaN
            partial_r_p_cs[partial_r_p_cs > sig_threshold] = np.nan

            # Plot the p-values for the partial correlation between the initialized
            # forecast and the observations after removing the influence of the
            # uninitialized forecast
            ax2.contourf(lons_cs, lats_cs, partial_r_p_cs, hatches=['....'], alpha=0.,
                            transform=proj)
        else:
            # If any of the partial_r values are NaN
            if np.isnan(partial_r).any():
                print("there are NaNs in the partial_r array")
                # Set the partial_r_p values to NaN at those points
                partial_r_p[partial_r == np.nan] = np.nan

            # If any of the partial_r_p values are greater than the significance
            # threshold - set them to NaN
            partial_r_p[partial_r_p > sig_threshold] = np.nan

            # Plot the p-values for the partial correlation between the initialized
            # forecast and the observations after removing the influence of the
            # uninitialized forecast
            ax2.contourf(lons, lats, partial_r_p, hatches=['....'], alpha=0.,
                            transform=proj)
        
        # Add a textbox with the figure label
        ax2.text(0.95, 0.05, ax_labels[(ncols*i)+1], transform=ax2.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the plot name
        ax2.text(0.05, 0.95, plot_names[1], transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the method
        # to the top right of the plot
        ax2.text(0.95, 0.95, method, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # # add a textbox with the season
        # # to the bottom left of the plot
        # ax2.text(0.05, 0.05, season, transform=ax2.transAxes,
        #             verticalalignment='bottom', horizontalalignment='left',
        #             bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)

        # If plot_corr_diff is True
        if plot_corr_diff:
            print("plotting the difference in correlation - including third column")

            # Set up the axes for the difference in correlation
            ax3 = axs[i, 2]
            ax3.coastlines()

            # Set up the clevs for the difference in correlation
            # Set up the contour levels
            clevs_diff = np.arange(-0.6, 0.7, 0.1)

            # if the region_grid is China, add borders
            if region_name == 'china':
                ax3.add_feature(cfeature.BORDERS, linestyle=':')

            if plot_gridbox is not None:
                cf = ax3.contourf(lons_cs, lats_cs, corr_diff_cs, clevs_diff, cmap='RdBu_r',
                                    transform=proj, extend='both')
            else:
                cf = ax3.contourf(lons, lats, corr_diff, clevs_diff, cmap='RdBu_r',
                                    transform=proj, extend='both')

            # Append the contourf object to the list
            cf_list.append(cf)

            # if the gridbox is not None
            # Set up the p field
            if plot_gridbox is not None:
                # If any of the corr_diff_p values are situated at NaNs
                if np.isnan(corr_diff_cs).any():
                    print("there are NaNs in the corr_diff_p_cs array")
                    # Set the corr_diff_p values to NaN at those points
                    corr_diff_p_cs[corr_diff_cs == np.nan] = np.nan

                # If any of the corr_diff_p values are greater than the significance
                # threshold - set them to NaN
                corr_diff_p_cs[corr_diff_p_cs > sig_threshold] = np.nan

                # Plot the p-values for the difference in correlation between the
                # initialized forecast and the observations
                ax3.contourf(lons_cs, lats_cs, corr_diff_p_cs, hatches=['....'], alpha=0.,
                                transform=proj)
            else:
                # If any of the corr_diff_p values are situated at NaNs
                if np.isnan(corr_diff).any():
                    print("there are NaNs in the corr_diff_p array")
                    # Set the corr_diff_p values to NaN at those points
                    corr_diff_p[corr_diff == np.nan] = np.nan

                # If any of the corr_diff_p values are greater than the significance
                # threshold - set them to NaN
                corr_diff_p[corr_diff_p > sig_threshold] = np.nan

                # Plot the p-values for the difference in correlation between the
                # initialized forecast and the observations
                ax3.contourf(lons, lats, corr_diff_p, hatches=['....'], alpha=0.,
                                transform=proj)
                
            # Add a textbox with the figure label
            ax3.text(0.95, 0.05, ax_labels[(ncols*i)+2], transform=ax3.transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
            
            # Add a textbox with the plot name
            ax3.text(0.05, 0.95, plot_names[2], transform=ax3.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
            
            # Add a textbox with the method
            # to the top right of the plot
            ax3.text(0.95, 0.95, method, transform=ax3.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
            
    # Add a colorbar for the correlation
    cbar = fig.colorbar(cf_list[0], ax=axs[:, :2], orientation='horizontal', pad = 0.05,
                        aspect=50, shrink=0.8)
    cbar.set_label('correlation coefficient')

    # If plot_corr_diff is True
    if plot_corr_diff:
        # Add a colorbar for the correlation difference
        cbar_diff = fig.colorbar(cf_list[-1], ax=axs[:, 2], orientation='horizontal', pad = 0.05,
                                aspect=50)
        cbar_diff.set_label('correlation difference')

    # if seasons_list is not None
    if seasons_list is not None:
        seasons_list_str = '_'.join(seasons_list)
        seasons_list = seasons_list_str

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/raw_init_impact_{variable}_{season}_" + \
    f"{forecast_range}_different_methods_{no_bootstraps}_{start_year}" + \
    f"_{finish_year}_{region_name}_seasons_{seasons_list}.png"

    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # show the figure
    plt.show()

# Define a function which loads the files and plots the different methods
# for the same season and variable
def load_files_and_plot(variable: str, region: str, season: str, forecast_range: str, methods_list: list,
                        no_bootstraps: int, plots_dir: str, bootstrap_base_dir: str,
                        gridbox: dict = None, figsize_x: int = 10, figsize_y: int = 10,
                        plot_gridbox: dict = None, region_name: str = None,
                        plot_different_methods: bool = True,
                        plot_different_seasons: bool = False,
                        seasons_list: list = None,
                        plot_corr_diff: bool = False) -> None:
    
    """
    Wrapper function which loads the required files and plots the different
    methods for the same season and variable.
    
    Args:
    
        variable (str): variable to plot
        
        region (str): region to plot
        
        season (str): season to plot
        
        forecast_range (str): forecast range to plot
        
        methods_list (list): list of methods to plot (e.g. raw, lagged, nao_matched)
        
        no_bootstraps (int): number of bootstraps to plot
        
        plots_dir (str): path to the directory to save the plots
        
        bootstrap_base_dir (str): path to the directory containing the bootstrapped
                                    files
                                    
        gridbox (dict): dictionary containing the gridbox to plot. Default is None.
                        contains constrained gridbox with dimensions as follows:
                            'lon1': lower longitude bound
                            'lon2': upper longitude bound
                            'lat1': lower latitude bound
                            'lat2': upper latitude bound
                            
        figsize_x (int): size of the figure in the x direction. Default is 10.
        
        figsize_y (int): size of the figure in the y direction. Default is 12.

        plot_gridbox (dict): dictionary containing the gridbox to plot. Default is None.
                        contains constrained gridbox with dimensions as follows:
                            'lon1': lower longitude bound
                            'lon2': upper longitude bound
                            'lat1': lower latitude bound
                            'lat2': upper latitude bound

        region_name (str): name of the region to plot. Default is None.
        
        plot_different_methods (bool): whether to plot the different methods for the
                                        same season and variable. Default is True.

        plot_different_seasons (bool): whether to plot the different seasons for the
                                        same variable and method. Default is False.

        seasons_list (list): list of seasons to plot. Default is None.
                            Needed only if plot_different_seasons is True.

        plot_corr_diff (bool): whether to plot the difference in correlation between
                                the initialized forecast and the observations. Default  
                                is False.

    Returns:
    
        None
    """

    # Set up the lists
    paths_list = []

    # Assert that if one of the boolean flags is True, the other is False
    assert (plot_different_methods and not plot_different_seasons) or \
            (plot_different_seasons and not plot_different_methods), \
            "One of plot_different_methods and plot_different_seasons must be True, " + \
            "and the other must be False"

    # Assert that if plot_different_seasons is True, seasons_list is not None
    assert (plot_different_seasons and seasons_list is not None) or \
            (not plot_different_seasons and seasons_list is None), \
            "If plot_different_seasons is True, seasons_list must not be None"

    # If plot_different_methods is True
    if plot_different_methods:
        # Form the paths for the different methods
        for method in methods_list:

            # Use os.path.join to join the paths
            path = os.path.join(bootstrap_base_dir, variable, region, season, forecast_range, method, f"no_bootstraps_{no_bootstraps}")
            
            # Print the path
            print(path)

            # # Assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist for method {method}" + \
            f" and no_bootstraps {no_bootstraps}"

            # Append the path to the list
            paths_list.append(path)
        
    elif plot_different_seasons:
        # Form the paths for the different seasons
        for season in seasons_list:

            # Use os.path.join to join the paths
            path = os.path.join(bootstrap_base_dir, variable, region, season, 
                                forecast_range, methods_list[0], 
                                f"no_bootstraps_{no_bootstraps}")
            
            # Print the path
            # print(path)

            # # Assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist for season {season}" + \
            f" and no_bootstraps {no_bootstraps}"

            # Append the path to the list
            paths_list.append(path)
    else:
        raise ValueError("plot_different_methods and plot_different_seasons cannot both be False")

    # Loop over the paths to get the values
    values_list = [extract_values_from_txt(path, variable) for path in paths_list]

    # Loop over the paths to get the arrays
    arrays_list = [load_arrays_from_npy(path, variable) for path in paths_list]

    # Set the number of bootstraps to 1
    # to extract the time series
    # if variable != 'ua':
    no_bootstraps = 1

    # Initialize a list for the paths to the time series files
    paths_list_ts = []

    # If plot_different_methods is True
    if plot_different_methods:
        # Form the paths for the different methods
        for method in methods_list:

            # Set up the path to the file using os.path.join
            path = os.path.join(bootstrap_base_dir, variable, region, season, forecast_range, method, f"no_bootstraps_{no_bootstraps}")

            # # Set up the path to the file
            # path = f"{bootstrap_base_dir}/{variable}/{region}/{season}/" + \
            #         f"{forecast_range}/{method}/no_bootstraps_{no_bootstraps}"
            
            print(path)

            # Assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist for method {method}" + \
            f" and no_bootstraps {no_bootstraps}"

            # Append the path to the list
            paths_list_ts.append(path)
    elif plot_different_seasons:
        # Form the paths for the different seasons
        for season in seasons_list:

            # Set up the path to the file using os.path.join
            path = os.path.join(bootstrap_base_dir, variable, region, season, 
                                forecast_range, methods_list[0], 
                                f"no_bootstraps_{no_bootstraps}")

            # # Set up the path to the file
            # path = f"{bootstrap_base_dir}/{variable}/{region}/{season}/" + \
            #         f"{forecast_range}/{method}/no_bootstraps_{no_bootstraps}"
            
            print(path)

            # Assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist for season {season}" + \
            f" and no_bootstraps {no_bootstraps}"

            # Append the path to the list
            paths_list_ts.append(path)

    # Loop over the paths to get the timeseries arrays
    ts_arrays_list = [load_arrays_from_npy(path, variable, timeseries=True) for path in paths_list_ts]

    # Reset the number of bootstraps
    no_bootstraps = 1000

    # If plot_different_methods is True
    if plot_different_methods:
        # Plot the different methods
        plot_different_methods_same_season_var(arrays_list, values_list, variable, season,
                                                forecast_range, methods_list, no_bootstraps,
                                                plots_dir, gridbox=gridbox,
                                                figsize_x=figsize_x, figsize_y=figsize_y,
                                                plot_gridbox=plot_gridbox,
                                                ts_arrays=ts_arrays_list,
                                                region_name=region_name,
                                                plot_corr_diff=plot_corr_diff)
    elif plot_different_seasons:
        # Set the season to None
        season=None

        # Call the function to plot the different seasons
        plot_different_methods_same_season_var(arrays_list, values_list, variable, season,
                                                forecast_range, methods_list,
                                                no_bootstraps, plots_dir,
                                                gridbox=gridbox,
                                                figsize_x=figsize_x, figsize_y=figsize_y,
                                                plot_gridbox=plot_gridbox,
                                                ts_arrays=ts_arrays_list,
                                                region_name=region_name,
                                                seasons_list=seasons_list,
                                                plot_corr_diff=plot_corr_diff)
    else:
        raise ValueError("plot_different_methods and plot_different_seasons cannot both be False")
    
    return None

# Define a function similar to load files and plot
# Which will load the time series arrays and calculate the correlation
# coefficients for the different methods
def load_files_and_plot_corr(variable: str, region: str, seasons_list: list, 
                                forecast_range: str, methods_list: list,
                                plots_dir: str, bootstrap_base_dir: str,
                                gridbox: dict, figsize_x: int = 10, figsize_y: int = 10,
                                no_bootstraps: int = 1) -> None:
    """
    Wrapper function which loads the required files and plots a scatter plot
    showing the correlation coefficients for the different methods and seasons.

    Args:

        variable (str): variable to plot

        region (str): region to plot

        seasons_list (list): list of seasons to plot

        forecast_range (str): forecast range to plot

        methods_list (list): list of methods to plot (e.g. raw, lagged, nao_matched)

        plots_dir (str): path to the directory to save the plots

        bootstrap_base_dir (str): path to the directory containing the bootstrapped
                                    files

        gridbox (dict): dictionary containing the gridbox to plot. Default is None.
                        contains constrained gridbox with dimensions as follows:
                            'lon1': lower longitude bound
                            'lon2': upper longitude bound
                            'lat1': lower latitude bound
                            'lat2': upper latitude bound

        figsize_x (int): size of the figure in the x direction. Default is 10.

        figsize_y (int): size of the figure in the y direction. Default is 10.

        no_bootstraps (int): number of bootstraps to plot. Default is 1.

    Returns:

        None
    """

    # Set up the lists
    paths_list = []

    # Set up the lats and lons
    lats = np.arange(-90, 90, 2.5)
    lons = np.arange(-180, 180, 2.5)

    # Assert that gridbox is not None and is a dictionary
    assert gridbox is not None and isinstance(gridbox, dict), "gridbox must be specified and must be a dictionary"

    # Extract the gridbox values
    if 'south' and 'north' not in gridbox:
        print("gridbox specified, but doesn't contain south and north")
        
        # Extract the gridbox values
        lon1 = gridbox['lon1'] ; lon2 = gridbox['lon2']

        lat1 = gridbox['lat1'] ; lat2 = gridbox['lat2']
    elif 'south' and 'north' in gridbox:
        print("gridbox specified which contains south and north")

        # Extract the south and north gridbox values
        s_gridbox = gridbox['south'] ; n_gridbox = gridbox['north']

        # Extract the gridbox values
        lon1_n = n_gridbox['lon1'] ; lon2_n = n_gridbox['lon2']

        lat1_n = n_gridbox['lat1'] ; lat2_n = n_gridbox['lat2']

        lon1_s = s_gridbox['lon1'] ; lon2_s = s_gridbox['lon2']

        lat1_s = s_gridbox['lat1'] ; lat2_s = s_gridbox['lat2']

    # Form the paths for the different methods
    for season in seasons_list:
        
        for method in methods_list:

            # Set up the path to the file
            path = f"{bootstrap_base_dir}/{variable}/{region}/{season}/" + \
                    f"{forecast_range}/{method}/no_bootstraps_{no_bootstraps}"

            # Assert that the path exists
            assert os.path.exists(path), f"Path {path} does not exist for method {method}" + \
            f" and no_bootstraps {no_bootstraps}"

            # Append the path to the list
            paths_list.append(path)

    # Loop over the paths to extract the time series arrays
    arrays_ts_list = [load_arrays_from_npy(path, variable, timeseries=True) for path in paths_list]

    # if south and north are not in gridbox
    if 'south' and 'north' not in gridbox:
        # Initialize an empty list for the correlation coefficients
        r_list = []

        p_list = []

        partial_r_list = []

        partial_r_p_list = []

    elif 'south' and 'north' in gridbox:
        # Initialize an empty list for the correlation coefficients
        r_list_n = [] ; p_list_n = []

        r_list_s = [] ; p_list_s = []

        partial_r_list_n = [] ; partial_r_p_list_n = []

        partial_r_list_s = [] ; partial_r_p_list_s = []

    # Loop over the arrays_ts_list
    for arrays_ts in arrays_ts_list:

        # Extract the initialized forecast timeseries
        fcst1_ts = arrays_ts['fcst1_ts'] ; obs_ts = arrays_ts['obs_ts']

        # Extract the residual timeseries
        fcst1_em_resid = arrays_ts['fcst1_em_resid'] ; obs_resid = arrays_ts['obs_resid']

        # if south and north are not in gridbox
        if 'south' and 'north' not in gridbox:
            print("gridbox specified, but doesn't contain south and north")

            # Set up the lat indices which correspond to the gridbox
            # find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats - lat1))

            lat2_idx = np.argmin(np.abs(lats - lat2))

            # Set up the lon indices which correspond to the gridbox
            # find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons - lon1))

            lon2_idx = np.argmin(np.abs(lons - lon2))

            # Constrain all the arrays to the gridbox
            fcst1_ts_gridbox = fcst1_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            obs_ts_gridbox = obs_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            fcst1_em_resid_gridbox = fcst1_em_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            obs_resid_gridbox = obs_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the gridbox mean of all the arrays
            fcst1_ts_mean = np.nanmean(fcst1_ts_gridbox, axis=(1, 2))

            obs_ts_mean = np.nanmean(obs_ts_gridbox, axis=(1, 2))

            fcst1_em_resid_mean = np.nanmean(fcst1_em_resid_gridbox, axis=(1, 2))

            obs_resid_mean = np.nanmean(obs_resid_gridbox, axis=(1, 2))

            # Calculate the correlation between the initialized forecast and the observations
            r, p = pearsonr(fcst1_ts_mean, obs_ts_mean)

            # Calculate the partial correlation between the initialized forecast and the observations
            partial_r, partial_r_p = pearsonr(fcst1_em_resid_mean, obs_resid_mean)

            # Append the correlation coefficients to the list
            r_list.append(r) ; p_list.append(p)

            partial_r_list.append(partial_r) ; partial_r_p_list.append(partial_r_p)

        # if south and north are in gridbox
        elif 'south' and 'north' in gridbox:

            # Find the indices of the lats which correspond to the gridbox
            lat1_idx_n = np.argmin(np.abs(lats - lat1_n))

            lat2_idx_n = np.argmin(np.abs(lats - lat2_n))

            # Find the indices of the lons which correspond to the gridbox
            lon1_idx_n = np.argmin(np.abs(lons - lon1_n))

            lon2_idx_n = np.argmin(np.abs(lons - lon2_n))

            # For the south gridbox do the same
            # Find the indices of the lats which correspond to the gridbox
            lat1_idx_s = np.argmin(np.abs(lats - lat1_s))

            lat2_idx_s = np.argmin(np.abs(lats - lat2_s))

            # Find the indices of the lons which correspond to the gridbox
            lon1_idx_s = np.argmin(np.abs(lons - lon1_s))

            lon2_idx_s = np.argmin(np.abs(lons - lon2_s))

            # Constrain all the arrays to the north gridbox
            fcst1_ts_gridbox_n = fcst1_ts[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            obs_ts_gridbox_n = obs_ts[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            fcst1_em_resid_gridbox_n = fcst1_em_resid[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            obs_resid_gridbox_n = obs_resid[:, lat1_idx_n:lat2_idx_n, lon1_idx_n:lon2_idx_n]

            # Constrain all the arrays to the south gridbox
            fcst1_ts_gridbox_s = fcst1_ts[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            obs_ts_gridbox_s = obs_ts[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            fcst1_em_resid_gridbox_s = fcst1_em_resid[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            obs_resid_gridbox_s = obs_resid[:, lat1_idx_s:lat2_idx_s, lon1_idx_s:lon2_idx_s]

            # Calculate the gridbox mean of all the arrays north
            fcst1_ts_mean_n = np.nanmean(fcst1_ts_gridbox_n, axis=(1, 2))

            obs_ts_mean_n = np.nanmean(obs_ts_gridbox_n, axis=(1, 2))

            fcst1_em_resid_mean_n = np.nanmean(fcst1_em_resid_gridbox_n, axis=(1, 2))

            obs_resid_mean_n = np.nanmean(obs_resid_gridbox_n, axis=(1, 2))

            # Calculate the gridbox mean of all the arrays south

            fcst1_ts_mean_s = np.nanmean(fcst1_ts_gridbox_s, axis=(1, 2))

            obs_ts_mean_s = np.nanmean(obs_ts_gridbox_s, axis=(1, 2))

            fcst1_em_resid_mean_s = np.nanmean(fcst1_em_resid_gridbox_s, axis=(1, 2))

            obs_resid_mean_s = np.nanmean(obs_resid_gridbox_s, axis=(1, 2))

            # Calculate the correlation between the initialized forecast and the observations
            r_n, p_n = pearsonr(fcst1_ts_mean_n, obs_ts_mean_n)

            # Calculate the partial correlation between the initialized forecast and the observations
            partial_r_n, partial_r_p_n = pearsonr(fcst1_em_resid_mean_n, obs_resid_mean_n)

            # Calculate the correlation between the initialized forecast and the observations
            r_s, p_s = pearsonr(fcst1_ts_mean_s, obs_ts_mean_s)

            # Calculate the partial correlation between the initialized forecast and the observations
            partial_r_s, partial_r_p_s = pearsonr(fcst1_em_resid_mean_s, obs_resid_mean_s)

            # Append the correlation coefficients to the list
            r_list_n.append(r_n) ; p_list_n.append(p_n)

            partial_r_list_n.append(partial_r_n) ; partial_r_p_list_n.append(partial_r_p_n)

            r_list_s.append(r_s) ; p_list_s.append(p_s)

            partial_r_list_s.append(partial_r_s) ; partial_r_p_list_s.append(partial_r_p_s)
        else:
            raise ValueError("gridbox must contain either south and north or neither")

    # Set up the figure for a single plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figsize_x, figsize_y))

    # Set up the title
    title = f"scatter plot of correlation coefficients for {variable} " + \
            f"for {region} for {forecast_range} for different methods"
    
    # Set the title
    plt.suptitle(title, fontsize=8)

    # Set up the x-axis labels as the seasons list
    x_labels = seasons_list

    # if south and north are not in gridbox
    if 'south' and 'north' not in gridbox:
        print("gridbox specified, but doesn't contain south and north, plotting")

        # Extract the raw correlation coefficients
        # Which will the index 0, 3, 6, 9
        r_raw = r_list[0::3] ; p_raw = p_list[0::3]

        # Extract the lagged correlation coefficients
        # Which will the index 1, 4, 7, 10
        r_lagged = r_list[1::3] ; p_lagged = p_list[1::3]

        # Extract the nao-matched correlation coefficients
        # Which will the index 2, 5, 8, 11
        r_nao_matched = r_list[2::3] ; p_nao_matched = p_list[2::3]

        # Also extract the partial correlation coefficients
        partial_r_raw = partial_r_list[0::3] ; partial_r_p_raw = partial_r_p_list[0::3]

        partial_r_lagged = partial_r_list[1::3] ; partial_r_p_lagged = partial_r_p_list[1::3]

        partial_r_nao_matched = partial_r_list[2::3] ; partial_r_p_nao_matched = partial_r_p_list[2::3]

        # Plot the raw correlation coefficients
        ax.scatter(x_labels, r_raw, label='raw', color='blue')

        # Plot the lagged correlation coefficients
        ax.scatter(x_labels, r_lagged, label='lagged', color='orange')

        # Plot the nao-matched correlation coefficients
        ax.scatter(x_labels, r_nao_matched, label='nao-matched', color='green')

        # Plot the partial raw correlation coefficients
        ax.scatter(x_labels, partial_r_raw, label='partial_raw', color='blue', marker='x')

        # Plot the partial lagged correlation coefficients
        ax.scatter(x_labels, partial_r_lagged, label='partial_lagged', color='orange', marker='x')

        # Plot the partial nao-matched correlation coefficients
        ax.scatter(x_labels, partial_r_nao_matched, label='partial_nao-matched', color='green', marker='x')

    elif 'south' and 'north' in gridbox:
        print("gridbox specified which contains south and north, plotting")

        # Extract the raw correlation coefficients
        # Which will the index 0, 3, 6, 9
        r_raw_n = r_list_n[0::3] ; p_raw_n = p_list_n[0::3]

        r_raw_s = r_list_s[0::3] ; p_raw_s = p_list_s[0::3]

        # Extract the lagged correlation coefficients
        # Which will the index 1, 4, 7, 10
        r_lagged_n = r_list_n[1::3] ; p_lagged_n = p_list_n[1::3]

        r_lagged_s = r_list_s[1::3] ; p_lagged_s = p_list_s[1::3]

        # Extract the nao-matched correlation coefficients
        # Which will the index 2, 5, 8, 11
        r_nao_matched_n = r_list_n[2::3] ; p_nao_matched_n = p_list_n[2::3]

        r_nao_matched_s = r_list_s[2::3] ; p_nao_matched_s = p_list_s[2::3]

        # Also extract the partial correlation coefficients
        partial_r_raw_n = partial_r_list_n[0::3] ; partial_r_p_raw_n = partial_r_p_list_n[0::3]

        partial_r_lagged_n = partial_r_list_n[1::3] ; partial_r_p_lagged_n = partial_r_p_list_n[1::3]

        partial_r_nao_matched_n = partial_r_list_n[2::3] ; partial_r_p_nao_matched_n = partial_r_p_list_n[2::3]

        partial_r_raw_s = partial_r_list_s[0::3] ; partial_r_p_raw_s = partial_r_p_list_s[0::3]

        partial_r_lagged_s = partial_r_list_s[1::3] ; partial_r_p_lagged_s = partial_r_p_list_s[1::3]

        partial_r_nao_matched_s = partial_r_list_s[2::3] ; partial_r_p_nao_matched_s = partial_r_p_list_s[2::3]

        # Plot the raw correlation coefficients
        ax.scatter(x_labels, r_raw_n, label='raw', color='blue', marker='^')

        ax.scatter(x_labels, r_raw_s, color='blue', marker='v')

        # Plot the lagged correlation coefficients
        ax.scatter(x_labels, r_lagged_n, label='lagged', color='orange', marker='^')

        ax.scatter(x_labels, r_lagged_s, color='orange', marker='v')

        # Plot the nao-matched correlation coefficients
        ax.scatter(x_labels, r_nao_matched_n, label='nao-matched', color='green', marker='^')

        ax.scatter(x_labels, r_nao_matched_s, color='green', marker='v')

        # Plot the partial raw correlation coefficients
        ax.scatter(x_labels, partial_r_raw_n, color='blue', marker='^', edgecolors='red')

        ax.scatter(x_labels, partial_r_raw_s, color='blue', marker='v', edgecolors='red')

        # Plot the partial lagged correlation coefficients
        ax.scatter(x_labels, partial_r_lagged_n, color='orange', marker='^', edgecolors='red')

        ax.scatter(x_labels, partial_r_lagged_s, color='orange', marker='v', edgecolors='red')

        # Plot the partial nao-matched correlation coefficients
        ax.scatter(x_labels, partial_r_nao_matched_n, color='green', marker='^', edgecolors='red')

        ax.scatter(x_labels, partial_r_nao_matched_s, color='green', marker='v', edgecolors='red')
    else:
        raise ValueError("gridbox must contain either south and north or neither")

    # Add a legend
    ax.legend()

    # Set up the ylims for this plot
    ax.set_ylim(-0.5, 1)

    # Set the x-axis label
    ax.set_xlabel('season')

    # Add in a faint grey line for the zero correlation
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

    # Set the y-axis label
    ax.set_ylabel('correlation coefficient')

    # Set the x-axis ticks
    ax.set_xticks(x_labels)

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/corr_coeff_{variable}_{region}_" + \
    f"{forecast_range}_different_methods_{no_bootstraps}.png"

    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Show the figure
    plt.show()

    return None

# Write a new function to plot the time series of the forecasts
# for the initialized, uninitialized and observed data
def plot_diff_methods_same_season_var_timeseries(ts_arrays: list, values: list,
                                                 variable: str, season: str,
                                                 forecast_range: str, method_list: list,
                                                 no_bootstraps: int, plots_dir: str,
                                                 gridbox: dict = None,
                                                 figsize_x: int = 10, figsize_y: int = 10) -> None:
    """
    Plots a 3 x 2 matrix of subplots. The first column is the total correlation
    timeseries and the second column is the residual correlation timeseries. The
    rows are for raw, lagged and nao-matched data, as defined in methods list.

    Args:

        ts_arrays (list): list of dicts containing the arrays to plot.
                            Indexed by the methods in method_list.
                            Each dictionary contains the following keys:
                                fcst1_ts (array) [time, lat, lon]: ensemble mean initialized forecast timeseries
                                fcst2_ts (array) [time, lat, lon]: ensemble mean uninitialized forecast timeseries
                                obs_ts (array) [time, lat, lon]: observed timeseries
                                fcst1_em_resid (array) [time, lat, lon]: ensemble mean initialized forecast timeseries with the ensemble mean of the uninitialized forecast removed
                                obs_resid (array) [time, lat, lon]: observed timeseries with the ensemble mean of the uninitialized forecast removed.

        values (list): list of dicts containing the values to plot.
                        Indexed by the methods in method_list.
                        Each dictionary contains the following keys:
                            nens1 (int): number of ensemble members in the first ensemble (initialized)
                            nens2 (int): number of ensemble members in the second ensemble (uninitialized)
                            start_year (int): start year of the forecast
                            end_year (int): end year of the forecast

        variable (str): variable to plot

        season (str): season to plot

        forecast_range (str): forecast range to plot

        method_list (list): list of methods to plot (e.g. raw, lagged, nao_matched)

        no_bootstraps (int): number of bootstraps to plot

        plots_dir (str): path to the directory to save the plots

        gridbox (dict): dictionary containing the gridbox to calculate the time
                        series for. Default is None.
                        Contains constrained gridbox with dimensions as follows:
                            'lon1': lower longitude bound
                            'lon2': upper longitude bound
                            'lat1': lower latitude bound
                            'lat2': upper latitude bound

        figsize_x (int): size of the figure in the x direction. Default is 10.

        figsize_y (int): size of the figure in the y direction. Default is 10.

    Returns:

        None

    """

    # Set up the axis labels
    ax_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Set up the plot_names
    plot_names = ['total skill', 'residual corr', 'corr diff']

    # Assert that gridbox is not None
    # as gridbox is needed to collapse 3D array into time series
    assert gridbox is not None, "gridbox must be specified"

    # Set up the figure size
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(figsize_x, figsize_y),
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
                            sharex='col', sharey='row')
    
    # Extract a start year and finish year from the values_list
    start_year = values[0]['start_year']
    finish_year = values[0]['end_year']

    # Set up the title
    title = 'Total and residual timeseries for ' + variable + ' in ' + season + ' for ' + forecast_range + \
        ' between ' + str(start_year) + ' and ' + str(finish_year) + \
        ' using different methods' + 'no_bootstraps = ' + str(no_bootstraps)
    
    # set up the supertitle
    fig.suptitle(title, fontsize=8, y=0.95)

    # Set up the lats and lons
    lons = np.arange(-180, 180, 2.5)
    lats = np.arange(-90, 90, 2.5)

    # Set up the significance threshold as 0.05
    sig_threshold = 0.05

    # Use the gridbox to constrain the lats and lons
    # find the indices of the lats which correspond to the gridbox
    lat1_idx = np.argmin(np.abs(lats - gridbox['lat1']))
    lat2_idx = np.argmin(np.abs(lats - gridbox['lat2']))

    # find the indices of the lons which correspond to the gridbox
    lon1_idx = np.argmin(np.abs(lons - gridbox['lon1']))
    lon2_idx = np.argmin(np.abs(lons - gridbox['lon2']))

    # Loop over the methods
    for i, method in enumerate(method_list):
        print("plotting index: ", i, " for method: ", method)

        # Extract the dictionaries for this method
        method_arrays = ts_arrays[i]
        method_values = values[i]

        # From the dictionaries, extract the arrays
        fcst1_ts = method_arrays['fcst1_ts']
        fcst2_ts = method_arrays['fcst2_ts']
        obs_ts = method_arrays['obs_ts']
        fcst1_em_resid = method_arrays['fcst1_em_resid']
        obs_resid = method_arrays['obs_resid']

        # From the dictionaries, extract the values
        nens1 = method_values['nens1']
        nens2 = method_values['nens2']
        start_year = method_values['start_year']
        end_year = method_values['end_year']

        # Process the arrays to get the time series
        # Frist constrain the arrays to the gridbox
        fcst1_ts_gridbox = fcst1_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
        fcst2_ts_gridbox = fcst2_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
        obs_ts_gridbox = obs_ts[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
        fcst1_em_resid_gridbox = fcst1_em_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]
        obs_resid_gridbox = obs_resid[:, lat1_idx:lat2_idx, lon1_idx:lon2_idx]

        # Calculate the mean over the gridbox
        fcst1_ts_mean = np.nanmean(fcst1_ts_gridbox, axis=(1, 2))
        fcst2_ts_mean = np.nanmean(fcst2_ts_gridbox, axis=(1, 2))
        obs_ts_mean = np.nanmean(obs_ts_gridbox, axis=(1, 2))
        fcst1_em_resid_mean = np.nanmean(fcst1_em_resid_gridbox, axis=(1, 2))
        obs_resid_mean = np.nanmean(obs_resid_gridbox, axis=(1, 2))

        # Set up the x-axis for the time series
        years = np.arange(start_year, end_year + 1)

        # If the method is lagged or nao_matched
        if method == 'lagged':
            print("Skipping the first three years of the time series")
            
            # Skip the first three years of the time series
            years = years[3:]
        else:
            print("Not skipping the first three years of the time series")

        # TODO: Potentially add the r-values and p-values to the plot

        # Set up the axes for the total skill
        ax1 = axs[i, 0]
        ax1.plot(years, fcst1_ts_mean, color='red', label='init')
        ax1.plot(years, fcst2_ts_mean, color='purple', label='unin')
        ax1.plot(years, obs_ts_mean, color='black', label='obs')

        # Set consitenet y-limits
        ax1.set_ylim([-0.8, 1.0])

        # Add a textbox with the figure label
        ax1.text(0.95, 0.05, ax_labels[2 * i], transform=ax1.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the method
        # to the top right of the plot
        ax1.text(0.95, 0.95, method, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)

        if i == 0:
            # Add a legend
            ax1.legend()

            # Set the title to 'anomalies'
            ax1.set_title('anomalies')

        # Set up the axes for the residual correlation
        ax2 = axs[i, 1]
        ax2.plot(years, fcst1_em_resid_mean, color='red', label='init')
        ax2.plot(years, obs_resid_mean, color='black', label='obs')
        # Add a zero dashed line in black at y = 0
        ax2.axhline(y=0, color='black', linestyle='--')

        # Add a textbox with the figure label
        ax2.text(0.95, 0.05, ax_labels[(2*i)+1], transform=ax2.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
        # Add a textbox with the method
        # to the top right of the plot
        ax2.text(0.95, 0.95, method, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)

        if i == 0:
            # Set the title to 'residuals'
            ax2.set_title('residuals')

            # Include a legend in the botom left corner
            ax2.legend(loc='lower left')

    # show the figure
    plt.show()

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/raw_init_impact_{variable}_{season}_" + \
    f"{forecast_range}_different_methods_timeseries_{no_bootstraps}_{start_year}" + \
    f"_{finish_year}.png"

    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    return None







