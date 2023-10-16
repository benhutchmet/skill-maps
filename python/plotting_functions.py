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
def plot_raw_init_impact_subplots(corr1_dict: dict, corr1_p_dict: dict, rpartial_dict: dict,
                                  rpartial_p_dict: dict, variable: str, seasons_list: list,
                                  forecast_range: str, method: str, no_bootstraps: int,
                                  nens1_dict: dict, nens2_dict: dict, start_year: int,
                                  finish_year: int, plots_dir: str) -> None:
    """
    Plots two subplots alongside (similar to Doug Smith's 2019 - Robust skill
    figure 3) each other. Left subplot is the correlation between the
    initialized forecast and the obs, and the right subplot is the residual
    correlation between the initialized forecast and the observations after
    removing the influence of the uninitialized forecast. 

    Subplots are for the seasons in the seasons_list.

    Args:
        corr1_dict (dict): dictionary of correlations between the initialized
                            forecast and the observations. Indexed by season.
        corr1_p_dict (dict): dictionary of p-values for the correlation
                                between the initialized forecast and the
                                observations. Indexed by season.
        rpartial_dict (dict): dictionary of residual correlations between the
                                initialized forecast and the observations after
                                removing the influence of the uninitialized
                                forecast. Indexed by season.
        rpartial_p_dict (dict): dictionary of p-values for the residual
                                    correlations between the initialized
                                    forecast and the observations after
                                    removing the influence of the
                                    uninitialized forecast. Indexed by season.
        variable (str): variable to plot
        seasons_list (list): list of seasons to plot
        forecast_range (str): forecast range to plot
        method (str): method to plot
        no_bootstraps (int): number of bootstraps to plot
        nens1_dict (dict): dictionary of number of ensemble members in the
                            first ensemble (initialized). Indexed by season.
        nens2_dict (dict): dictionary of number of ensemble members in the
                            second ensemble (uninitialized). Indexed by season.
        start_year (int): start year of the forecast
        finish_year (int): end year of the forecast
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
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 12),
                            subplot_kw={'projection': proj}, 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Set up the title
    title = 'Total skill and impact of initialization for ' + variable + ' for ' + forecast_range + \
        ' between ' + str(start_year) + ' and ' + str(finish_year) + ' using ' + method + \
        ' method' + 'no_bootstraps = ' + str(no_bootstraps)
    
    # set up the supertitle
    fig.suptitle(title, fontsize=10, y=0.90)

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
