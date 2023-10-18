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
                                            figsize_x: int = 10, figsize_y: int = 12) -> None:
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

        figsize_x (int): size of the figure in the x direction. Default is 10.

        figsize_y (int): size of the figure in the y direction. Default is 12.

    Returns:

        None

    """

    # Set up the axis labels
    ax_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Set up the plot_names
    plot_names = ['total skill', 'residual corr']

    # Set up the projection
    proj = ccrs.PlateCarree()

    # Set up the figure size
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(figsize_x, figsize_y),
                            subplot_kw={'projection': proj}, 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    
    # Extract a start year and finish year from the values_list
    start_year = values[0]['start_year']
    finish_year = values[0]['end_year']

    # Set up the title
    title = 'Total skill and impact of initialization for ' + variable + ' in ' + season + ' for ' + forecast_range + \
        ' between ' + str(start_year) + ' and ' + str(finish_year) + \
        ' using different methods' + 'no_bootstraps = ' + str(no_bootstraps)
    
    # set up the supertitle
    fig.suptitle(title, fontsize=6, y=0.90)

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
    for i, method in enumerate(method_list):
        print("plotting index: ", i, " for method: ", method)

        # Extract the dictionaries for this method
        method_arrays = arrays[i]
        method_values = values[i]

        # From the dictionaries, extract the arrays
        corr1 = method_arrays['corr1']
        corr1_p = method_arrays['corr1_p']
        partial_r = method_arrays['partial_r']
        partial_r_p = method_arrays['partial_r_p']

        # From the dictionaries, extract the values
        nens1 = method_values['nens1']
        nens2 = method_values['nens2']
        start_year = method_values['start_year']
        end_year = method_values['end_year']

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
        
        # add a textbox with the season
        # to the bottom left of the plot
        ax1.text(0.05, 0.05, season, transform=ax1.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
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
        
        # add a textbox with the season
        # to the bottom left of the plot
        ax2.text(0.05, 0.05, season, transform=ax2.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
    # Add a colorbar for the correlation
    cbar = fig.colorbar(cf_list[0], ax=axs, orientation='horizontal', pad = 0.05,
                        aspect=50, shrink=0.8)
    cbar.set_label('correlation coefficient')

    # Set up the pathname for saving the figure
    fig_name = f"{plots_dir}/raw_init_impact_{variable}_{season}_" + \
    f"{forecast_range}_different_methods_{no_bootstraps}_{start_year}" + \
    f"_{finish_year}.png"

    fig_path = os.path.join(plots_dir, fig_name)

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # show the figure
    plt.show()
