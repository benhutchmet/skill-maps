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
                                            plot_gridbox: dict = None) -> None:
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
    fig.suptitle(title, fontsize=8, y=0.90)

    # If the gridbox is not None
    if gridbox is not None:
        # Set up the lats and lons for this gridbox
        lon1, lon2 = gridbox['lon1'], gridbox['lon2']
        lat1, lat2 = gridbox['lat1'], gridbox['lat2']

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
    for i, method in enumerate(method_list):
        # print("plotting index: ", i, " for method: ", method)

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

        # If the plot_gridbox is not None
        if plot_gridbox is not None:
            # Find the indices of the lats which correspond to the gridbox
            lat1_idx = np.argmin(np.abs(lats - plot_lat1))
            lat2_idx = np.argmin(np.abs(lats - plot_lat2))

            # Find the indices of the lons which correspond to the gridbox
            lon1_idx = np.argmin(np.abs(lons - plot_lon1))
            lon2_idx = np.argmin(np.abs(lons - plot_lon2))

            # Constrain the lats and lon arrays to the gridbox
            lats = lats[lat1_idx:lat2_idx]
            lons = lons[lon1_idx:lon2_idx]

            # Constrain the corr1 array to the gridbox
            corr1 = corr1[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the corr1_p array to the gridbox
            corr1_p = corr1_p[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the partial_r array to the gridbox
            partial_r = partial_r[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Constrain the partial_r_p array to the gridbox
            partial_r_p = partial_r_p[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

        # Set up the axes for the total skill
        ax1 = axs[i, 0]
        ax1.coastlines()
        cf = ax1.contourf(lons, lats, corr1, clevs, cmap='RdBu_r', transform=proj)

        # If the gridbox is not None
        if gridbox is not None:
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

            # Constrain the corr1 array to the gridbox
            corr1_gridbox = corr1[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the mean corr1 over the gridbox
            corr1_mean = np.nanmean(corr1_gridbox)

            # Show this values in the lower left textbox
            ax1.text(0.05, 0.05, f"r = {corr1_mean:.2f}", transform=ax1.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)

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
        cf = ax2.contourf(lons, lats, partial_r, clevs, cmap='RdBu_r', 
                            transform=proj)
        
        # if the gridbox is not None
        if gridbox is not None:
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


            # Constrain the partial_r array to the gridbox
            partial_r_gridbox = partial_r[lat1_idx:lat2_idx, lon1_idx:lon2_idx]

            # Calculate the mean partial_r over the gridbox
            partial_r_mean = np.nanmean(partial_r_gridbox)

            # Show this values in the lower left textbox
            ax2.text(0.05, 0.05, f"r' = {partial_r_mean:.2f}", transform=ax2.transAxes,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)

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
        
        # # add a textbox with the season
        # # to the bottom left of the plot
        # ax2.text(0.05, 0.05, season, transform=ax2.transAxes,
        #             verticalalignment='bottom', horizontalalignment='left',
        #             bbox=dict(facecolor='white', alpha=0.5), fontsize = 8)
        
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

# Define a function which loads the files and plots the different methods
# for the same season and variable
def load_files_and_plot(variable: str, region: str, season: str, forecast_range: str, methods_list: list,
                        no_bootstraps: int, plots_dir: str, bootstrap_base_dir: str,
                        gridbox: dict = None, figsize_x: int = 10, figsize_y: int = 10,
                        plot_gridbox: dict = None) -> None:
    
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
        
    Returns:
    
        None
    """

    # Set up the lists
    paths_list = []

    # Form the paths for the different methods
    for method in methods_list:

        # Set up the path to the file
        path = f"{bootstrap_base_dir}/{variable}/{region}/{season}/" + \
                f"{forecast_range}/{method}/no_bootstraps_{no_bootstraps}"
        
        # Assert that the path exists
        assert os.path.exists(path), f"Path {path} does not exist for method {method}" + \
        f" and no_bootstraps {no_bootstraps}"

        # Append the path to the list
        paths_list.append(path)

    # Loop over the paths to get the values
    values_list = [extract_values_from_txt(path, variable) for path in paths_list]

    # Loop over the paths to get the arrays
    arrays_list = [load_arrays_from_npy(path, variable) for path in paths_list]

    # Plot the different methods
    plot_different_methods_same_season_var(arrays_list, values_list, variable, season,
                                            forecast_range, methods_list, no_bootstraps,
                                            plots_dir, gridbox=gridbox,
                                            figsize_x=figsize_x, figsize_y=figsize_y)
    
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
    plot_names = ['total skill', 'residual corr']

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







