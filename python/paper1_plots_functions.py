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
                                        obs_var_name=obs_var_name)
        
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


