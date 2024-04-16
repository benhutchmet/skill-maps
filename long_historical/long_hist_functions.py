"""
Functions for processing and creating long historical period data.

Ben Hutchins

April 2024

To create historical time series which are valid against the dcpp experiments
which run past 2014, we have to combine the historical runs from CMIP
(1850-2014) with the ssp245 runs from ScenarioMIP (2015-2100).

To string the time series together, they should have the same r*i?p?f? 
(experimental set up).

We want to avoid downloading data where possible, so we want to see how many of
these files are available in /badc/ on JASMIN.
"""

# Import local modules
import os
import re
import sys
import glob

# Import third party modules
import numpy as np
import pandas as pd
import xarray as xr

# Import dictionaries
sys.path.append("/home/users/benhutch/skill-maps/")
import dictionaries as dicts

# define a function to find where the same files exists
def find_hist_ssp_members(
    variables: list,
    models: list,
    hist_base_path: str = "/badc/cmip6/data/CMIP6/CMIP/",
    ssp_base_path: str = "/badc/cmip6/data/CMIP6/ScenarioMIP/",
    ssp: str = "ssp245",
    experiment: str = "Amon",
):
    """
    Finds where identical experiment set ups (r*i?p?f?) exist for both historical
    and ssp245 runs.

    Args:
        variables (list): list of variables to search for
        models (list): list of models to search for
        hist_base_path (str): path to historical data
        ssp_base_path (str): path to ssp245 data

    Returns:
        df (pd.DataFrame): DataFrame of where identical experiment set ups exist

    """
    # badc
    # /badc/cmip6/data/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/sfcWind/gn/files/d20181126

    # scenariomip
    # /badc/cmip6/data/CMIP6/ScenarioMIP/BCC/BCC-CSM2-MR/ssp245/r1i1p1f1/Amon/sfcWind/gn/files/d20190314/

    # unique model members list
    unique_model_members_hist = []

    # unique model members ssp
    unique_model_members_ssp = []

    # hist_paths
    hist_paths = []

    # ssp_paths
    ssp_paths = []



    # Loop over variables and models
    for variable, model in zip(variables, models):

        # Find historical files
        hist_files = glob.glob(
            f"{hist_base_path}/*/{model}/historical/*/{experiment}/{variable}/g?/files/*/*.nc"
        )

        # Find ssp245 files
        ssp_files = glob.glob(
            f"{ssp_base_path}/*/{model}/{ssp}/*/{experiment}/{variable}/g?/files/*/*.nc"
        )

        # Create an empty list of unique members
        unique_members = []

        # Loop over historical files
        for hist_file in hist_files:

            # sfcWind_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc
            # Extract the final part of the path
            hist_file_nc = hist_file.split("/")[-1]

            # Use re to find the pattern matching r*i?p?f?
            hist_member_id = re.search(r"r\d+i\d+p\d+f\d+", hist_file_nc).group()

            # assert that hist_member_id is not none
            assert hist_member_id is not None

            # if f"{model}_{hist_member_id}" not in unique_model_members_hist:
            if hist_member_id not in unique_model_members_hist:
                unique_model_members_hist.append(hist_member_id)

            # Append to hist_paths
            hist_paths.append(hist_file)

        # Loop over ssp245 files
        for ssp_file in ssp_files:

            # sfcWind_Amon_BCC-CSM2-MR_ssp245_r1i1p1f1_gn_201501-210012.nc
            # Extract the final part of the path
            ssp_file_nc = ssp_file.split("/")[-1]

            # Use re to find the pattern matching r*i?p?f?
            ssp_member_id = re.search(r"r\d+i\d+p\d+f\d+", ssp_file_nc).group()

            # assert that ssp_member_id is not none
            assert ssp_member_id is not None

            # if f"{model}_{ssp_member_id}" not in unique_model_members_ssp:
            if ssp_member_id not in unique_model_members_ssp:
                unique_model_members_ssp.append(ssp_member_id)

            # Append to ssp_paths
            ssp_paths.append(ssp_file)

            