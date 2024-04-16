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
from tqdm import tqdm

# Import cdo
from cdo import Cdo
cdo = Cdo()

# Import dictionaries
sys.path.append("/home/users/benhutch/skill-maps/")
import dictionaries as dicts

# define a function to find where the same files exists
def find_hist_ssp_members(
    variables: list,
    models: list,
    hist_base_path: str = "/gws/nopw/j04/canari/users/benhutch/historical/",
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

    # canari
    # /gws/nopw/j04/canari/users/benhutch/historical/tas/BCC-CSM2-MR/regrid/tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc_global_regrid.nc

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

    # Create an emty dataframe
    df = pd.DataFrame()

    # Loop over variables and models
    for variable in tqdm(variables):
        for model in (models):

            # reset unique members
            unique_model_members_hist = []

            # reset unique members
            unique_model_members_ssp = []

            # Find historical files
            hist_files = glob.glob(
                f"{hist_base_path}{variable}/{model}/regrid/*regrid.nc"
            )

            # Find ssp245 files
            ssp_files = glob.glob(
                f"{ssp_base_path}*/{model}/{ssp}/*/{experiment}/{variable}/g?/files/*/*.nc"
            )

            # Create an empty list of unique members
            unique_members = []

            # Loop over historical files
            for hist_file in hist_files:

                # sfcWind_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc
                # Extract the final part of the path
                hist_file_nc = hist_file.split("/")[-1]

                # Use re to find the pattern matching r*i?p?f?
                hist_member_id = hist_file_nc.split("_")[4]

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
                ssp_member_id = ssp_file_nc.split("_")[4]

                # assert that ssp_member_id is not none
                assert ssp_member_id is not None

                # if f"{model}_{ssp_member_id}" not in unique_model_members_ssp:
                if ssp_member_id not in unique_model_members_ssp:
                    unique_model_members_ssp.append(ssp_member_id)

                # Append to ssp_paths
                ssp_paths.append(ssp_file)

            # print the hist and ssp members
            print(f"hist_members {unique_model_members_hist} for {model}")
            print(f"ssp_members {unique_model_members_ssp} for {model}")

            # Find where the same members exist in both historical and ssp245
            common_members = list(set(unique_model_members_hist) & set(unique_model_members_ssp))

            # Print the common members
            print(f"common members {common_members} for {model} and {variable}")

            # Loop over common members
            for common_member in common_members:

                # Find the index of the common member in hist_member_id
                hist_member = glob.glob(
                f"{hist_base_path}{variable}/{model}/regrid/*{common_member}*regrid.nc"
                )

                # Find the index of the common member in ssp_member_id
                ssp_member = glob.glob(
                    f"{ssp_base_path}*/{model}/{ssp}/{common_member}/{experiment}/{variable}/g?/files/*/*.nc"
                )[0]

                # assert that only a single member
                assert len(hist_member) == 1, f"hist_member {hist_member} is not unique"

                # Append to the dataframe
                # usinf pd.concat
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "model": [model],
                                "variable": [variable],
                                "common_member": [common_member],
                                "hist_member": [hist_member[0]],
                                "ssp_member": [ssp_member],
                            }
                        ),
                    ]
                )

    # Return the dataframe
    return df

# Write a function which regrids the ssp data
def regrid_ssp(
    df: pd.DataFrame,
    models: list,
    ssp: str = "ssp245",
    region: str = "global",
    gridspec_file: str = "/home/users/benhutch/gridspec/gridspec-global.txt",
    output_dir: str = "/work/scratch-nopw2/benhutch",
):
    """
    Regrids SSP data to the same grid as the historical data.
    Using bilinear interpolation.
    
    Args:
        df (pd.DataFrame): DataFrame of where identical experiment set ups exist
        models (list): list of models to search for
        ssp (str): ssp scenario
        region (str): region to regrid to
        gridspec_file (str): path to gridspec file
        output_dir (str): path to output directory
        
    Returns:
        None
    """

    # assert that the gridspec file exists
    assert os.path.exists(gridspec_file), f"{gridspec_file} does not exist"

    # Loop over models
    for model in tqdm(models):
        # Assert that the model is in the dataframe
        assert model in df["model"].values, f"{model} not in df"

        # print the common members for the model
        common_members = df[df["model"] == model]["common_member"].values

        # print these
        print(f"common_members {common_members} for {model}")

        # extract the rows for the model
        model_df = df[df["model"] == model]

        # Loop over the rows
        for _, row in model_df.iterrows():
            # Form the output path
            output_dir = f"{output_dir}/{row['variable']}/{model}/{region}/{ssp}/"

            # /badc/cmip6/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp245/r1i1p1f1/Amon/tas/gn/files/d20190429/tas_Amon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.nc

            # form the output file name
            output_file = row["ssp_member"].split("/")[-1].replace('.nc', '_regrid.nc')

            # form the output path
            output_path = os.path.join(output_dir, output_file)

            # if the output dir does not exist, create it
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # if the output file does not exist, regrid
            if not os.path.exists(output_path):
                try:
                    # regrid the data
                    cdo.remapbil(gridspec_file, input=row["ssp_member"], output=output_path)
                except Exception as e:
                    print(f"Error regridding {row['ssp_member']} {e}")
            else:
                print(f"{output_path} exists")

            # add a new column to the df
            df.loc[df["ssp_member"] == row["ssp_member"], "ssp_member_regrid"] = output_path

    # return the dataframe
    return df
            


            
