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
import shutil

# Import third party modules
import numpy as np
import pandas as pd
import xarray as xr
import cftime
from tqdm import tqdm

# Import cdo
from cdo import Cdo

cdo = Cdo()

# Import dictionaries
sys.path.append("/home/users/benhutch/skill-maps/")
import dictionaries as dicts


# Define a function to merge the files stored on badc
def merge_regrid_hist_files(
    variables: list,
    models: list,
    experiment: str = "Amon",
    region: str = "global",
    hist_base_path: str = "/badc/cmip6/data/CMIP6/CMIP/",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical/",
    temp_dir: str = "/work/scratch-nopw2/benhutch/",
    gridspec_file: str = "/home/users/benhutch/gridspec/gridspec-global.txt",
    fname: str = "hist_files.csv",
):
    """
    Merges the historical files stored on badc.

    Args:
        variables (list): list of variables to search for
        models (list): list of models to search for
        experiment (str): experiment type
        region (str): region to regrid to
        hist_base_path (str): path to historical data
        save_dir (str): path to save the csv
        temp_dir (str): path to temporary directory
        gridspec_file (str): path to gridspec file
        fname (str): filename of the csv

    Returns:
        df (pd.DataFrame): DataFrame of where identical experiment set ups exist

    """

    # if the save_dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # badc
    # /badc/cmip6/data/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/sfcWind/gn/files/d20181126
    # /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r1i1p1f3/Amon/pr/gn/files/d20191207/

    # Create an empty dataframe
    df = pd.DataFrame()

    # loop over the variables and models
    for variable in variables:
        for model in tqdm(models, desc="Merging files"):
            # Find all of the valid directories
            hist_dirs = glob.glob(
                f"{hist_base_path}*/{model}/historical/r*i?p?f?/{experiment}/{variable}/g?/files/*/"
            )

            if len(hist_dirs) == 0:
                print(f"No directories found for {model} and {variable}")
                continue

            # Loop over the dirs
            for dir in hist_dirs:
                # List the files in the directory
                files = glob.glob(f"{dir}*.nc")

                # assert that the len of files is 1 or greater
                assert len(files) >= 1, f"{files} is not 1 or greater"

                # /work/scratch-nopw2/benhutch/pr/BCC-CSM2-MR/global/
                # Set up the output_dir
                output_dir = f"{temp_dir}{variable}/{model}/{region}/historical_temp/"

                # if the output_dir does not exist, create it
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if len(files) > 1:
                    # Extract the filenames
                    filenames = [file.split("/")[-1] for file in files]

                    # Extract all sequences of 6 digits from the filenames
                    sequences = [
                        re.findall(r"\d{6}", filename) for filename in filenames
                    ]

                    # Flatten the list of sequences
                    sequences = [seq for sublist in sequences for seq in sublist]

                    # Convert the sequences to integers
                    sequences = [int(seq) for seq in sequences]

                    # Find the smallest and largest sequences
                    smallest_seq = min(sequences)
                    largest_seq = max(sequences)

                    # Extract the first fname
                    first_fname = filenames[0]

                    # Replace _185001-186912.nc or _??????-??????.nc
                    # with _{smallest_seq}-{largest_seq}.nc
                    output_fname = first_fname.replace(
                        first_fname.split("_")[-1], f"{smallest_seq}-{largest_seq}.nc"
                    )

                    # form the output path
                    output_path = os.path.join(output_dir, output_fname)

                    # if the output file does not exist, merge
                    if not os.path.exists(output_path):
                        try:
                            # merge the files
                            cdo.mergetime(input=files, output=output_path)
                        except Exception as e:
                            print(f"Error merging {files} {e}")

                    # add to the variable, model, merged_file columns
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "variable": [variable],
                                    "model": [model],
                                    "merged_file": [output_path],
                                }
                            ),
                        ]
                    )

                else:
                    # print that only one file exists
                    print(f"Only one file exists for {model} and {variable}")
                    print(f"copying {files[0]} to {output_dir}")

                    # if the file does not already exist in the output dir
                    if not os.path.exists(
                        os.path.join(output_dir, os.path.basename(files[0]))
                    ):
                        # copy the file to the output_dir
                        shutil.copy(files[0], output_dir)

                    # add to the variable, model, merged_file columns
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "variable": [variable],
                                    "model": [model],
                                    "merged_file": [files[0]],
                                }
                            ),
                        ]
                    )

    # With the dataframe, loop over the paths in ['merged_file']
    # and regrid these to the global grid
    # Iterate over the rows in the dataframe
    for _, row in tqdm(df.iterrows(), desc="Regridding files"):
        # Set up the directory
        regrid_dir = f"{save_dir}{row['variable']}/{row['model']}/regrid/"

        # if the regrid_dir does not exist, create it
        if not os.path.exists(regrid_dir):
            os.makedirs(regrid_dir)

        # Set up the output fname
        # Extract the fname from the 'merged_file'
        regrid_fname = row["merged_file"].split("/")[-1].replace(".nc", "_regrid.nc")

        # Set up the output path
        regrid_path = os.path.join(regrid_dir, regrid_fname)

        # if the regrid path already exists
        if not os.path.exists(regrid_path):
            # regrid the file
            try:
                cdo.remapbil(
                    gridspec_file, input=row["merged_file"], output=regrid_path
                )
            except Exception as e:
                print(f"Error regridding {row['merged_file']} {e}")

        # Add a new column to the dataframe
        df.loc[df["merged_file"] == row["merged_file"], "regrid_file"] = regrid_path

    # Return the dataframe
    return df


# define a function to find where the same files exists
def find_hist_ssp_members(
    variables: list,
    models: list,
    hist_base_path: str = "/gws/nopw/j04/canari/users/benhutch/historical/",
    ssp_base_path: str = "/badc/cmip6/data/CMIP6/ScenarioMIP/",
    ssp: str = "ssp245",
    experiment: str = "Amon",
    save_dir: str = "gws/nopw/j04/canari/users/benhutch/ssp245/",
    fname: str = "hist_ssp_members.csv",
):
    """
    Finds where identical experiment set ups (r*i?p?f?) exist for both historical
    and ssp245 runs.

    Args:
        variables (list): list of variables to search for
        models (list): list of models to search for
        hist_base_path (str): path to historical data
        ssp_base_path (str): path to ssp245 data
        ssp (str): ssp scenario
        experiment (str): experiment type
        save_dir (str): path to save the csv
        fname (str): filename of the csv

    Returns:
        df (pd.DataFrame): DataFrame of where identical experiment set ups exist

    """
    # badc
    # /badc/cmip6/data/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/sfcWind/gn/files/d20181126
    # /badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r1i1p1f3/Amon/pr/gn/files/d20191207/

    # canari
    # /gws/nopw/j04/canari/users/benhutch/historical/tas/BCC-CSM2-MR/regrid/tas_Amon_BCC-CSM2-MR_historical_r1i1p1f1_gn_185001-201412.nc_global_regrid.nc

    # scenariomip
    # /badc/cmip6/data/CMIP6/ScenarioMIP/BCC/BCC-CSM2-MR/ssp245/r1i1p1f1/Amon/sfcWind/gn/files/d20190314/

    # if the save_dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up the filename
    filepath = f"{save_dir}{fname}"

    # Check if the file exists
    if not os.path.exists(filepath):
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
            for model in models:

                # reset unique members
                unique_model_members_hist = []

                # reset unique members
                unique_model_members_ssp = []

                if variable == "ua":
                    hist_files = glob.glob(
                        f"{hist_base_path}{variable}/{model}/regrid/*global*regrid.nc"
                    )
                else:
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
                common_members = list(
                    set(unique_model_members_hist) & set(unique_model_members_ssp)
                )

                # Print the common members
                print(f"common members {common_members} for {model} and {variable}")

                # Loop over common members
                for common_member in common_members:
                    
                    if variable == "ua":
                        print("Variable is ua")

                        # Find the index of the common member in hist_member_id
                        hist_member = glob.glob(
                            f"{hist_base_path}{variable}/{model}/regrid/*{common_member}*global*regrid.nc"
                        )

                    else:
                        print("Variable is not ua")
                        # Find the index of the common member in hist_member_id
                        hist_member = glob.glob(
                            f"{hist_base_path}{variable}/{model}/regrid/*{common_member}*regrid.nc"
                        )

                    # print the path
                    print(f"hist_member {hist_member}")

                    # Find the index of the common member in ssp_member_id
                    ssp_member = glob.glob(
                        f"{ssp_base_path}*/{model}/{ssp}/{common_member}/{experiment}/{variable}/g?/files/*/*.nc"
                    )[0]

                    # assert that only a single member
                    assert (
                        len(hist_member) == 1
                    ), f"hist_member {hist_member} is not unique"

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

        # Save the dataframe
        df.to_csv(f"{save_dir}/{fname}", index=False)
    else:
        # Read the dataframe
        df = pd.read_csv(filepath)

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
        for _, row in tqdm(model_df.iterrows()):
            # Form the output path
            output_dir_model = f"{output_dir}/{row['variable']}/{model}/{region}/{ssp}/"

            # /badc/cmip6/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp245/r1i1p1f1/Amon/tas/gn/files/d20190429/tas_Amon_CanESM5_ssp245_r1i1p1f1_gn_201501-210012.nc

            # extrcat the filename
            ssp_fname = row["ssp_member"].split("/")[-1]

            # split by _ and extract the 6th
            year_range_nc = ssp_fname.split("_")[6]

            # split by . and extract the 0th
            year_range = year_range_nc.split(".")[0]

            # Extract the first yyyymm
            first_yyyymm = year_range.split("-")[0]
            last_yyyymm = year_range.split("-")[1]

            if last_yyyymm < "202303":
                # replace the yyyymm.nc with *.nc
                files = row["ssp_member"].replace(year_range_nc, "*.nc")

                # extract the first two files using glob
                files = glob.glob(files)[:10]

                # print the first two
                print(f"merging {files}")

                # split the second file by /
                second_file = files[-1].split("/")[-1]
                second_file = second_file.split("_")[6]
                second_file = second_file.split(".")[0]
                second_file_last_yyyymm = second_file.split("-")[1]

                # merge the two files
                output_dir_temp = os.path.join(
                    "/work/scratch-nopw2/benhutch/", "temp", ssp, row["variable"]
                )

                # if the output dir does not exist, create it
                if not os.path.exists(output_dir_temp):
                    os.makedirs(output_dir_temp)

                # Set up the output fname
                output_fname = ssp_fname.replace(
                    year_range_nc, f"{first_yyyymm}-{second_file_last_yyyymm}.nc"
                )

                # Set up the output path
                output_path = os.path.join(output_dir_temp, output_fname)

                # if the output file does not exist, merge
                if not os.path.exists(output_path):
                    try:
                        # merge the two files
                        cdo.mergetime(input=files, output=output_path)
                    except Exception as e:
                        print(f"Error merging {files} {e}")

                # regrid the data
                output_file = output_fname.replace(".nc", "_regrid.nc")

                # output dir
                output_dir_rg = f"/work/scratch-nopw2/benhutch/{row['variable']}/{model}/{region}/{ssp}/"

                # if the output dir does not exist, create it
                if not os.path.exists(output_dir_rg):
                    os.makedirs(output_dir_rg)

                # form the output path
                output_path_rg = os.path.join(output_dir_rg, output_file)

                # if the output file does not exist, regrid
                if not os.path.exists(output_path_rg):
                    try:
                        # regrid the data
                        cdo.remapbil(
                            gridspec_file, input=output_path, output=output_path_rg
                        )
                    except Exception as e:
                        print(f"Error regridding {output_path} {e}")

                # Set the ssp_member_regrid column
                df.loc[df["ssp_member"] == row["ssp_member"], "ssp_member_regrid"] = (
                    output_path_rg
                )
            else:
                # form the output file name
                output_file = (
                    row["ssp_member"].split("/")[-1].replace(".nc", "_regrid.nc")
                )

                # form the output path
                output_path = os.path.join(output_dir_model, output_file)

                # if the output dir does not exist, create it
                if not os.path.exists(output_dir_model):
                    os.makedirs(output_dir_model)

                # if the output file does not exist, regrid
                if not os.path.exists(output_path):
                    try:
                        # regrid the data
                        cdo.remapbil(
                            gridspec_file, input=row["ssp_member"], output=output_path
                        )
                    except Exception as e:
                        print(f"Error regridding {row['ssp_member']} {e}")
                else:
                    print(f"{output_path} exists")

                # add a new column to the df
                df.loc[df["ssp_member"] == row["ssp_member"], "ssp_member_regrid"] = (
                    output_path
                )

    # return the dataframe
    return df


# Write a function to merge the historical and ssp data
def merge_hist_ssp(
    df: pd.DataFrame,
    models: list,
    hist_base_path: str = "/gws/nopw/j04/canari/users/benhutch/historical/",
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/",
):
    """
    Merges the historical and ssp data.

    Args:
        df (pd.DataFrame): DataFrame of where identical experiment set ups exist
        models (list): list of models to search for
        hist_base_path (str): path to historical data
        output_dir (str): path to output directory

    Returns:
        None
    """

    # if the output dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over models
    for model in tqdm(models):
        # Assert that the model is in the dataframe
        assert model in df["model"].values, f"{model} not in df"

        # extract the rows for the model
        model_df = df[df["model"] == model]

        # Loop over the rows
        for _, row in tqdm(model_df.iterrows()):
            # assert that the hist_member exists
            assert os.path.exists(
                row["hist_member"]
            ), f"{row['hist_member']} does not exist"

            # assert that the ssp_member_regrid exists
            assert os.path.exists(
                row["ssp_member_regrid"]
            ), f"{row['ssp_member_regrid']} does not exist"

            # set up the output dir
            output_dir_model = f"{output_dir}/{row['variable']}/{model}/"

            # if the output dir does not exist, create it
            if not os.path.exists(output_dir_model):
                os.makedirs(output_dir_model)

            # split the hist_member by /
            hist_fname = row["hist_member"].split("/")[-1]

            # split the ssp_member_regrid by /
            ssp_fname = row["ssp_member_regrid"].split("/")[-1]

            # Split by _ and extract the 6th
            year_range_nc_hist = hist_fname.split("_")[6]

            # Split by _ and extract the 6th for ssp
            year_range_nc_ssp = ssp_fname.split("_")[6]

            # Split by . and extract the 0th
            year_range_hist = year_range_nc_hist.split(".")[0]

            # Split by - and extract the 0th
            last_yyyymm_ssp = year_range_nc_ssp.split("-")[1]

            # Extract the first yyyymm
            first_yyyymm_hist = year_range_hist.split("-")[0]

            # assert that the first yyyymm is digits
            assert first_yyyymm_hist.isdigit(), f"{first_yyyymm_hist} is not digits"

            # assert that the last yyyymm is digits
            assert last_yyyymm_ssp.isdigit(), f"{last_yyyymm_ssp} is not digits"

            # print the hist and ssp members
            print(f"hist_member {hist_fname} for {model}")
            print(f"ssp_member_regrid {ssp_fname} for {model}")

            # print the first yyyymm and last yyyymm
            print(f"first_yyyymm_hist {first_yyyymm_hist} for {model}")
            print(f"last_yyyymm_ssp {last_yyyymm_ssp} for {model}")

            if int(first_yyyymm_hist[:4]) > 1960:
                print(
                    f"{model} {row['variable']} {first_yyyymm_hist} {last_yyyymm_ssp}"
                )
                print(
                    f"value of first_yyyymm_hist {int(first_yyyymm_hist[:4])} is greater than 1960"
                )

                # raise a value error
                print(
                    f"{model} {row['variable']} {first_yyyymm_hist} {last_yyyymm_ssp} greater than 1960"
                )

                # print that we are continuing to the next
                print(f"continuing to the next {model} {row['variable']}")
                continue

            # Set up the output fname
            # tas_Amon_MPI-ESM1-2-HR_historical_r2i1p1f1_g?_1850-2014.nc_global_regrid.nc for MPI-ESM1-2-HR
            hist_fname_parts = hist_fname.split("_")

            # Get the first 5 elements
            hist_fname_parts = hist_fname_parts[:5]

            # join the first 5 elements
            hist_fname_parts = "_".join(hist_fname_parts)

            # Set up the output fname
            output_fname = (
                f"{hist_fname_parts}_{first_yyyymm_hist}-{last_yyyymm_ssp}.nc"
            )

            # print the output fname
            print(f"output_fname {output_fname}")

            # Set up the output path
            output_path = os.path.join(output_dir_model, output_fname)

            # if the output file does not exist, merge
            if not os.path.exists(output_path):
                try:
                    # merge the two files
                    cdo.mergetime(
                        input=[row["hist_member"], row["ssp_member_regrid"]],
                        output=output_path,
                    )
                except Exception as e:
                    print(
                        f"Error merging {row['hist_member']} {row['ssp_member_regrid']} {e}"
                    )

            # Set the hist_ssp_member column
            df.loc[df["hist_member"] == row["hist_member"], "hist_ssp_member"] = (
                output_path
            )

    # return the dataframe
    return df


# Write a function to process the merged data
def process_hist_ssp(
    variable: str,
    months: list[int],
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    models: list,
    data_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/",
    level: int = 0,
    lag: int = 0,
):
    """
    Processes the merged historical and ssp data.

    Args:
        variable: str: variable to process
        months: list[int]: months to process
        season: str: season to process
        forecast_range: str: forecast range to process
        start_year: int: start year to process
        end_year: int: end year to process
        models: list: list of models to process
        data_dir: str: path to data directory
        level: int: level to process
        lag: int: lag to process

    Returns:
        None
    """

    # assert that months is a list of ints
    assert all(isinstance(i, int) for i in months), f"{months} is not a list of ints"

    # empty list of files
    all_files = []

    # loop over the models
    for model in tqdm(models):
        # form the path to the data
        data_path = f"{data_dir}{variable}/{model}/"

        # assert that this directory is not empty
        assert os.path.exists(data_path), f"{data_path} does not exist"

        # find the files
        model_files = glob.glob(f"{data_path}*.nc")

        # extend the all_files
        all_files.extend(model_files)

    # print the all_files
    print(f"all_files {all_files}")
    print(f"len(all_files) {len(all_files)}")

    # initialize model data
    dss = []

    # Set up the start and end month
    start_month = months[0]
    end_month = months[-1]

    # Loop over the files
    for file in tqdm(all_files, desc="Processing files"):

        if lag != 0:
            for lag_idx in range(lag):
                # Open the files
                ds = xr.open_mfdataset(
                    file,
                    preprocess=lambda ds: preprocess(
                        ds,
                        start_year,
                        end_year,
                        start_month,
                        end_month,
                        months,
                        season,
                        forecast_range,
                        lag=True,
                        level=level,
                        lag_idx=lag_idx,
                    ),
                    combine="nested",
                    concat_dim="time",
                    join="override",
                    coords="minimal",
                    engine="netcdf4",
                    parallel=True,
                )

                # append to dss
                dss.append(ds)

        else:
            # Open the files
            ds = xr.open_mfdataset(
                file,
                preprocess=lambda ds: preprocess(
                    ds,
                    start_year,
                    end_year,
                    start_month,
                    end_month,
                    months,
                    season,
                    forecast_range,
                    level=level,
                ),
                combine="nested",
                concat_dim="time",
                join="override",
                coords="minimal",
                engine="netcdf4",
                parallel=True,
            )

            # append to dss
            dss.append(ds)

    # Concatenate the dss
    ds = xr.concat(dss, dim="ensemble_member")

    # # Initialize the model data
    dss_anoms = []

    # print the coordinates of the dataset
    print(f"Coordinates: {ds.coords}")

    # Print the dimensions of the dataset
    print(f"Dimensions: {ds.dims}")

    # # extract the models
    # by finding the unique values of the ensemble_member
    # when ensemble member is split by _
    models = np.unique([m.split("_")[0] for m in ds["ensemble_member"].values])

    # print the models
    print(f"Models: {models}")

    # Loop over the model dimension within ds
    for model in tqdm(models, desc="Calculating anomalies"):
        print(f"Model: {model}")

        # Select the ensemble members which contain
        # the model in the first part of the string
        ds_model = ds.sel(
            ensemble_member=[
                m for m in ds["ensemble_member"].values if m.split("_")[0] == model
            ]
        )

        # # print the ds_model
        # print(f"ds_model: {ds_model} for {model}")

        # Calculate the mean over ensemble_member and time
        ds_mean = ds_model.mean(dim=["ensemble_member", "time"])

        # Loop over the ensemble_member dimension
        for member in ds_model["ensemble_member"].values:
            # print the member
            print(f"Member: {member}")

            # Calculate the anomalies
            ds_anom = ds_model.sel(ensemble_member=member) - ds_mean

            # Append to the list
            dss_anoms.append(ds_anom)

    # Concatenate the dss_anoms
    ds_anoms = xr.concat(dss_anoms, dim="ensemble_member")

    return ds, ds_anoms


# TODO: optional lag argument for historical data
# define the preprocess function
def preprocess(
    ds: xr.Dataset,
    start_year: int,
    end_year: int,
    start_month: int,
    end_month: int,
    months: list[int],
    season: str,
    forecast_range: str,
    centred: bool = True,
    lag: bool = False,
    level: int = 0,
    lag_idx: int = 0,
):
    """
    Preprocesses the data to include an ensemble_member dimension.
    And a model dimension.

    Args:
        ds (xr.Dataset): Dataset to preprocess
        start_year (int): start year to process
        end_year (int): end year to process
        start_month (int): start month to process
        end_month (int): end month to process
        months (list[int]): months to process
        season (str): season to process
        forecast_range (str): forecast range to process
        centred (bool): centred or not for rolling mean
        lag (bool): lag to process (optional)
        level (int): level to process
        lag_idx (int): lag to process (optional)

    Returns:
        ds (xr.Dataset): Preprocessed Dataset
    """

    # Expand dimensions to include ensemble member
    ds = ds.expand_dims("ensemble_member")

    if lag is True:
        # Shift the dataset if necessary
        ds["ensemble_member"] = [
            f"{ds.attrs['source_id']}_{ds.attrs['variant_label']}_{lag_idx}"
        ]
    else:
        # Set the ensemble_member
        ds["ensemble_member"] = [f"{ds.attrs['source_id']}_{ds.attrs['variant_label']}"]

    # if level is not 0
    if level != 0:
        try:
            # Select the level
            ds = ds.sel(plev=level)
        except Exception as e:
            print(f"Error selecting level {e}")

    # If the start or end month is a single digit
    if start_month < 10:
        start_month = f"0{start_month}"

    if end_month < 10:
        end_month = f"0{end_month}"

    # Set the start and end date
    start_date = f"{start_year}-{start_month}-01"
    end_date = f"{end_year}-{end_month}-30"

    # Select the time range
    ds = ds.sel(time=slice(start_date, end_date))

    # Select the months
    ds = ds.sel(time=ds["time.month"].isin(months))

    # Shift the dataset if necessary
    if season in ["DJFM", "NDJFM", "ONDJFM"]:
        ds = ds.shift(time=-3)
    elif season in ["DJF", "NDJF", "ONDJF"]:
        ds = ds.shift(time=-2)
    elif season in ["NDJ", "ONDJ"]:
        ds = ds.shift(time=-1)
    else:
        ds = ds

    # Calculate the annual mean anomalies
    ds = ds.resample(time="Y").mean("time")

    # Calculate the window
    ff_year = int(forecast_range.split("-")[0])
    lf_year = int(forecast_range.split("-")[1])

    # Calculate the window
    window = (lf_year - ff_year) + 1  # e.g. (9 - 2) + 1 = 8

    # Calculate the rolling mean
    ds = ds.rolling(time=window, center=centred).mean()

    # if the time type is not cftime.DatetimeNoLeap
    if not isinstance(ds["time"].values[0], cftime.DatetimeNoLeap):
        # Convert the time index to cftime.DatetimeNoLeap
        ds["time"] = (
            ds["time"]
            .to_series()
            .map(lambda x: cftime.DatetimeNoLeap(x.year, x.month, x.day))
        )

    # if lag is true
    if lag is True:
        # Shift the dataset if necessary
        ds = ds.shift(time=lag_idx)

    # Return the dataset
    return ds


# define a function to save the data as a .nc file
def save_data(
    ds: xr.Dataset,
    variable: str,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    ssp: str = "ssp245",
    data_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/saved_files/",
    lag: int = 0,
):
    """
    Saves the data as a .nc file.

    Args:
        ds (xr.Dataset): Dataset to save
        variable (str): variable to save
        season (str): season to save
        forecast_range (str): forecast range to save
        start_year (int): start year to save
        end_year (int): end year to save
        data_dir (str): path to data directory
        lag (int): lag to save

    Returns:
        None
    """

    # if the data_dir does not exist, create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Set up the output dir
    output_dir = f"{data_dir}{variable}/{season}/{forecast_range}/"

    # if the output dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if lag != 0:
        # Set up the output file
        output_file = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_lag_{lag}_historical_{ssp}.nc"
    else:
        # Set up the output file
        output_file = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_historical_ssp245.nc"

    # Set up the output path
    output_path = os.path.join(output_dir, output_file)

    # Save the dataset
    ds.to_netcdf(output_path)

    # Print the output path
    print(f"Saved to {output_path}")

    return None


# define a function to process the data
# to be the same length as the model data and observations
# then convert to an array (+ save this array)
def constrain_to_arr(
    ds: xr.Dataset,
    variable: str,
    season: str,
    forecast_range: str,
    start_year: int,
    end_year: int,
    lag_first_year: int,
    lag_last_year: int,
    raw_first_year: int,
    raw_last_year: int,
    ssp: str = "ssp245",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/historical_ssp245/saved_files/arrays/",
    lag: int = 0,
):
    """
    Constrains the data to the same length as the model data and observations.
    Then converts to an array and saves this array.

    Args:
        ds (xr.Dataset): Dataset to constrain
        variable (str): variable to constrain
        season (str): season to constrain
        forecast_range (str): forecast range to constrain
        start_year (int): start year to constrain
        end_year (int): end year to constrain
        lag_first_year (int): lag first year to constrain
        lag_last_year (int): lag last year to constrain
        raw_first_year (int): raw first year to constrain
        raw_last_year (int): raw last year to constrain
        save_dir (str): path to save directory
        lag (int): lag to constrain

    Returns:
        None
    """

    # if the save_dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up the output dir
    output_dir = f"{save_dir}{variable}/{season}/{forecast_range}/"

    # if the output dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if lag != 0:
        # Set up the output file
        output_file_raw = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_lag_{lag}_historical_{ssp}_raw.npy"

        # Set up the output file
        output_file_lag = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_lag_{lag}_historical_{ssp}_lag.npy"
    else:
        # Set up the output file
        output_file_raw = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_historical_ssp245_raw.npy"

        # Set up the output file
        output_file_lag = f"{variable}_{season}_{forecast_range}_{start_year}-{end_year}_historical_ssp245_lag.npy"

    # Set up the output path
    output_path_raw = os.path.join(output_dir, output_file_raw)

    # Set up the output path
    output_path_lag = os.path.join(output_dir, output_file_lag)

    # process the ds_lag
    ds_lag = ds.sel(time=slice(f"{lag_first_year}-01-01", f"{lag_last_year}-12-31"))

    # process the ds_raw
    ds_raw = ds.sel(time=slice(f"{raw_first_year}-01-01", f"{raw_last_year}-12-31"))

    # # Check for NaNs
    # check_nans(ds_lag)

    # # Check for NaNs
    # check_nans(ds_raw)

    # print
    print(f"ds_lag: {ds_lag}")
    print(f"ds_raw: {ds_raw}")

    # print the types
    print(f"ds_lag type: {type(ds_lag)}")
    print(f"ds_raw type: {type(ds_raw)}")

    # Convert to an array
    arr_lag = ds_lag[variable].values

    # Convert to an array
    arr_raw = ds_raw[variable].values

    # print the shapes of the arrays
    print(f"arr_lag shape: {arr_lag.shape}")

    # print the shapes of the arrays
    print(f"arr_raw shape: {arr_raw.shape}")

    # Save the array
    np.save(output_path_lag, arr_lag)

    # Save the array
    np.save(output_path_raw, arr_raw)

    # Print the output path
    print(f"Saved to {output_path_lag}")

    # Print the output path
    print(f"Saved to {output_path_raw}")

    return None


# define a function to check for Nan values
def check_nans(
    ds: xr.Dataset,
):
    """
    Checks for NaN values in the dataset.
    Removes years containing NaNs.

    Args:
        ds (xr.Dataset): Dataset to check

    Returns:
        None
    """

    # loop over the years
    for year in ds.time.dt.year.values:
        # select the year
        ds_year = ds.sel(time=f"{year}")

        # check for NaNs
        # If there are any nans, raise an error
        if np.isnan(ds_year).any():
            print("Nans found in obs for year:", year)
            if np.isnan(ds_year).all():
                print("All values are nan")
                raise ValueError("All values are nan")

    # return non
    return None
