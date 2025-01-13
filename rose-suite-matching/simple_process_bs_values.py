"""

simple_process_bs_values.py

Author: Ben Hutchins
January 2025

Simple bootstrapping for ua ONDJFM and AYULGS (once processed).

Usage
-----

    python simple_process_bs_values.py <nboot> <variable> <region>
                                        <season> <forecast_range>

Example
-------

    python simple_process_bs_values.py 1000 ua global ONDJFM 2-9
                                        

Args
----

    nboot: int
        Number of bootstraps to perform.

    variable: str
        Variable to process.

    region: str
        Region to process.

    season: str
        Season to process.

    forecast_range: str
        Forecast range to process.

Outputs
-------

    Saves the bootstrapped values to a file.

"""

# local imports
import os
import sys
import argparse

# third-party imports
import numpy as np


# Import the functions
sys.path.append("/home/users/benhutch/skill-maps/python")
from functions import forecast_stats, process_observations

def main():

    save_dir = "/gws/nopw/j04/canari/users/benhutch/bootstrapping"

    # set up the args
    parser = argparse.ArgumentParser()

    # add the arguments
    parser.add_argument("nboot", type=int, help="Number of bootstraps to perform.")
    parser.add_argument("variable", type=str, help="Variable to process.")
    parser.add_argument("region", type=str, help="Region to process.")
    parser.add_argument("season", type=str, help="Season to process.")
    parser.add_argument("forecast_range", type=str, help="Forecast range to process.")

    # parse the arguments
    args = parser.parse_args()

    # set up the number of bootstraps
    print(f"Number of bootstraps: {args.nboot}")
    print(f"Variable: {args.variable}")
    print(f"Region: {args.region}")
    print(f"Season: {args.season}")
    print(f"Forecast range: {args.forecast_range}")

    # Set up the save path
    save_path_nao_matched = (
        save_dir
        + "/"
        + args.variable
        + "/"
        + args.region
        + "/"
        + args.season
        + "/"
        + args.forecast_range
        + "/"
        + "nao_matched"
        + "/"
        + "no_bootstraps_"
        + str(args.nboot)
        + "/"
    )

    # If the save path doesn't exist, create it
    if not os.path.exists(save_path_nao_matched):
        os.makedirs(save_path_nao_matched)

    # Set up the file names for the arrays
    corr1_name = f"corr1_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    corr1_p_name = f"corr1_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    corr2_name = f"corr2_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    corr2_p_name = f"corr2_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    corr10_name = f"corr10_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    corr10_p_name = f"corr10_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    partial_r_name = f"partial_r_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Partial r min and max values
    partial_r_min_name = (
        f"partial_r_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    partial_r_max_name = (
        f"partial_r_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    # Also the partial r bias
    partial_r_bias_name = (
        f"partial_r_bias_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    obs_resid_name = f"obs_resid_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Save the forecaast 1 residual array
    fcst1_em_resid_name = (
        f"fcst1_em_resid_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    partial_r_p_name = (
        f"partial_r_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    sigo = f"sigo_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    sigo_resid = f"sigo_resid_{args.variable}_{args.region}_{args.season}_" + f"{args.forecast_range}.npy"

    # Also save arrays for the correlation differences
    corr_diff_name = f"corr_diff_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Min and max values
    corr_diff_min_name = (
        f"corr_diff_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    corr_diff_max_name = (
        f"corr_diff_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    corr_diff_p_name = (
        f"corr_diff_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    # Also save arrays for the RPC and RPC_p
    rpc1_name = f"rpc1_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Min and max arrays
    rpc1_min_name = f"rpc1_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    rpc1_max_name = f"rpc1_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    rpc1_p_name = f"rpc1_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    rpc2_name = f"rpc2_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Min and max arrays
    rpc2_min_name = f"rpc2_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    rpc2_max_name = f"rpc2_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    rpc2_p_name = f"rpc2_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the arrays for MSSS1 and MSSS2

    msss1_name = f"msss1_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the arrays for the min and max values of MSS1 and MSS2
    msss1_min_name = f"msss1_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    msss1_max_name = f"msss1_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the arrays for the MSSS1 and MSSS2 p values
    msss1_p_name = f"msss1_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the corr1 and corr2 min and max values
    corr1_min_name = f"corr1_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    corr1_max_name = f"corr1_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    corr2_min_name = f"corr2_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    corr2_max_name = f"corr2_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the corr10 min and max values
    corr10_min_name = (
        f"corr10_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    corr10_max_name = (
        f"corr10_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    corr12_name = f"corr12_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Also save the corr12 min and max values
    corr12_min_name = (
        f"corr12_min_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    corr12_max_name = (
        f"corr12_max_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"
    )

    # Also save the corr12 p values
    corr12_p_name = f"corr12_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}" + ".npy"

    # Set up the names for the forecast time series
    fcst1_ts_name = f"fcst1_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    fcst2_ts_name = f"fcst2_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    fcst10_ts_name = f"fcst10_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    obs_ts_name = f"obs_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.npy"

    # Set up the names for the values of the forecast stats
    nens1_name = f"nens1_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.txt"

    nens2_name = f"nens2_{args.variable}_{args.region}_{args.season}_{args.forecast_range}.txt"

    start_end_years = (
        f"start_end_years_{args.variable}_{args.region}_{args.season}_" + f"{args.forecast_range}.txt"
    )

    # Set the names for the new variables
    start_end_years_short = (
        f"start_end_years_{args.variable}_{args.region}_{args.season}_" + f"{args.forecast_range}.txt"
    )

    # Set up the names  for the short time series
    fcst1_ts_short_name = (
        f"fcst1_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}_short.npy"
    )

    obs_ts_short_name = (
        f"obs_ts_{args.variable}_{args.region}_{args.season}_{args.forecast_range}_short.npy"
    )

    corr1_short_name = f"corr1_{args.variable}_{args.region}_{args.season}_{args.forecast_range}_short.npy"

    corr1_p_short_name = (
        f"corr1_p_{args.variable}_{args.region}_{args.season}_{args.forecast_range}_short.npy"
    )

    # set up the base directory
    base_dir = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data/test-sfcWind/"

    # set up the fname
    fname_ua_ondjfm = "ua_ONDJFM_global_1961_2014_2-9_4_20_1736483972.787185_nao_matched_members.npy"

    # load in the data
    fcst1_nm = np.load(os.path.join(base_dir, fname_ua_ondjfm))

    # Process the observations
    obs = process_observations(
        variable="ua",
        region="global",
        region_grid=None,
        forecast_range="2-9",
        season="ONDJFM",
        observations_path="/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_var131_85000.nc",
        obs_var_name="var131",
        plev=85000,
    )

    # constrain obs between 1969-01-01 and 2020-01-01
    obs_constr = obs.loc["1969-01-01":"2019-12-31"]

    # extract the value of the observations
    obs_values = obs_constr.values

    # create a dummy array of the same shape as fcst1_nm
    fcst2_ph = np.zeros_like(fcst1_nm)

    fcst_stats = forecast_stats(
    obs=obs_values,
    forecast1=fcst1_nm,
    forecast2=fcst2_ph,
    no_boot=args.nboot,
    )

    # set the save path
    save_path = save_path_nao_matched

        # If the save path doesn't exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the arrays
    # if the file already exists, don't overwrite it
    np.save(save_path + corr1_name, fcst_stats["corr1"])

    # Save the min and max values
    np.save(save_path + corr1_min_name, fcst_stats["corr1_min"])

    np.save(save_path + corr1_max_name, fcst_stats["corr1_max"])

    np.save(save_path + corr1_p_name, fcst_stats["corr1_p"])

    np.save(save_path + corr2_name, fcst_stats["corr2"])

    # Save the min and max values
    np.save(save_path + corr2_min_name, fcst_stats["corr2_min"])

    np.save(save_path + corr2_max_name, fcst_stats["corr2_max"])

    np.save(save_path + corr2_p_name, fcst_stats["corr2_p"])

    np.save(save_path + corr10_name, fcst_stats["corr10"])

    # Save the min and max values
    np.save(save_path + corr10_min_name, fcst_stats["corr10_min"])

    np.save(save_path + corr10_max_name, fcst_stats["corr10_max"])

    np.save(save_path + corr10_p_name, fcst_stats["corr10_p"])

    # Save the MSSS1 and MSSS2 arrays
    np.save(save_path + msss1_name, fcst_stats["msss1"])

    # Save the min and max values
    np.save(save_path + msss1_min_name, fcst_stats["msss1_min"])

    np.save(save_path + msss1_max_name, fcst_stats["msss1_max"])

    # Save the MSSS1 and MSSS2 p values

    np.save(save_path + msss1_p_name, fcst_stats["msss1_p"])

    # Save the RPC1 and RPC2 arrays
    np.save(save_path + rpc1_name, fcst_stats["rpc1"])

    np.save(save_path + rpc2_name, fcst_stats["rpc2"])

    # Save the min and max values
    np.save(save_path + rpc1_min_name, fcst_stats["rpc1_min"])

    np.save(save_path + rpc1_max_name, fcst_stats["rpc1_max"])

    np.save(save_path + rpc2_min_name, fcst_stats["rpc2_min"])

    np.save(save_path + rpc2_max_name, fcst_stats["rpc2_max"])

    # Save the RPC1 and RPC2 p values
    np.save(save_path + rpc1_p_name, fcst_stats["rpc1_p"])

    np.save(save_path + rpc2_p_name, fcst_stats["rpc2_p"])

    # Save the corr_diff arrays
    np.save(save_path + corr_diff_name, fcst_stats["corr_diff"])

    # Save the min and max values
    np.save(save_path + corr_diff_min_name, fcst_stats["corr_diff_min"])

    np.save(save_path + corr_diff_max_name, fcst_stats["corr_diff_max"])

    np.save(save_path + corr_diff_p_name, fcst_stats["corr_diff_p"])

    # Save the partial r min and max values
    np.save(save_path + partial_r_min_name, fcst_stats["partialr_min"])

    np.save(save_path + partial_r_max_name, fcst_stats["partialr_max"])

    # Save the partial r bias
    np.save(save_path + partial_r_bias_name, fcst_stats["partialr_bias"])

    # Save the fcst1_em_resid array
    np.save(save_path + fcst1_em_resid_name, fcst_stats["fcst1_em_resid"])

    np.save(save_path + partial_r_name, fcst_stats["partialr"])

    np.save(save_path + obs_resid_name, fcst_stats["obs_resid"])

    np.save(save_path + partial_r_p_name, fcst_stats["partialr_p"])

    np.save(save_path + sigo, fcst_stats["sigo"])

    np.save(save_path + sigo_resid, fcst_stats["sigo_resid"])

    # Save the values of the forecast stats
    np.savetxt(save_path + nens1_name, np.array([fcst_stats["nens1"]]))

    np.savetxt(save_path + nens2_name, np.array([fcst_stats["nens2"]]))

    # Save the forecast time series
    np.save(save_path + fcst1_ts_name, fcst_stats["f1_ts"])

    np.save(save_path + fcst2_ts_name, fcst_stats["f2_ts"])

    np.save(save_path + fcst10_ts_name, fcst_stats["f10_ts"])

    np.save(save_path + obs_ts_name, fcst_stats["o_ts"])




if __name__ == "__main__":
    main()