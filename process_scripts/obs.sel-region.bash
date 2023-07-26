#!/bin/bash
#
# obs.sel-region.bash
#
# Selects a region for the observations.
#
# Usage: obs.sel-region.bash <region> <input-file>
#
# For example: obs.sel-region.bash north-atlantic ~/ERA5_psl/long-ERA5-full.nc
#

# Check the number of arguments
if [ $# -ne 1 ]; then
    echo "Usage: obs.sel-region.bash <region>"
    exit 1
fi

# Set the region
region=$1

# Extract the input file path
input_file=$2

# If the input file does not exist, then exit
if [ ! -f $input_file ]; then
    echo "[ERROR] The input file does not exist: $input_file"
    exit 1
fi

# Set the region file
grid="~/gridspec/gridspec-${region}.txt"

# If the gridspec file does not exist, then exit
if [ ! -f $grid ]; then
    echo "[ERROR] The gridspec file does not exist: $grid"
    exit 1
fi

# Echo the gridspec file path
echo "The gridspec file is: $grid"

# Set up the output file
output_file="${input_file%.*}-${region}.nc"

# Echo the output file path
echo "The output file is: $output_file"

# Activate the environment containing cdo
module load jaspy

# Select the region using remapbil
cdo remapbil,$grid $input_file $output_file