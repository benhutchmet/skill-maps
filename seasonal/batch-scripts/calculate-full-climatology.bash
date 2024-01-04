#!/bin/bash
#SBATCH --partition=test
#SBATCH --job-name=ben-calculate-full-climatology
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH --time=15:00

# Print the CLIs
echo "My CLIs: ${CLI}"
echo "The number of CLIs: ${#CLI[@]}"
echo "The desired number of arguments is: 3"

# Check if the correct number of arguments is passed
if [ $# -ne 3 ]; then
    echo "Usage: sbatch calculate-full-climatology.bash <experiment> <variable> <init_month>"
    echo "Example: sbatch calculate-full-climatology.bash ASF20C SLP Nov"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
init_month=$3

# Print the arguments
echo "My experiment: ${experiment}"
echo "My variable: ${variable}"
echo "My init_month: ${init_month}"

# Load cdo
module load jaspy

# Find the folder over which to take the ensemble mean
base_path_canari="/gws/nopw/j04/canari/users/benhutch/seasonal"

# Set up the folder in which the year climatologies are stored
clim_folder="${base_path_canari}/${experiment}/${variable}/${init_month}_START/climatology"

# If this folder does not exist, raise an error
if [ ! -d ${clim_folder} ]; then
    echo "The folder ${clim_folder} does not exist"
    exit 1
fi

# Take the ensemble mean over the climatology folder
echo "Taking the ensemble mean over the climatology folder"

# Set up the output file name
output_file="${clim_folder}/full-climatology.nc"

# If this file already exists, raise an error
if [ -f ${output_file} ]; then
    echo "The file ${output_file} already exists"
    exit 1
fi

# Take the ensemble mean
cdo ensmean ${clim_folder}/*_clim.nc ${output_file}