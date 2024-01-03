#!/bin/bash
#
# calculate-climatology.bash
#
# Removes the first time step (november) from the seasonal forecast data
# and calculates the climatology over all of the ensemble members and years
#
# Usage: bash calculate-climatology.bash <experiment> <variable> <init_month> <year>
# 
# Example: bash calculate-climatology.bash ASF20C SLP Nov 1901
#

# Set up the usage message
usage="Usage: bash calculate-climatology.bash <experiment> <variable> <init_month> <year>"

# Check if the correct number of arguments is passed
if [ $# -ne 4 ]; then
    echo ${usage}
    echo "Example: bash calculate-climatology.bash ASF20C SLP Nov 1901"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
init_month=$3
year=$4

# Set up the base path for the data to be processed
base_path_canari="/gws/nopw/j04/canari/users/benhutch/seasonal"

# Set up the canari folder for saving
canari_folder="${base_path_canari}/${experiment}/${variable}/${init_month}_START"

# If this folder does not exist, exit
if [ ! -d ${canari_folder} ]; then
    echo "The folder ${canari_folder} does not exist"
    exit 1
fi

# Set up the DJF folder
djf_folder="${canari_folder}/DJF"

# If this folder does not exist, create it
if [ ! -d ${djf_folder} ]; then
    echo "The folder ${djf_folder} does not exist"
    echo "Creating the folder"
    mkdir -p ${djf_folder}
fi

# Loop over all the files in the canari folder
for file in ${canari_folder}/*${year}*.nc; do

    # Extract the file name
    file_name=$(basename ${file})

    # Cut the .nc extension
    file_name_nonc=$(echo ${file_name} | cut -d'.' -f1)

    # Echo the file name
    echo "Processing ${file_name}"

    # Set up the output file name
    output_file="${djf_folder}/${file_name_nonc}_DJF.nc"

    # If the output file already exists, skip this file
    if [ -f ${output_file} ]; then
        echo "The file ${output_file} already exists"
        echo "Skipping this file"
        continue
    fi

    # Set up the cdo command
    cdo delete,timestep=1 ${file} ${output_file}

done

# # Echo that we are now calculating the climatology
# echo "Calculating the climatology"

# # Set up the folder for the climatology
# clim_folder="${canari_folder}/climatology"

# # If this folder does not exist, create it
# if [ ! -d ${clim_folder} ]; then
#     echo "The folder ${clim_folder} does not exist"
#     echo "Creating the folder"
#     mkdir -p ${clim_folder}
# fi

# # Set up the output file
# output_file="${clim_folder}/${variable}_climatology.nc"

# # If the output file already exists, skip this file
# if [ -f ${output_file} ]; then
#     echo "The file ${output_file} already exists"
#     echo "Skipping this file"
#     continue
# fi

# # Set up the cdo command
# cdo ensmean ${djf_folder}/*.nc ${output_file}

# Echo that the script is finished
echo "Finished"