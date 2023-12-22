#!/bin/bash
#
# regid-file.bash
# 
# Regrids all ensemble members of a variable for a given experiment and year
#
# Usage: bash regrid-file.bash <experiment> <variable> <nens> <init_month> <year>
#
# Example: bash regrid-file.bash ASF20C SLP 50 Nov 1901
#

# Set up the usage message
usage="Usage: bash regrid-file.bash <experiment> <variable> <nens> <init_month> <year>"

# Check if the correct number of arguments is passed
if [ $# -ne 4 ]; then
    echo ${usage}
    echo "Example: bash regrid-file.bash ASF20C SLP 50 Nov 1901"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
nens=$3
init_month=$4
year=$4

# Set up the base path for the data to be processed
base_path_20c="/badc/deposited2020/seasonal-forecasts-20thc/data"

# Set up the folder name
folder_name="${variable}monthly_${experiment}_${init_month}START_ENSmems"

# Set up the path to the folder
path_to_folder="${base_path_20c}/${folder_name}"

# Check if the folder exists
if [ ! -d ${path_to_folder} ]; then
    echo "The folder ${path_to_folder} does not exist"
    exit 1
fi

# Set up the path to the output folder
base_path_canari="/gws/nopw/j04/canari/users/benhutch/seasonal"

# Set up the canari folder for saving
canari_folder="${base_path_canari}/${experiment}/${variable}/${init_month}_START"

# Check if the folder exists
if [ ! -d ${canari_folder} ]; then
    echo "The folder ${canari_folder} does not exist"
    echo "Making the folder"
    # Make the folder
    mkdir -p ${canari_folder}
fi

# Set up the gridspec file
gridspec_file="/home/users/benhutch/gridspec/gridspec-global.txt"

# If this file does not exist, then exit
if [ ! -f ${gridspec_file} ]; then
    echo "The gridspec file ${gridspec_file} does not exist"
    exit 1
fi

# Set up the files to be processed
files=(${path_to_folder}/${variable}monthly_${year}*.nc)

# Check if the files exist
if [ ${#files[@]} -eq 0 ]; then
    echo "The files ${path_to_folder}/${variable}monthly_${year}*.nc do not exist"
    exit 1
fi

# Loop over the files
for file in ${files[@]}; do
    # Extract the filename
    filename=$(basename ${file})
    # Extract the ensemble member
    ens_member=$(echo ${filename} | cut -d'_' -f4 | cut -d'.' -f1)
    # Set up the output file
    output_file="${canari_folder}/${variable}monthly_${year}_${ens_member}.nc"
    # Check if the output file exists
    if [ -f ${output_file} ]; then
        echo "The output file ${output_file} already exists"
        echo "Skipping"
        continue
    fi
    # Echo the command
    echo "cdo remapbil,${gridspec_file} ${file} ${output_file}"
    # Run the command
    cdo remapbil,${gridspec_file} ${file} ${output_file}
done