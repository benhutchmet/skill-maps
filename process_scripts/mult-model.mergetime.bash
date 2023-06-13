#!/bin/bash
#
# multi-model.mergetime.bash
#
# Script to merge the time dimension of multiple files
# For each individual ensemble member (~150 ish)
#
# For example: multi-model.mergetime.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 1 1
#

USAGE_MESSAGE="Usage: multi-model.mergetime.bash <model> <variable> <region> <forecast-range> <season> <run> <init_scheme>"

# check that the correct number of arguments have been passed
if [ $# -ne 7 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5

# extract the run and init_scheme
run=$6
init_scheme=$7

# make sure that cdo is loaded
module load jaspy

# set up the files
files=/work/scratch-nopw/benhutch/$variable/$model/$region/year_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r${run}i{init_scheme}*.nc


# echo the files to be merged
echo "Files to be merged: $files"

# set the output directory
OUTPUT_DIR=/work/scratch-nopw/benhutch/$variable/$model/$region/year_${forecast_range}/$season/outputs/mergetime
mkdir -p $OUTPUT_DIR

# set the output file
mergetime_fname="mergetime_${model}_${variable}_${region}_${forecast_range}_${season}-r${run}i${init_scheme}.nc"
OUTPUT_FILE=$OUTPUT_DIR/$mergetime_fname

# echo the output file
echo "Output file: $OUTPUT_FILE"

# merge the files
cdo mergetime $files $OUTPUT_FILE