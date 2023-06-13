#!/bin/bash
#
# calculate-anomalies.bash
#
# test script for removing the model mean state from 
# each ensemble member of a given model
#
# For example: calculate-anomalies.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 1 1
#

# we want to remove the anomalies for each ensemble member
# and each initialisation method for each ensemble member

# set the usage message
USAGE_MESSAGE="Usage: calculate-anomalies.bash <model> <variable> <region> <forecast-range> <season> <run> <init_scheme>"

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

# set up all files
# from which the model mean state will be calculated
# for most models (1 init scheme) this will be //
# the mean of all ensemble members over all initialisation //
# years
# for some models (2/3 init schemes) this will be //
# the mean of all ensemble members over all initialisation //
# years for each init scheme
# NorCPM1 had 2 init schemes (i1 and i2)
# EC-Earth3 has up to 3 init schemes (i1, i2 and i4)
# if the model is any of the others:
# first specify the NorCPM1 and EC-Earth3 init schemes
# then specify the other models
if [ "$model" == "NorCPM1" ]; then
    # files which are used to calc MMS
    # for init scheme i1
    all_files_i1=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i1*.nc

    # files which are used to calc MMS
    # for init scheme i2
    all_files_i2=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i2*.nc
elif [ "$model" == "EC-Earth3" ]; then
    # files which are used to calc MMS
    # for init scheme i1
    all_files_i1=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i1*.nc

    # files which are used to calc MMS
    # for init scheme i2
    all_files_i2=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i2*.nc

    # files which are used to calc MMS
    # for init scheme i4
    # some models will generate ERROR for i4
    # as they do not have this init scheme i4
    all_files_i4=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i4*.nc
else
    # files which are used to calc MMS
    # for init scheme i*
    all_files=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_*.nc
fi

# set up the output directory
# for the anoms files
output_dir=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/anoms
# make the output directory
mkdir -p $output_dir

# loop over the files
