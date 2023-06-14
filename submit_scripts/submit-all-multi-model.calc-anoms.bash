#!/bin/bash
#
# submit-all-multi-model.calc-anoms.bash
#
# For example: submit-all-multi-model.calc-anoms.bash CMCC-CM2-SR5 psl north-atlantic 2-5 DJFM
#
# Removes the model mean state from the forecast data
# and calculates the anomalies from the model mean state

# import the models list
source $PWD/dictionaries.bash
# echo the multi-models list
echo "[INFO] models list: $models"

# set the usage message
USAGE_MESSAGE="Usage: submit-all-multi-model.calc-anoms.bash <model> <variable> <region> <forecast-range> <season>"

# check that the correct number of arguments have been passed
if [ $# -ne 5 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5

# set the extractor script
EXTRACTOR=$PWD/process_scripts/multi-model.calc-anoms.bash

# make sure that cdo is loaded
module load jaspy

# if model=all, then run a for loop over all of the models
if [ "$model" == "all" ]; then

# set up the model list
echo "[INFO] Extracting data for all models: $models"

    # loop over the models
    for model in $models; do

    # Echo the model name
    echo "[INFO] Calculating anomalies for model: $model"

    # Set up the output directory
    # For the LOTUS outputs
    OUTPUT_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/lotus-outputs"
    mkdir -p $OUTPUT_DIR

    # Echo the output directory
    echo "[INFO] Output directory: $OUTPUT_DIR"

    # Echo info for job submission
    echo "[INFO] Submitting job for model: $model, variable: $variable, region: $region, forecast range: $forecast_range, season: $season"

    # Submit the job
    sbatch --partition=short-serial -t 10 -o $OUTPUT_DIR/$model.$variable.$region.$forecast_range.$season.out -e $OUTPUT_DIR/$model.$variable.$region.$forecast_range.$season.err $EXTRACTOR $model $variable $region $forecast_range $season

    done
    
else

# Individual model case
echo "[INFO] Calculating anomalies for model: $model"

# Set up the output directory
# For the LOTUS outputs
OUTPUT_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/lotus-outputs"
mkdir -p $OUTPUT_DIR

# Echo the output directory
echo "[INFO] Output directory: $OUTPUT_DIR"

# Echo info for job submission
echo "[INFO] Submitting job for model: $model, variable: $variable, region: $region, forecast range: $forecast_range, season: $season"

# Submit the job
sbatch --partition=short-serial -t 10 -o $OUTPUT_DIR/$model.$variable.$region.$forecast_range.$season.out -e $OUTPUT_DIR/$model.$variable.$region.$forecast_range.$season.err $EXTRACTOR $model $variable $region $forecast_range $season

fi