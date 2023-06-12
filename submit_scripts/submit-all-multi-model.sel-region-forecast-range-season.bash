#!/bin/bash
#
# submit-all-multi-model.merge-multi-file.bash
#
# For example: submit-all-multi-model.merge-multi-file.bash HadGEM3-GC31-MM 1960 1969 psl north-atlantic 2-5 DJFM
#


# import the models list
source $PWD/dictionaries.bash
# echo the multi-models list
echo "[INFO] models list: $models"

# set the usage message
USAGE_MESSAGE="Usage: submit-all-multi-model.merge-multi-file.bash <model> <initial-year> <final-year> <variable> <region> <forecast-range> <season>"

# check that the correct number of arguments have been passed
if [ $# -ne 7 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, initial year and final year
model=$1
initial_year=$2
final_year=$3
variable=$4
region=$5
forecast_range=$6
season=$7

# set the extractor script
EXTRACTOR=$PWD/process_scripts/multi-model.sel-region-forecast-range-season.bash

# if model=all, then run a for loop over all of the models
if [ "$model" == "all" ]; then

    # set up the model list
    echo "[INFO] Extracting data for all models: $models"

    for model in $models; do

    # Echo the model name
    echo "[INFO] Extracting data for model: $model"

    # Set up the number of ensemble members using a case statement
    case $model in
        BCC-CSM2-MR) run=8;;
        MPI-ESM1-2-HR) run=10;;
        CanESM5) run=20;;
        CMCC-CM2-SR5) run=10;;
        HadGEM3-GC31-MM) run=10;;
        EC-Earth3) run=10;;
        MRI-ESM2-0) run=10;;
        MPI-ESM1-2-LR) run=16;;
        FGOALS-f3-L) run=9;;
        MIROC6) run=10;;
        IPSL-CM6A-LR) run=10;;
        CESM1-1-CAM5-CMIP5) run=40;;
        NorCPM1) run=10;;
        *) echo "[ERROR] Model not recognised"; exit 1;;
    esac

    # echo the number of ensemble members
    echo "[INFO] Number of ensemble members: $run"

    # Set the output directory for the LOTUS outputs
    OUTPUTS_DIR="/work/scratch-nopw/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
    mkdir -p $OUTPUTS_DIR

    # sequence through the years specified
    for year in $(seq $initial_year $final_year); do

        # Echo the year
        echo "Current year: ${year}"

        # Loop through the ensemble members
        for run in $(seq 1 $run); do

            # set the date
            year=$(printf "%d" $year)
            run=$(printf "%d" $run)
            echo "[INFO] Submitting job for $model, s$year, r$run, for variable $variable in region $region, for forecast period year $forecast_range and season $season"

            # submit the job to LOTUS
            # this is wrong lol
            sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/$model-s$year-r$run--$variable.%j.out -e $OUTPUT_DIR/$model-s$year-r$run--$variable.%j.err $EXTRACTOR $model $year $run $init $variable $region $season $forecast_range
        done