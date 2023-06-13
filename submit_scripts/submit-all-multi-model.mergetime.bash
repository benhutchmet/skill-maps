#!/bin/bash
#
# submit-all-multi-model.mergetime.bash
#
# Submit script for merging all the year files
# into a single file for each model, run and init method
#
# For example: submit-all-multi-model.mergetime.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF
#

# import the models list
source $PWD/dictionaries.bash
# echo the multi-models list
echo "[INFO] models list: $models"

# set the usage message
USAGE_MESSAGE="Usage: submit-all-multi-model.mergetime.bash <model> <variable> <region> <forecast-range> <season>"

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
EXTRACTOR=$PWD/process_scripts/multi-model.mergetime.bash

# make sure that cdo is loaded
module load jaspy

# if model=all, then run a for loop over all of the models
if [ "$model" == "all" ]; then

# set up the model list
echo "[INFO] Extracting data for all models: $models"

    for model in $models; do

    # Echo the model name
    echo "[INFO] Extracting data for model: $model"

    # Set up the number of ensemble members and initialisation methods using a case statement
    case $model in
        BCC-CSM2-MR)
            run=8
            init_methods=1
            ;;
        MPI-ESM1-2-HR)
            run=10
            init_methods=1
            ;;
        CanESM5)
            run=20
            init_methods=1
            ;;
        CMCC-CM2-SR5)
            run=10
            init_methods=1
            ;;
        HadGEM3-GC31-MM)
            run=10
            init_methods=1
            ;;
        EC-Earth3)
            run=10
            init_methods=4
            ;;
        MRI-ESM2-0)
            run=10
            init_methods=1
            ;;
        MPI-ESM1-2-LR)
            run=16
            init_methods=1
            ;;
        FGOALS-f3-L)
            run=9
            init_methods=1
            ;;
        CNRM-ESM2-1)
            run=10
            init_methods=1
            ;;
        MIROC6)
            run=10
            init_methods=1
            ;;
        IPSL-CM6A-LR)
            run=10
            init_methods=1
            ;;
        CESM1-1-CAM5-CMIP5)
            run=10
            init_methods=1
            ;;
        NorCPM1)
            run=10
            init_methods=2
            ;;
        *)
            echo "[ERROR] Model $model not found in dictionary"
            exit 1
            ;;
    esac

    # set the output directory
    OUTPUTS_DIR="${scratch_path}/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
    mkdir -p $OUTPUTS_DIR

        # loop over the ensemble members
        for run in $(seq 1 $run); do

            # if init_methods is greater than 1, then loop over the init methods
            if [ $init_methods -gt 1 ]; then

                # loop over the init methods
                for init in $(seq 1 $init_methods); do

                    # echo the init method
                    echo "[INFO] Initialisation method: $init"

                    # set the output file name
                    OUTPUT_FILE="${OUTPUTS_DIR}/${model}_r${run}_i${init}_year_${forecast_range}_${season}.nc"

                    # echo the output file name
                    echo "[INFO] Output file name: $OUTPUT_FILE"

                    # submit the job to LOTUS
                    sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.out -e $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.err $EXTRACTOR $model $variable $region $forecast_range $season $run $init
                    

                done

            # otherwise, just run the single init method
            else

                # set the output file name
                OUTPUT_FILE="${OUTPUTS_DIR}/${model}_${run}_${init_method}_year_${forecast_range}_${season}.nc"

                # echo the output file name
                echo "[INFO] Output file name: $OUTPUT_FILE"

                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.out -e $OUTPUT_DIR/merge.${model}.${run}.${init_methods}.${forecast_range}.${season}.err $EXTRACTOR $model $variable $region $forecast_range $season $run $init_methods

            fi
        done
    done
else

# Echo the model name
# In the case of an individual model
echo "[INFO] Extracting data for model: $model"

# Set up the output directory
OUTPUTS_DIR="${scratch_path}/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
# Make the output directory
mkdir -p $OUTPUTS_DIR

# Statement for merging the data
echo "[INFO] Merging data for model: $model"

# Set up the case statement for the number of ensemble members and initialisation methods
# Set up the number of ensemble members and initialisation methods using a case statement
case $model in
    BCC-CSM2-MR)
        run=8
        init_methods=1
        ;;
    MPI-ESM1-2-HR)
        run=10
        init_methods=1
        ;;
    CanESM5)
        run=20
        init_methods=1
        ;;
    CMCC-CM2-SR5)
        run=10
        init_methods=1
        ;;
    HadGEM3-GC31-MM)
        run=10
        init_methods=1
        ;;
    EC-Earth3)
        run=10
        init_methods=4
        ;;
    MRI-ESM2-0)
        run=10
        init_methods=1
        ;;
    MPI-ESM1-2-LR)
        run=16
        init_methods=1
        ;;
    FGOALS-f3-L)
        run=9
        init_methods=1
        ;;
    CNRM-ESM2-1)
        run=10
        init_methods=1
        ;;
    MIROC6)
        run=10
        init_methods=1
        ;;
    IPSL-CM6A-LR)
        run=10
        init_methods=1
        ;;
    CESM1-1-CAM5-CMIP5)
        run=10
        init_methods=1
        ;;
    NorCPM1)
        run=10
        init_methods=2
        ;;
    *)
        echo "[ERROR] Model $model not found in dictionary"
        exit 1
        ;;
esac

# loop over the ensemble members
for run in $(seq 1 $run); do

    # if init_methods is greater than 1, then loop over the init methods
    if [ $init_methods -gt 1 ]; then

        # loop over the init methods
        for init in $(seq 1 $init_methods); do

            # echo the init method
            echo "[INFO] Initialisation method: $init"

            # set the output file name
            OUTPUT_FILE="${OUTPUTS_DIR}/${model}_r${run}_i${init}_year_${forecast_range}_${season}.nc"

            # echo the output file name
            echo "[INFO] Output file name: $OUTPUT_FILE"

            # submit the job to LOTUS
            sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.out -e $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.err $EXTRACTOR $model $variable $region $forecast_range $season $run $init
            

        done

    # otherwise, just run the single init method
    else

        # set the output file name
        OUTPUT_FILE="${OUTPUTS_DIR}/${model}_${run}_${init_method}_year_${forecast_range}_${season}.nc"

        # echo the output file name
        echo "[INFO] Output file name: $OUTPUT_FILE"

        # submit the job to LOTUS
        sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/merge.${model}.${run}.${init}.${forecast_range}.${season}.out -e $OUTPUT_DIR/merge.${model}.${run}.${init_methods}.${forecast_range}.${season}.err $EXTRACTOR $model $variable $region $forecast_range $season $run $init_methods

    fi
done
fi