#!/bin/bash
#
# submit-all-multi-model.merge-multi-file.bash
#
# For example: submit-all-multi-model.merge-multi-file.bash HadGEM3-GC31-MM 1960 1969 psl
#

USAGE_MESSAGE="Usage: submit-all-multi-model.merge-multi-file.bash <model> <initial-year> <final-year> <variable>"

# check that the correct number of arguments have been passed
if [ $# -ne 4 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, initial year and final year
model=$1
initial_year=$2
final_year=$3

# specify the initialization method
# 1 in this case
# ---ASK DOUG ABOUT THIS---
init=1

# set the extractor script
EXTRACTOR=$PWD/process_scripts/multi-model.merge-multi-file.bash

# set up the model list
models="EC-Earth3 HadGEM3-GC31-MM EC-Earth3-HR"

# run the extractor script for each model
if [ $model == "all" ]; then

    for model in $models; do

        echo "[INFO] Submitting jobs for $model"

        # set the number of ensemble members
        if [ $model == "HadGEM3-GC31-MM" ]; then
            run=10
        elif [ $model == "EC-Earth3" ]; then
            run=10
        else
            echo "[ERROR] No. of ensemble members not known for $model"
            exit 1
        fi

        # echo the no. of ens members
        echo "[INFO] No. of ensemble members: $run"

        # set the output directory
        OUTPUT_DIR=/work/scratch-nopw/benhutch/$variable/$model/lotus-outputs/
        # make the output directory if it doesn't exist
        mkdir -p $OUTPUT_DIR

        for year in $(seq $initial_year $final_year); do
            
        # echo the year
        echo "[INFO] Year being processed: $year"

        # if model=EC-Earth3, then we need to loop through the initializations
        # watch out for this!!
        # -------------------
        # init scheme = 2/4 - check for all var names
        if [ $model == "EC-Earth3" ]; then

            # set the number of initializations
            init=2

            # loop over the initializations
            for init in $(seq 1 $init); do

                # echo the initialization
                echo "initialization scheme: $init"

                # loop over the ensemble members
                for run in $(seq 1 $run); do
                    
                    # i2 only exists for r6-10
                    # this should spit an error for r1-5 i2
                    # but this isn't a problem

                    # set the date
                    year=$(printf "%d" $year)
                    run=$(printf "%d" $run)
                    init=$(printf "%d" $init)
                    echo "[INFO] Submitting job for $model, s$year, r$run, i$init for variable $variable"

                    # submit the job to LOTUS
                    sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.out -e $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.err $EXTRACTOR $model $year $run $init $variable

                done
            done

        # else if model is HadGEM3-GC31-MM
        else

            # set the number of initializations
            # HadGEM3-GC31-MM only has one initialization scheme
            if [ $model == "HadGEM3-GC31-MM" ]; then
                init=1
            elif [ $model == "EC-Earth3-HR" ]; then
                init=4 # covers i1, i2, and i4
                # psl and sfcWind are i1 and i2 only
                # tas and pr are i1 and i4 only
                # ignore the erros resulting from this
            else
                echo "[ERROR] No. of initializations not known for $model"
                exit 1
            fi

            # echo the number of initializations and the model
            echo "[INFO] No. of initializations: $init for model: $model"

            # loop over the ensemble members
            for run in $(seq 1 $run); do
                
                # set the date
                year=$(printf "%d" $year)
                run=$(printf "%d" $run)
                init=$(printf "%d" $init)
                echo "[INFO] Submitting job for $model, s$year, r$run, i$init for variable $variable"

                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.out -e $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.err $EXTRACTOR $model $year $run $init $variable

            done
        fi
        done
    done
else
    
    # echo the model name
    echo "[INFO] Submitting jobs for $model"

    # set up the output directory
    OUTPUT_DIR=/work/scratch-nopw/benhutch/$variable/$model/lotus-outputs/
    # make the output directory if it doesn't exist
    mkdir -p $OUTPUT_DIR

    # set the number of ensemble members
    if [ $model == "HadGEM3-GC31-MM" ]; then
        run=10
    elif [ $model == "EC-Earth3" ]; then
        run=10
    else
        echo "[ERROR] No. of ensemble members not known for $model"
        exit 1
    fi

    # echo the no. of ens members
    echo "[INFO] No. of ensemble members: $run"

    # create an outer loop to loop through the years
    for year in $(seq $initial_year $final_year); do
        
        # echo the year
        echo "[INFO] Year being processed: $year"

        # if model=EC-Earth3, then we need to loop through the initializations
        # if model=EC-Earth3, then we need to loop through the initializations
        if [ $model == "EC-Earth3" ]; then

            # set the number of initializations
            init=4 # covers all posible initializations

            # loop over the initializations
            for init in $(seq 1 $init); do

                # echo the initialization
                echo "initialization scheme: $init"

                # loop over the ensemble members
                for run in $(seq 1 $run); do
                    
                    # i2 only exists for r6-10
                    # this should spit an error for r1-5 i2
                    # but this isn't a problem

                    # set the date
                    year=$(printf "%d" $year)
                    run=$(printf "%d" $run)
                    init=$(printf "%d" $init)
                    echo "[INFO] Submitting job for $model, s$year, r$run, i$init for variable $variable"

                    # submit the job to LOTUS
                    sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.out -e $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.err $EXTRACTOR $model $year $run $init $variable

                done
            done

        else

            # set the number of initializations
            # HadGEM3-GC31-MM only has one initialization scheme
            if [ $model == "HadGEM3-GC31-MM" ]; then
                init=1
            else
                echo "[ERROR] No. of initializations not known for $model"
                exit 1
            fi

            # echo the number of initializations and the model
            echo "[INFO] No. of initializations: $init for model: $model"

            # loop over the ensemble members
            for run in $(seq 1 $run); do
                
                # set the date
                year=$(printf "%d" $year)
                run=$(printf "%d" $run)
                init=$(printf "%d" $init)
                echo "[INFO] Submitting job for $model, s$year, r$run, i$init"

                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.out -e $OUTPUT_DIR/$model-s$year-r$run-i$init-$variable.%j.err $EXTRACTOR $model $year $run $init $variable

            done
        fi
    done
fi 

