#!/bin/bash
#
# multi-file-mergetime-test.bash
#
# For example: multi-model.merge-multi-file.bash HadGEM3-GC31-MM 1960 1 1 psl
#

USAGE_MESSAGE="Usage: multi-file-mergetime-test.bash <model> <initialization-year> <run-number> <initialization-number> <variable>"

# check that the correct number of arguments have been passed
if [ $# -ne 5 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# set the model and initialization year
model=$1
init_year=$2

# set the run number and initialization number
run=$3
init=$4

# set the variable
variable=$5

# set the output directory
OUTPUT_DIR=/work/scratch-nopw/benhutch/$variable/$model/outputs/mergetime
# make the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# set up an if loop for the model name
if [ $model == "HadGEM3-GC31-MM" ]; then
    # set the files to be processed
    model_group="MOHC"
elif [ $model == "EC-Earth3" ]; then
    # set the files to be processed
    model_group="EC-Earth-Consortium"
else 
    echo "[ERROR] Model name not recognised"
    exit 1
fi

# echo the model name and group
echo "Model: $model"
echo "Model group: $model_group"


# All on JASMIN
# APART from EC-Earth3 for sfcWind
# set up the files
# if the model is HadGEM3-GC31-MM or EC-Earth3
# then the files are in the format:
# /badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${init_year}-r${run}i${init}p?f?/Amon/psl/g?/files/d????????/*.nc
if [ $model == "HadGEM3-GC31-MM" ]; then
    if [ $variable == "psl" ] || [ $variable == "sfcWind" ] || [ $variable == "tas" ] || [ $variable == "rsds" ]; then
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${init_year}-r${run}i${init}p?f?/Amon/$variable/g?/files/d????????/*.nc"
    else
        echo "[ERROR] Variable not recognised"
        exit 1
    fi
    fi
# if the model is EC-Earth3
if [ $model == "EC-Earth3" ]; then
    if [ $variable == "psl" ] || [ $variable == "tas" ] || [ $variable == "rsds" ]; then
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${init_year}-r${run}i${init}p?f?/Amon/$variable/g?/files/d????????/*.nc"
    elif [ $variable == "sfcWind" ]; then
        files="/work/xfc/vol5/user_cache/benhutch/${variable}/${model}"
    else
        echo "[ERROR] Variable not recognised"
        exit 1
    fi
fi


# test the files
# /work/xfc/vol5/user_cache/benhutch/EC-Earth-Consortium/EC-Earth3-HR/psl_Amon_EC-Earth3-HR_dcppA-hindcast_s1995-r5i2p?f?_g?_*.nc

# activate the environment containing CDO
module load jaspy

# set up the final year for the filename
if [ $model == "HadGEM3-GC31-MM" ] || [ $model == "EC-Earth3" ]; then
    final_year=$((init_year+11))
else
    echo "[ERROR] Model name not recognised"
    exit 1
fi

# set up the filename for the merged file
merged_fname="${variable}_Amon_${model}_dcppA-hindcast_s${init_year}-r${run}i${init}_gn_${init_year}11-${final_year}03.nc"
# set up the path for the merged file
merged_file="$OUTPUT_DIR/$merged_fname"

# merge the files into a single file
# by the time axis
echo "[INFO] Merging files by time axis for $model, s$init_year, r$run, i$init, $variable"
echo "[INFO] Files to be merged: $files"

# merge the files
cdo mergetime $files $merged_file

