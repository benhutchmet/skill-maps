#!/bin/bash -x
# [debugging] -x option prints each command before executing it
# calculate-anomalies.bash
#
# test script for removing the model mean state from 
# each ensemble member of a given model
#
# For example: calculate-anomalies.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF
#

# we want to remove the anomalies for each ensemble member
# and each initialisation method for each ensemble member

# set the usage message
USAGE_MESSAGE="Usage: calculate-anomalies.bash <model> <variable> <region> <forecast-range> <season>"

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

# set up an output directory for the temporary time mean files
output_dir_tmp=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp
# make the output directory
mkdir -p $output_dir_tmp

# loop over the files
# and calculate the anomalies
if [ "$model" == "NorCPM1" ]; then
    
    # create a file name for the temp model mean state file
    # for init scheme i1
    temp_file_i1=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state_i1.nc

    # loop over all of the files to process the time mean
    for file in ${all_files_i1}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*i1*.nc ${temp_file_i1}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file_i1 ]; then
        echo "ERROR: model mean state file not created i1"
        exit 1
    fi

    # loop over the files
    # and calculate the anomalies
    for file in ${all_files_i1}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_i1} ${output_dir}/${filename%.nc}-anoms.nc
    done

    # create a file name for the temp model mean state file
    # for init scheme i2
    temp_file_i2=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state_i2.nc

    # calculate the model mean state
    # for init scheme i2
    # loop over all of the files to process the time mean
    for file in ${all_files_i2}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*i2*.nc ${temp_file_i2}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file_i2 ]; then
        echo "ERROR: model mean state file not created i2"
        exit 1
    fi


    # loop over the files
    # and calculate the anomalies
    for file in ${all_files_i2}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_i2} ${output_dir}/${filename%.nc}-anoms.nc
    done

    # Perform the subtraction to calculate anomalies
elif [ "$model" == "EC-Earth3" ]; then
        
    # create a file name for the temp model mean state file
    # for init scheme i1
    temp_file_i1=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state_i1.nc
    
    # calculate the model mean state
    # for init scheme i1
    # loop over all of the files to process the time mean
    for file in ${all_files_i1}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*i1*.nc ${temp_file_i1}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file_i1 ]; then
        echo "ERROR: model mean state file not created i1"
        exit 1
    fi

    # loop over the files
    # and calculate the anomalies
    for file in ${all_files_i1}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_i1} ${output_dir}/${filename%.nc}-anoms.nc
    done

    # create a file name for the temp model mean state file
    # for init scheme i2
    temp_file_i2=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state_i2.nc

    # calculate the model mean state
    # for init scheme i2
    # loop over all of the files to process the time mean
    for file in ${all_files_i2}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*i2*.nc ${temp_file_i2}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file_i2 ]; then
        echo "ERROR: model mean state file not created i2"
        exit 1
    fi

    # loop over the files
    # and calculate the anomalies
    for file in ${all_files_i2}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_i2} ${output_dir}/${filename%.nc}-anoms.nc
    done

    # create a file name for the temp model mean state file
    # for init scheme i4
    temp_file_i4=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state_i4.nc

    # calculate the model mean state
    # for init scheme i4
    # loop over all of the files to process the time mean
    for file in ${all_files_i4}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*i4*.nc ${temp_file_i4}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file_i4 ]; then
        echo "ERROR: model mean state file not created i4"
        exit 1
    fi

    # loop over the files
    # and calculate the anomalies
    for file in ${all_files_i4}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_i4} ${output_dir}/${filename%.nc}-anoms.nc
    done
else

    # create a file name for the temp model mean state file
    temp_file_mms=/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp/model_mean_state.nc

    # calculate the model mean state
    # loop over all of the files to process the time mean
    for file in ${all_files}; do
        
        # set up a temporary file name
        temp_fname="temp-$(basename ${file})"
        temp_file=${output_dir_tmp}/${temp_fname}

        # calculate the time mean for each file
        cdo timmean ${file} ${temp_file}

    done

    # now take the ensemble mean of the time mean files
    cdo ensmean ${output_dir_tmp}/temp-*.nc ${temp_file_mms}

    # ensure that the model mean state file has been created
    if [ ! -f $temp_file ]; then
        echo "ERROR: model mean state file not created"
        exit 1
    fi

    # loop over the files
    # and calculate the anomalies
    for file in ${all_files}; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_file_mms} ${output_dir}/${filename%.nc}-anoms.nc
    done

fi

# Clean up temporary files
rm -f ${output_dir_tmp}/temp-*.nc

echo "Anomalies have been calculated for $model $variable $region $forecast_range $season and saved in $output_dir"

# End of script
exit 0
