#!/bin/bash
#
# calculate-anomalies.bash
#
# Script for removing the model mean state //
# from each ensemble member of a given model.
#
# For example: calculate-anomalies.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF

# Set the usage message
USAGE_MESSAGE="Usage: calculate-anomalies.bash <model> <variable> <region> <forecast-range> <season>"

# Check that the correct number of arguments have been passed
if [ $# -ne 5 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5

# Load cdo
module load jaspy

# Base directory
base_dir="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs"

# Function for processing files
process_files() {
    init_scheme=$1
    files_path="$base_dir/mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r*${init_scheme}*.nc"

    # Echo the files to be processed
    echo "Model mean state calculated and anomalies calculated for: $files_path"

    temp_model_mean_state="$base_dir/tmp/model_mean_state_${init_scheme}.nc"

    # Calculate the model mean state
    for file in $files_path; do
        temp_fname="temp-$(basename ${file})"
        temp_file="$base_dir/tmp/${temp_fname}"
        cdo timmean ${file} ${temp_file}
    done

    # Take the ensemble mean of the time mean files
    cdo ensmean ${base_dir}/tmp/temp-*${init_scheme}*.nc ${temp_model_mean_state}

    # Ensure that the model mean state file has been created
    if [ ! -f $temp_model_mean_state ]; then
        echo "ERROR: model mean state file not created ${init_scheme}"
        exit 1
    fi

    # Calculate the anomalies
    for file in $files_path; do
        filename=$(basename ${file})
        cdo sub ${file} ${temp_model_mean_state} "$base_dir/anoms/${filename%.nc}-anoms.nc"
    done
}

# Create output directories
mkdir -p $base_dir/anoms
mkdir -p $base_dir/tmp

# Processing
case $model in
    "NorCPM1")
        process_files "i1"
        process_files "i2"
        ;;
    "EC-Earth3")
        process_files "i1"
        process_files "i2"
        process_files "i4"
        ;;
    *)
        # For all other models, use a wildcard for init_scheme
        process_files "i*"
        ;;
esac

# Clean up temporary files
rm -f ${base_dir}/tmp/temp-*.nc

echo "Anomalies have been calculated for $model $variable $region $forecast_range $season and saved in $base_dir/anoms"

# End of script
exit 0
