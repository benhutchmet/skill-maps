#!/bin/bash

# Load the necessary module
module load jaspy

# Function to submit a job
submit_job() {
    local model=$1
    local variable=$2
    local region=$3
    local forecast_range=$4
    local season=$5
    local output_dir="/work/scratch-nopw/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"

    mkdir -p "$output_dir"
    echo "[INFO] Output directory: $output_dir"
    echo "[INFO] Submitting job for model: $model, variable: $variable, region: $region, forecast range: $forecast_range, season: $season"

    sbatch --partition=short-serial -t 10 -o "${output_dir}/${model}.${variable}.${region}.${forecast_range}.${season}.out" -e "${output_dir}/${model}.${variable}.${region}.${forecast_range}.${season}.err" "$EXTRACTOR" "$model" "$variable" "$region" "$forecast_range" "$season"
}

# Main script
main() {
    # Source the models list
    source "$PWD/dictionaries.bash"
    echo "[INFO] Models list: $models"

    # Usage message
    local usage_message="Usage: $(basename "$0") <model> <variable> <region> <forecast-range> <season>"

    # Check the number of arguments
    if [ $# -ne 5 ]; then
        echo "$usage_message"
        exit 1
    fi

    # Extract arguments
    local model=$1
    local variable=$2
    local region=$3
    local forecast_range=$4
    local season=$5

    # Set the extractor script
    EXTRACTOR="$PWD/process_scripts/multi-model.calc-anoms.bash"

    # If model is 'all', loop through all models, else process the single model
    if [ "$model" == "all" ]; then
        echo "[INFO] Calculating anomalies for all models: $models"
        for model in $models; do
            echo "[INFO] Calculating anomalies for model: $model"
            submit_job "$model" "$variable" "$region" "$forecast_range" "$season"
        done
    else
        echo "[INFO] Calculating anomalies for model: $model"
        submit_job "$model" "$variable" "$region" "$forecast_range" "$season"
    fi
}

# Execute main with all arguments passed to the script
main "$@"
