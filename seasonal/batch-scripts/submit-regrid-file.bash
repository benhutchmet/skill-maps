#!/bin/bash
#SBATCH --partition=test
#SBATCH --job-name=ben-array-regrid-test
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH --time=10:00
#SBATCH --array=1901-1910

# Echo the task id
echo "My SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# Print the CLIs
echo "My CLIs: ${CLI}"
echo "The number of CLIs: ${#CLI[@]}"
echo "The desired number of arguments is: 3"

# Check if the correct number of arguments is passed
if [ $# -ne 3 ]; then
    echo "Usage: sbatch submit-regrid-file.bash <experiment> <variable> <nens>"
    echo "Example: sbatch submit-regrid-file.bash ASF20C SLP 50"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
nens=$3

# Print the arguments
echo "My experiment: ${experiment}"
echo "My variable: ${variable}"
echo "My nens: ${nens}"

# Load cdo
module load jaspy

# Set up the process script
# This script takes args:
# <experiment> <variable> <nens> <year>
process_script=$PWD/regrid-file.bash

# Echo that we are running the process script
echo "Running the process script: ${process_script}"
echo "With args: ${experiment} ${variable} ${nens} ${SLURM_ARRAY_TASK_ID}"

# Run the process script
bash ${process_script} ${experiment} ${variable} ${nens} ${SLURM_ARRAY_TASK_ID}

