#!/bin/bash
#SBATCH --partition=short-serial
#SBATCH --job-name=ben-array-regrid-test
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH --time=10:00
#SBATCH --array=1901-2009

# Echo the task id
echo "My SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# Print the CLIs
echo "My CLIs: ${CLI}"
echo "The number of CLIs: ${#CLI[@]}"
echo "The desired number of arguments is: 4"

# Check if the correct number of arguments is passed
if [ $# -ne 4 ]; then
    echo "Usage: sbatch submit-regrid-file.bash <experiment> <variable> <nens> <init_month>"
    echo "Example: sbatch submit-regrid-file.bash ASF20C SLP 50 Nov"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
nens=$3
init_month=$4

# Print the arguments
echo "My experiment: ${experiment}"
echo "My variable: ${variable}"
echo "My nens: ${nens}"
echo "My init_month: ${init_month}"

# Load cdo
module load jaspy

# Set up the process script
# This script takes args:
# <experiment> <variable> <nens> <init_month> <SLURM_ARRAY_TASK_ID>
process_script="/home/users/benhutch/skill-maps/seasonal/batch-scripts/regrid-file.bash"

# Echo that we are running the process script
echo "Running the process script: ${process_script}"
echo "With args: ${experiment} ${variable} ${nens} ${init_month} ${SLURM_ARRAY_TASK_ID}"

# Run the process script
bash ${process_script} ${experiment} ${variable} ${nens} ${init_month} ${SLURM_ARRAY_TASK_ID}

