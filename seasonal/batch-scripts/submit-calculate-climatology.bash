#!/bin/bash
#SBATCH --partition=short-serial
#SBATCH --job-name=ben-calculate-climatology
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH --time=10:00
#SBATCH --array=1901-2009

# Echo the task id
echo "My SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# Print the CLIs
echo "My CLIs: ${CLI}"
echo "The number of CLIs: ${#CLI[@]}"
echo "The desired number of arguments is: 3"

# Check if the correct number of arguments is passed
if [ $# -ne 3 ]; then
    echo "Usage: sbatch submit-calculate-climatology.bash <experiment> <variable> <init_month>"
    echo "Example: sbatch submit-calculate-climatology.bash ASF20C SLP Nov"
    exit 1
fi

# Load cdo

# Set up the process script
process_script="/home/users/benhutch/skill-maps/seasonal/batch-scripts/calculate-climatology.bash"

# Echo that we are running the process script
echo "Running the process script: ${process_script}"
echo "With args: ${experiment} ${variable} ${init_month} ${SLURM_ARRAY_TASK_ID}"

# Run the process script
bash ${process_script} ${experiment} ${variable} ${init_month} ${SLURM_ARRAY_TASK_ID}


