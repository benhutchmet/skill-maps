#!/bin/bash
#SBATCH --job-name=sub-boot
#SBATCH --partition=short-serial
#SBATCH --mem=30000
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/logs/submit-bs-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/logs/submit-bs-%A_%a.err
#SBATCH --mail-user=benwhutchins25@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the usage messages
usage="Usage: ${variable} ${obs_var_name} ${region} ${season} ${forecast_range} ${start_year} ${end_year} ${lag} ${no_subset_members} ${method} ${nboot} ${level} ${full_period}"

# Check the number of CLI arguments
if [ "$#" -ne 13 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up the CLI arguments
variable=$1
obs_var_name=$2
region=$3
season=$4
forecast_range=$5
start_year=$6
end_year=$7
lag=$8
no_subset_members=$9
method=${10}
nboot=${11}
level=${12}
full_period=${13}

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/skill-maps/rose-suite-matching/process_bs_values.py"

# Echo the CLI arguments
echo "variable: ${variable}"
echo "obs_var_name: ${obs_var_name}"
echo "region: ${region}"
echo "season: ${season}"
echo "forecast_range: ${forecast_range}"
echo "start_year: ${start_year}"
echo "end_year: ${end_year}"
echo "lag: ${lag}"
echo "no_subset_members: ${no_subset_members}"
echo "method: ${method}"
echo "nboot: ${nboot}"
echo "level: ${level}"
echo "full_period: ${full_period}"

# Run the script
python ${process_script} ${variable} ${obs_var_name} ${region} ${season} ${forecast_range} ${start_year} ${end_year} ${lag} ${no_subset_members} ${method} ${nboot} ${level} ${full_period}