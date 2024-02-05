#!/bin/bash
#SBATCH --job-name=sub-boot
#SBATCH --partition=high-mem
#SBATCH --mem=60000
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/submit-bootstrap-test/logs/submit-bootstrap-test-%A_%a.out
#SBATCH -e /home/users/benhutch/submit-bootstrap-test/logs/submit-bootstrap-test-%A_%a.err
#SBATCH --mail-user=benwhutchins25@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the Usage message
usage_msg="Usage: submit-bootstrap-test.bash <variable> <obs_var_name> <region> <season> <forecast_range> <start_year> <end_year> <lag> <no_subset_members> <method> <nboot> <level> <full_period>"

# Echo the number of CLIs
echo "Number of CLIs: $#"

# Check the number of CLIs
if [ $# -ne 13 ]; then
    echo "Incorrect number of CLIs"
    echo $usage_msg
    exit 1
fi

# Set the CLIs
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

# Echo the CLIs
echo "variable: $variable"
echo "obs_var_name: $obs_var_name"
echo "region: $region"
echo "season: $season"
echo "forecast_range: $forecast_range"
echo "start_year: $start_year"
echo "end_year: $end_year"
echo "lag: $lag"
echo "no_subset_members: $no_subset_members"
echo "method: $method"
echo "nboot: $nboot"
echo "level: $level"
echo "full_period: $full_period"

# Set up the process script
process_script="/home/users/benhutch/skill-maps/rose-suite-matching/process_bs_values.py"

# Load jaspy
module load jaspy

# Run the script
python $process_script $variable $obs_var_name $region $season $forecast_range $start_year $end_year $lag $no_subset_members $method $nboot $level $full_period
