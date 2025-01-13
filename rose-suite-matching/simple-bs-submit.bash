#!/bin/bash
#SBATCH --job-name=sub-simple-bs
#SBATCH --partition=high-mem
#SBATCH --mem=400000
#SBATCH --time=1200:00
#SBATCH -o /home/users/benhutch/skill-maps/logs/sub-simple-bs-%A_%a.out
#SBATCH -e /home/users/benhutch/skill-maps/logs/sub-simple-bs-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch simple-bs-submit.bash <nboot> <variable> <region> <season> <forecast_range>"

# Check the number of CLI arguments
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/skill-maps/rose-suite-matching/simple_process_bs_values.py"

# Run the script
python ${process_script} \
    $1 \
    $2 \
    $3 \
    $4 \
    $5


# End of file
echo "End of file"