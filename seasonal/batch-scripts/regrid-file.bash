#!/bin/bash
#
# regid-file.bash
# 
# Regrids all ensemble members of a variable for a given experiment and year
#
# Usage: bash regrid-file.bash <experiment> <variable> <nens> <year>
#
# Example: bash regrid-file.bash ASF20C SLP 50 1901
#

# Set up the usage message
usage="Usage: bash regrid-file.bash <experiment> <variable> <nens> <year>"

# Check if the correct number of arguments is passed
if [ $# -ne 4 ]; then
    echo ${usage}
    echo "Example: bash regrid-file.bash ASF20C SLP 50 1901"
    exit 1
fi

# Extract the arguments
experiment=$1
variable=$2
nens=$3
year=$4