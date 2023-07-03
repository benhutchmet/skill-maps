#!/bin/bash
#
# multi-model.sel-region-forecast-range-season.bash
#
# For example: multi-model.sel-region-forecast-range-season.bash HadGEM3-GC31-MM 1960 1 psl north-atlantic 2-5 DJFM
#
# NOTE: Seasons should be formatted using: JFMAYULGSOND
#

# check if the correct number of arguments have been passed
if [ $# -ne 7 ]; then
    echo "Usage: multi-model.sel-region-forecast-range-season.bash <model> <initialization-year> <run-number> <variable> <region> <forecast-range> <season>"
    exit 1
fi

# extract the data from the command line
model=$1
year=$2
run=$3
variable=$4
region=$5
forecast_range=$6
season=$7

# set up the gridspec file
grid="/home/users/benhutch/gridspec/gridspec-${region}.txt"

# if the gridspec file does not exist, exit
if [ ! -f "$grid" ]; then
    echo "[ERROR] Gridspec file not found"
    exit 1
fi

# echo the gridspec file path
echo "Gridspec file: $grid"

# model name and family
# set up an if loop for the model name
if [ "$model" == "BCC-CSM2-MR" ]; then
    model_group="BCC"
elif [ "$model" == "MPI-ESM1-2-HR" ]; then
    model_group="MPI-M"
elif [ "$model" == "CanESM5" ]; then
    model_group="CCCma"
elif [ "$model" == "CMCC-CM2-SR5" ]; then
    model_group="CMCC"
elif [ "$model" == "HadGEM3-GC31-MM" ]; then
    model_group="MOHC"
elif [ "$model" == "EC-Earth3" ]; then
    model_group="EC-Earth-Consortium"
elif [ "$model" == "EC-Earth3-HR" ]; then
    model_group="EC-Earth-Consortium"
elif [ "$model" == "MRI-ESM2-0" ]; then
    model_group="MRI"
elif [ "$model" == "MPI-ESM1-2-LR" ]; then
    model_group="DWD"
elif [ "$model" == "FGOALS-f3-L" ]; then
    model_group="CAS"
elif [ "$model" == "CNRM-ESM2-1" ]; then
    model_group="CNRM-CERFACS"
elif [ "$model" == "MIROC6" ]; then
    model_group="MIROC"
elif [ "$model" == "IPSL-CM6A-LR" ]; then
    model_group="IPSL"
elif [ "$model" == "CESM1-1-CAM5-CMIP5" ]; then
    model_group="NCAR"
elif [ "$model" == "NorCPM1" ]; then
    model_group="NCC"
else
    echo "[ERROR] Model not recognised"
    exit 1
fi

# set up the files to be processed
# if the variable is psl
if [ "$variable" == "psl" ]; then
    # if the model is BCC-CSM2-MR or MPI-ESM1-2-HR or CanESM5 or CMCC-CM2-SR5
    if [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
    # set up the input files
    files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${year}-r${run}i?p?f?/Amon/psl/g?/files/d????????/*.nc"
    # for the single file models downloaded from ESGF
    elif [ "$model" == "MPI-ESM1-2-LR" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MIROC6" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "NorCPM1" ]; then
    # set up the input files from xfc
    # check that this returns the files
    files="/work/xfc/vol5/user_cache/benhutch/$model_group/$model/psl_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*p*f*_g*_*.nc"
    # if the model is HadGEM3 or EC-Earth3
    elif [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
    # set up the input files
    files="/work/scratch-nopw/benhutch/psl/${model}/outputs/mergetime/psl_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*.nc"
    else
    echo "[ERROR] Model not recognised for variable psl"
    exit 1
    fi
# if the variable is tas
elif [ "$variable" == "tas" ]; then
    # set up the models that have tas on JASMIN
    # these include NorCPM1, IPSL-CM6A-LR, MIROC6, BCC-CSM2-MR, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5, EC-Earth3, HadGEM3-GC31-MM 
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
    # set up the input files
    files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${year}-r${run}i?p?f?/Amon/tas/g?/files/d????????/*.nc"
    # for the files downloaded from ESGF
    # which includes CESM1-1-CAM5-CMIP5, FGOALS-f3-L, MPI-ESM1-2-LR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MPI-ESM1-2-LR" ]; then
    # set up the input files from xfc
    files="/work/xfc/vol5/user_cache/benhutch/tas/${model}/tas_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*p*f*_g*_*.nc"
    # if the model is HadGEM3 or EC-Earth3
    elif [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
    # set up the input files
    files="/work/scratch-nopw/benhutch/tas/${model}/outputs/mergetime/tas_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*.nc"
    else
    echo "[ERROR] Model not recognised for variable tas"
    exit 1
    fi
# if the variable is rsds
elif [ "$variable" == "rsds" ]; then
    # set up the models that have rsds on JASMIN
    # thes incldue NorCPM1, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
    # set up the input files
    files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/dcppA-hindcast/s${year}-r${run}i?p?f?/Amon/rsds/g?/files/d????????/*.nc"
    # for the files downloaded from ESGF
    # for models CESM1-1-CAM5-CMIP5, FGOALS-f3-L, BCC-CSM2-MR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ]; then
    # set up the input files from xfc
    files="/work/xfc/vol5/user_cache/benhutch/rsds/${model}/rsds_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*p*f*_g*_*.nc"
    elif [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
    # set up the input files
    files="/work/scratch-nopw/benhutch/rsds/${model}/outputs/mergetime/rsds_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*.nc"
    else
    echo "[ERROR] Model not recognised for variable rsds"
    exit 1
    fi
# if the variable is sfcWind - currently not downloaded
elif [ "$variable" == "sfcWind" ]; then
    # set up the models which have sfcWind on JASMIN
    # this includes HadGEM3-GC31-MM, EC-Earth3
    if [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
    # set up the input files
    files="/work/scratch-nopw/benhutch/sfcWind/${model}/outputs/mergetime/sfcWind_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*.nc"
    # set up the models downloaded from ESGF
    # this includes CESM1-1-CAM5-CMIP5, FGOALS-f3-L, BCC-CSM2-MR, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
    # set up the input files from xfc
    files="/work/xfc/vol5/user_cache/benhutch/sfcWind/${model}/sfcWind_Amon_${model}_dcppA-hindcast_s${year}-r${run}i*p*f*_g*_*.nc"
    else
    echo "[ERROR] Model not recognised for variable sfcWind"
    exit 1
    fi
else
    echo "[ERROR] Variable not recognised"
    exit 1
fi

# activate the environment containing cdo
module load jaspy

# set up the output directory
OUTPUT_DIR="/work/scratch-nopw/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/outputs"
mkdir -p $OUTPUT_DIR

# loop through the files and process them
for INPUT_FILE in $files; do

    # set up the output file names
    echo "Processing $INPUT_FILE"
    base_fname=$(basename "$INPUT_FILE")
    regridded_fname="regridded-${base_fname}"
    season_fname="years-${forecast_range}-${season}-${region}-${base_fname}"
    TEMP_FILE="$OUTPUT_DIR/temp-${base_fname}"
    REGRIDDED_FILE="$OUTPUT_DIR/${regridded_fname}"
    OUTPUT_FILE="$OUTPUT_DIR/${season_fname}"
    MEAN_FILE="$OUTPUT_DIR/mean-${season_fname}"

    # Regrid using bilinear interpolation
    # Selects region (as long as x and y dimensions divide by 2.5)
    cdo remapbil,$grid $INPUT_FILE $REGRIDDED_FILE

    # Extract initialization year from the input file name
    year=$(basename "$REGRIDDED_FILE" | sed 's/.*_s\([0-9]\{4\}\)-.*/\1/')

    # Set IFS to '-' and read into array
    IFS='-' read -ra numbers <<< "$forecast_range"

    # Extract numbers
    forecast_start_year=${numbers[0]}
    forecast_end_year=${numbers[1]}

    echo "First number: $forecast_start_year"
    echo "Second number: $forecast_end_year"

    # modify the start_year -1
    start_year=$((forecast_start_year - 1))


    # Declare the month codes
    declare -A months=( ["J"]=1 ["F"]=2 ["M"]=3 ["A"]=4 ["Y"]=5 ["U"]=6 ["L"]=7 ["G"]=8 ["S"]=9 ["O"]=10 ["N"]=11 ["D"]=12 )

    # Extract the month code from the season
    start_month=${months[${season:0:1}]}
    if [[ ${#season} -eq 2 ]]; then
    end_month=${months[${season:1:1}]}
    elif [[ ${#season} -eq 3 ]]; then
    end_month=${months[${season:2:1}]}
    else
    end_month=${months[${season:3:1}]}
    fi

    # if start_month or end_month is a single digit, add a 0 to the start
    if [[ ${#start_month} -eq 1 ]]; then
    start_month="0${start_month}"
    fi

    # for end month, add a 0 to the start
    if [[ ${#end_month} -eq 1 ]]; then
    end_month="0${end_month}"
    fi

    echo "Start month: $start_month"
    echo "End month: $end_month"

    # Calculate the start and end dates for the DJFM season
    start_date=$((year + start_year))"-${start_month}-01"
    end_date=$((year + forecast_end_year))"-${end_month}-31"

    echo "Start date: $start_date"
    echo "End date: $end_date"

    # convert from JFMAYULGSOND to JFMAMJJASOND format
    # if Y is in the season, replace with M
    if [[ $season == *"Y"* ]]; then
    season=${season//Y/M}
    fi

    # if U is in the season, replace with J
    if [[ $season == *"U"* ]]; then
    season=${season//U/J}
    fi

    # if L is in the season, replace with J
    if [[ $season == *"L"* ]]; then
    season=${season//L/J}
    fi

    # if G is in the season, replace with A
    if [[ $season == *"G"* ]]; then
    season=${season//G/A}
    fi

    # echo the season
    echo "Season: $season"

    # Constrain the input file to the DJFM season
    cdo select,season=${season} "$REGRIDDED_FILE" "$TEMP_FILE"

    # Extract the 2-9 years using cdo
    cdo select,startdate="$start_date",enddate="$end_date" "$TEMP_FILE" "$OUTPUT_FILE"

    # Take the time mean of the output file
    cdo timmean "$OUTPUT_FILE" "$MEAN_FILE"

    # Remove the temporary, regridded, and original output files
    rm "$TEMP_FILE"
    rm "$REGRIDDED_FILE"
    rm "$OUTPUT_FILE"

    echo "[INFO] Finished processing: $INPUT_FILE"
done