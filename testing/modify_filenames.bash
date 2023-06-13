#!/bin/bash

# Go to the directory containing the files (if not already there)
cd /work/scratch-nopw/benhutch/tas/HadGEM3-GC31-MM/outputs/mergetime

# Loop through each file that starts with 'psl_'
for file in psl_*.nc; do
    # Rename the file by replacing 'psl' with 'tas'
    mv "$file" "${file/psl_/tas_}"
done

# Go to the directory containing the files (if not already there)
cd /work/scratch-nopw/benhutch/tas/EC-Earth3/outputs/mergetime

# Loop through each file that starts with 'psl_'
for file in psl_*.nc; do
    # Rename the file by replacing 'psl' with 'tas'
    mv "$file" "${file/psl_/tas_}"
done

# Go to the directory containing the files (if not already there)
cd /work/scratch-nopw/benhutch/rsds/HadGEM3-GC31-MM/outputs/mergetime

# Loop through each file that starts with 'psl_'
for file in psl_*.nc; do
    # Rename the file by replacing 'psl' with 'rsds'
    mv "$file" "${file/psl_/rsds_}"
done

# Go to the directory containing the files (if not already there)
cd /work/scratch-nopw/benhutch/rsds/EC-Earth3/outputs/mergetime

# Loop through each file that starts with 'psl_'
for file in psl_*.nc; do
    # Rename the file by replacing 'psl' with 'rsds'
    mv "$file" "${file/psl_/rsds_}"
done