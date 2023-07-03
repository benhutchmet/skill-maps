#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <month code>"
    exit 1
fi

declare -A months=( ["J"]=1 ["F"]=2 ["M"]=3 ["A"]=4 ["Y"]=5 ["U"]=6 ["L"]=7 ["G"]=8 ["S"]=9 ["O"]=10 ["N"]=11 ["D"]=12 )

start_month=${months[${1:0:1}]}
if [[ ${#1} -eq 2 ]]; then
    end_month=${months[${1:1:1}]}
elif [[ ${#1} -eq 3 ]]; then
    end_month=${months[${1:2:1}]}
else
    end_month=${months[${1:3:1}]}
fi

echo "Start month: $start_month"
echo "End month: $end_month"