#test

# usage bash test.bash 2-5

forecast_range=$1

# Set IFS to '-' and read into array
IFS='-' read -ra numbers <<< "$forecast_range"

# Extract numbers
first_number=${numbers[0]}
second_number=${numbers[1]}

echo "First number: $first_number"
echo "Second number: $second_number"