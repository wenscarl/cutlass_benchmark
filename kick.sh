#!/bin/bash

# Check if the input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 input_file"
    exit 1
fi

input_file=$1

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi
output_file="execution_times.txt"
> "$output_file"
# Read each line from the file and execute the command
while IFS= read -r line; do
    # Extract M, N, K values from the line (assuming they are space-separated)
    read -r M K N <<< "$line"

    # Run the command with the extracted values
    echo "Running command: runme $M $N $K"
#    ./a.out $M $N $K
#    output=$(./a.out $M $N $K)
    output=$(XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0" python benchmark.py --m $M --n $N --k $K --a gelu)
    execution_time=$(echo "$output" | grep -oP 'Execution Time \(ms\): \K\d+\.\d+')
    echo "$execution_time" >> "$output_file"

done < "$input_file"
echo "Execution times extracted to '$output_file'"

