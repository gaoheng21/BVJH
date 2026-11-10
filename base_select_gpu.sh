#!/bin/bash

# Make sure to give execute permission to this script
# chmod +x run_inference.sh

# Function to print memory status of all GPUs
print_gpu_memory() {
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

# Function to find an available GPU with at least 40 GB of free memory
find_available_gpu() {
    # Minimum required memory in MB (40 GB = 40960 MB)
    MIN_FREE_MEMORY=40960

    # Print all GPU memory status before selecting
    print_gpu_memory

    # Find a GPU with memory free >= 40GB
    AVAILABLE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                    awk -v min_mem=$MIN_FREE_MEMORY '$2 >= min_mem {print $1, $2}' | \
                    sort -n -k2 -r | \
                    head -n 1 | \
                    awk '{print $1}')
    
    if [ -z "$AVAILABLE_GPU" ]; then
        echo "No GPU with at least 40 GB of free memory found. Exiting."
        exit 1
    else
        echo "Using GPU $AVAILABLE_GPU with sufficient memory."
    fi
}

# Find an available GPU
find_available_gpu

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU