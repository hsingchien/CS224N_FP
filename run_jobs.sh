#!/bin/bash

# Function to check if a GPU is idle
is_gpu_idle() {
    local gpu_id=$1
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    if [ "$utilization" -lt 10 ]; then
        echo 1
    else
        echo 0
    fi
}

# Function to run a job on a specific GPU
run_job_on_gpu() {
    local job=$1
    local gpu_id=$2
    CUDA_VISIBLE_DEVICES=$gpu_id $job &
    echo "Assigned job '$job' to GPU $gpu_id"
}

# List of jobs to run
job_queue=(
    "python train_model1.py"
    "python train_model2.py"
    "python train_model3.py"
    # Add more jobs as needed
)

# Main loop to check for idle GPUs and assign jobs
while [ ${#job_queue[@]} -gt 0 ]; do
    for gpu_id in 0 1; do
        if [ $(is_gpu_idle $gpu_id) -eq 1 ]; then
            job=${job_queue[0]}
            job_queue=("${job_queue[@]:1}")
            run_job_on_gpu "$job" $gpu_id
        fi
    done
    sleep 10  # Check every 10 seconds
done

echo "All jobs have been assigned and are running."
