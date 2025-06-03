#!/bin/bash

# Basic settings
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3")  # Using 4 GPUs
NUM_IMAGES=1000
MILESTONE=20
SEED=2025

# Diffusion parameters
NUM_INFERENCE_STEPS=100
METHODS=("rivaGan" "hidden" "dwtdct" "stable_signature")
NUM_LEVELS=("5000")
END_STEP=40

# Method-specific batch sizes
declare -A BATCH_SIZES=(
    ["hidden"]=100
    ["rivaGan"]=20
    ["stable_signature"]=5
    ["titan"]=5
    ["dwtdct"]=25
)

# Function to run experiment
run_experiment() {
    local device=$1
    local method=$2
    local num_level=$3
    
    echo "Running experiment for method: ${method} on ${device} with batch size ${BATCH_SIZES[$method]}"
    
    python src/inverseforge_t.py \
        --device ${device} \
        --batch_size ${BATCH_SIZES[$method]} \
        --num_images ${NUM_IMAGES} \
        --milestone ${MILESTONE} \
        --method ${method} \
        --num_inference_steps ${NUM_INFERENCE_STEPS} \
        --end_step ${END_STEP} \
        --seed ${SEED} \
        --num_level ${num_level} \
        --fix \
        --beta 100 \
        --M 100
    
    echo "Finished ${method} at level ${num_level} on ${device}"
}

# Run experiments in parallel
# for i in "${!METHODS[@]}"; do
#     method=${METHODS[$i]}
#     device=${DEVICES[$i % ${#DEVICES[@]}]}  # Distribute across GPUs
    
#     for num_level in "${NUM_LEVELS[@]}"; do
#         run_experiment "$device" "$method" "$num_level" &
#     done
# done

# # Wait for all background processes to finish
# wait
# echo "All experiments completed"


# Run experiments solely
method=${METHODS[0]}
device=${DEVICES[0]}  # Distribute across GPUs
num_level=${NUM_LEVELS[0]}
run_experiment "$device" "$method" "$num_level"



# for i in "${!NUM_LEVELS[@]}"; do
#     num_level=${NUM_LEVELS[$i]}
#     device=${DEVICES[$i % ${#DEVICES[@]}]}  # 每个 num_level 分一个 GPU

#     for method in "${METHODS[@]}"; do
#         run_experiment "$device" "$method" "$num_level" &
#     done
# done

# wait
# echo "All experiments completed"
