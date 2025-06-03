#!/bin/bash

# Basic settings
DEVICE="cuda:5"
BATCH_SIZE=1000
METHOD=("stable_signature")
DATASET=("mscoco" "celebahq" "imagenet" "diffusiondb")
SEED=2025


# Run experiments for different methods and datasets
for method in "${METHOD[@]}"; do
    for dataset in "${DATASET[@]}"; do
        echo "Running experiment for method: ${method}, dataset: ${dataset}"
        python src/meanforge.py \
            --device ${DEVICE} \
            --batch_size ${BATCH_SIZE} \
            --method ${method} \
            --dataset ${dataset} \
            --seed ${SEED} \
            --val_num_images 1000
        echo "Finished ${method} on ${dataset}"
    done
done
