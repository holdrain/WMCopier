#!/bin/bash

# Basic settings
DEVICE="cuda:0"
BATCH_SIZE=4
NUM_IMAGES=50
SEED=2025
END_STEP=40
SAVE_DIR='results'
METHOD='dwtdct'


python src/inverseforge_t.py \
        --device ${DEVICE} \
        --batch_size ${BATCH_SIZE} \
        --num_images ${NUM_IMAGES} \
        --method ${METHOD} \
        --end_step ${END_STEP} \
        --seed ${SEED} \
        --refinement \
        --beta 100 \
        --L 100 \
        --save_dir ${SAVE_DIR}

