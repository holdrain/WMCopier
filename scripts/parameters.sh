#!/bin/bash

METHOD=dwtdct
NUM_LEVEL=5000
MILESTONE=20
NUM_IMAGES=100
BATCH_SIZE=10
NUM_INFERENCE_STEPS=100
DEVICE=cuda:2
SEED=2025
SAMPLE_TYPE=zp


# for M in 100; do
#     for beta in 10 50 100 150 200 250 300; do
#         for eta in 1e-4; do
#             for v in 0.01; do
#                 echo "Running with M=$M, beta=$beta, eta=$eta"
#                 python src/forgestep.py \
#                     --method $METHOD \
#                     --num_level $NUM_LEVEL \
#                     --milestone $MILESTONE \
#                     --num_inference_steps $NUM_INFERENCE_STEPS \
#                     --device $DEVICE \
#                     --batch_size $BATCH_SIZE \
#                     --num_images $NUM_IMAGES \
#                     --seed $SEED \
#                     --sample_type $SAMPLE_TYPE \
#                     --M $M \
#                     --beta $beta \
#                     --eta $eta \
#                     --v $v
#             done
#         done
#     done
# done