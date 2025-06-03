#!/bin/bash

# distortion
JPEG_RATIO=90
GAUSSIAN_BLUR_R=1
GAUSSIAN_STD=0.05
NUM_LEVEL=5000
NUM_IMAGES=1000
DEVICE="cuda:4"

METHOD_LIST=("rivaGan")
DATASET_LIST=("mscoco" "celebahq" "imagenet" "diffusiondb")

for METHOD in "${METHOD_LIST[@]}"
do
  for DATASET in "${DATASET_LIST[@]}"
  do
    echo "[$METHOD | $DATASET] Running JPEG distortion..."
    python src/distort.py \
      --method $METHOD \
      --dataset $DATASET \
      --num_level $NUM_LEVEL \
      --jpeg_ratio $JPEG_RATIO \
      --num_images $NUM_IMAGES \
      --device $DEVICE

    echo "[$METHOD | $DATASET] Running Gaussian Blur distortion..."
    python src/distort.py \
      --method $METHOD \
      --dataset $DATASET \
      --num_level $NUM_LEVEL \
      --gaussian_blur_r $GAUSSIAN_BLUR_R \
      --num_images $NUM_IMAGES \
      --device $DEVICE

    echo "[$METHOD | $DATASET] Running Gaussian Noise distortion..."
    python src/distort.py \
      --method $METHOD \
      --dataset $DATASET \
      --num_level $NUM_LEVEL \
      --gaussian_std $GAUSSIAN_STD \
      --num_images $NUM_IMAGES \
      --device $DEVICE

    echo "[$METHOD | $DATASET] Running Brightness distortion..."
    python src/distort.py \
      --method $METHOD \
      --dataset $DATASET \
      --num_level $NUM_LEVEL \
      --num_images $NUM_IMAGES \
      --device $DEVICE \
      --brightness_factor 6
  done
done
