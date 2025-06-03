BATCH_SIZE=5
IMAGE_NUM=5000           # image number in each message
TOTAL_NUM=5000
SIZE=256
METHOD="rivaGan"
DATASETS=("diffusiondb")

# generate authentic watermark data for testing the performance of defense
for dataset in "${DATASETS[@]}"; do
    OUTPUT_DIR=""
    echo "Creating dataset for ${METHOD} on ${dataset}"
    python src/data/create_dataset.py \
        --dataset ${dataset} \
        --method ${METHOD} \
        --img_num ${IMAGE_NUM} \
        --total_num ${TOTAL_NUM} \
        --batch_size ${BATCH_SIZE} \
        --size ${SIZE} \
        --filetype .png \
        --message random
    echo "Finished ${method} on ${dataset}"
done
