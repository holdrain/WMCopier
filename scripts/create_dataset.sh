BATCH_SIZE=5
IMAGE_NUM=50           # image number in each message
TOTAL_NUM=50
METHOD="rivaGan"
DATASETS=("diffusiondb")


for dataset in "${DATASETS[@]}"; do
    echo "Creating dataset for ${METHOD} on ${dataset}"
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python src/data/create_dataset.py \
        --dataset ${dataset} \
        --method ${METHOD} \
        --img_num ${IMAGE_NUM} \
        --total_num ${TOTAL_NUM} \
        --batch_size ${BATCH_SIZE} \
        --filetype .png \
        --message default \
        --output_dir auxpath  # Auxiliary path
    echo "Finished ${method} on ${dataset}"
done
