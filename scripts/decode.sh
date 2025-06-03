# distored results
DISTORTIONS=("Blur" "JPEG" "Noise")
DATASETS=("celebahq" "imagenet" "mscoco")
NUM_LEVEL=("10000")
for method in "hidden" "rivaGan"; do
    for num_level in "${NUM_LEVEL[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for distortion in "${DISTORTIONS[@]}"; do
                size=256
                if [ "$method" = "hidden" ]; then
                    size=128
                elif [ "$method" = "stable_signature" ]; then
                    size=512
                fi
                
                CUDA_VISIBLE_DEVICES=3 python decode.py \
                    --method ${method} \
                    --forgery_data results/distored_results/${method}/${dataset}/${distortion} \
                    --size ${size}
            done
        done
    done
done
