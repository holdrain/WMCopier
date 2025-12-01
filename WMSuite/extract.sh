METHOD="vine" # stegastamp, dwtdct, stable_signature,hidden,rivaGan, vine
DATAPATH="outputs/vine"
LOG_DIR="./outputs"
DEVICE="cuda:0"
BETA=1e-6

# generate authentic watermark data for testing the performance of defense

echo "Extracting for ${METHOD} on ${DATAPATH}"
python -m WMSuite.wm.cli.extract \
    --data_path ${DATAPATH} \
    --log_dir ${LOG_DIR} \
    --method ${METHOD} \
    --device ${DEVICE} \
    --beta ${BETA}
echo "Finish!"
