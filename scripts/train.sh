

# Please specify your dataset path here
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  --main_process_port 29501 \
  src/train.py \
  --config  configs/config.yaml\
  --seed 2025 \
  --train_batchsize 12 \
  --num_samples 16 \
  --image_size 256 \
  --backbone unet \
  --learning_rate 1e-4 \
  --lr_scheduler no \
  --train_num_steps 20000 \
  --gradient_accumulate_every 6 \
  --ema_decay 0.995 \
  --amp \
  --exp_name test \
  --data_path /home/dongziping/WMSuite/sampled_train_5000 \
  --save_and_sample_every 10000 \
  --train_num 5000 \
  --beta_schedule linear