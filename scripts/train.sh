# rivaGAN
CUDA_VISIBLE_DEVICES=4,5,2,3 accelerate launch \
  --num_processes 2 \
  --main_process_port 29501 \
  src/train.py \
  --config  configs/config.yaml\
  --seed 2025 \
  --train_batchsize 8 \
  --num_samples 16 \
  --image_size 256 \
  --backbone unet \
  --learning_rate 1e-4 \
  --lr_scheduler no \
  --train_num_steps 20000 \
  --gradient_accumulate_every 2 \
  --ema_decay 0.995 \
  --amp \
  --exp_name ddim_rivagan_cosine \
  --data_path auxpath/train/rivaGan/diffusiondb/00001001111010001010110011101100 \
  --save_and_sample_every 10 \
  --train_num 50 \
  --beta_schedule cosine