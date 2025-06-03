
# hidden
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
#   --num_processes 4 \
#   --main_process_port 29502 \
#   new_train.py \
#   --config config/new_config.yaml\
#   --seed 2025 \
#   --train_batchsize 32 \
#   --num_samples 25 \
#   --image_size 128 \
#   --backbone unet \
#   --learning_rate 1e-4 \
#   --lr_scheduler no \
#   --train_num_steps 20000 \
#   --gradient_accumulate_every 16 \
#   --ema_decay 0.995 \
#   --amp \
#   --alpha 1.0 \
#   --ffl_w_start_step 0 \
#   --exp_name ddim_hidden_no \
#   --data_path /data/shared/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/aux/hidden \
#   --save_and_sample_every 1000 \
#   --train_num 1000
  



# rivaGAN
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#   --num_processes 4 \
#   --main_process_port 29503 \
#   new_train.py \
#   --config  config/new_config.yaml\
#   --seed 2025 \
#   --train_batchsize 8 \
#   --num_samples 16 \
#   --image_size 256 \
#   --backbone unet \
#   --learning_rate 1e-4 \
#   --lr_scheduler no \
#   --train_num_steps 20000 \
#   --gradient_accumulate_every 32 \
#   --ema_decay 0.995 \
#   --amp \
#   --alpha 1.0 \
#   --ffl_w_start_step 0 \
#   --exp_name ddim_rivagan_cosine \
#   --data_path /data/home/Huangpeng/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/aux/rivaGan/cluster \
#   --save_and_sample_every 1000 \
#   --train_num 10000 \
#   --beta_schedule cosine

# stable signature

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
#   --num_processes 6 \
#   --main_process_port 29501 \
#   new_train.py \
#   --config config/new_config.yaml \
#   --seed 2025 \
#   --train_batchsize 8 \
#   --num_samples 16 \
#   --image_size 512 \
#   --backbone unet \
#   --learning_rate 1e-5 \
#   --lr_scheduler no \
#   --train_num_steps 200000 \
#   --gradient_accumulate_every 4 \
#   --ema_decay 0.995 \
#   --amp \
#   --ffl_w 0.0 \
#   --alpha 1.0 \
#   --ffl_w_start_step 0 \
#   --exp_name ddim_stablesignature_no_new \
#   --data_path /mnt/shared/Dongziping/sharedcode/DiffForge/data/aux/stable_signature/111010110101000001010111010011010100010000100111 \
#   --save_and_sample_every 1000 \
#   --train_num 10000 \


  # titan
  # CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  # --num_processes 4 \
  # --main_process_port 29501 \
  # new_train.py \
  # --config config/new_config.yaml \
  # --seed 2025 \
  # --train_batchsize 1 \
  # --num_samples 16 \
  # --image_size 512 \
  # --backbone unet \
  # --learning_rate 1e-4 \
  # --lr_scheduler no \
  # --train_num_steps 20000 \
  # --gradient_accumulate_every 32 \
  # --ema_decay 0.995 \
  # --amp \
  # --alpha 1.0 \
  # --ffl_w_start_step 0 \
  # --exp_name ddim_titan_no \
  # --data_path /data/shared/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/titansum \
  # --save_and_sample_every 1000 \
  # --train_num 5000




  # finetune experiments
# rivaGAN
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#   --num_processes 4 \
#   --main_process_port 29501 \
#   new_train.py \
#   --config  newexperiments/ddim_rivagan_mul/03-25_22:48/Tconfig.json\
#   --seed 2025 \
#   --train_batchsize 32 \
#   --num_samples 16 \
#   --image_size 256 \
#   --backbone unet \
#   --learning_rate 1e-5 \
#   --lr_scheduler no \
#   --train_num_steps 200000 \
#   --gradient_accumulate_every 1 \
#   --ema_decay 0.995 \
#   --amp \
#   --alpha 1.0 \
#   --ffl_w_start_step 0 \
#   --exp_name ddim_rivagan_mul \
#   --data_path /data/shared/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/aux/rivagan_mul/rivaGan \
#   --save_and_sample_every 1000 \
#   --train_num 10000 \
#   --resume_from 26

  # CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  # --num_processes 4 \
  # --main_process_port 29501 \
  # new_train.py \
  # --config  config/new_config.yaml\
  # --seed 2025 \
  # --train_batchsize 8 \
  # --num_samples 16 \
  # --image_size 256 \
  # --backbone unet \
  # --learning_rate 1e-4 \
  # --lr_scheduler no \
  # --train_num_steps 5000 \
  # --gradient_accumulate_every 32 \
  # --ema_decay 0.995 \
  # --amp \
  # --alpha 1.0 \
  # --ffl_w_start_step 0 \
  # --exp_name ddim_rivagan_finetune10 \
  # --data_path /data/shared/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/aux/rivaGan/diffusiondb/rivaGan/00001100001001011110001101101110 \
  # --save_and_sample_every 50 \
  # --pretrained_ckp /data/shared/Dongziping/sharedcode/DiffusionWM/newexperiments/ddim_rivagan_no/01-08_20:29/ckp/model-20.pt \
  # --train_num 50


# dwtdct
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#   --num_processes 4 \
#   --main_process_port 29501 \
#   new_train.py \
#   --config  newexperiments/ddim_dwtdct_no_new/03-27_11:34/Tconfig.json \
#   --seed 2025 \
#   --train_batchsize 32 \
#   --num_samples 16 \
#   --image_size 256 \
#   --backbone unet \
#   --learning_rate 2e-4 \
#   --lr_scheduler no \
#   --train_num_steps 200000 \
#   --gradient_accumulate_every 1 \
#   --ema_decay 0.995 \
#   --amp \
#   --alpha 1.0 \
#   --ffl_w_start_step 0 \
#   --exp_name ddim_dwtdct_no_new \
#   --data_path /mnt/shared/Dongziping/sharedcode/DiffusionWM/AuxiliaryData/aux/dwtdctnew/new \
#   --save_and_sample_every 10000 \
#   --train_num 10000 \
#   --resume_from 10


  # rivaGan mul100 on 20000 images
  # CUDA_VISIBLE_DEVICES=0,1,4,3 accelerate launch \
  # --num_processes 4 \
  # --main_process_port 29501 \
  # src/train/new_train.py \
  # --config /home_new/dongziping/DiffusionWM/configs/new_config.yaml \
  # --seed 2025 \
  # --train_batchsize 8 \
  # --num_samples 16 \
  # --image_size 256\
  # --backbone unet \
  # --learning_rate 1e-4 \
  # --lr_scheduler no \
  # --train_num_steps 20000 \
  # --gradient_accumulate_every 16 \
  # --ema_decay 0.995 \
  # --amp \
  # --alpha 1.0 \
  # --ffl_w_start_step 0 \
  # --exp_name ddim_rivaGan_mul100_20000 \
  # --data_path /data/shared/Dongziping/sharedcode/DiffForge/data/aux/mul100_50000/rivaGan/diffusiondb \
  # --save_and_sample_every 1000 \
  # --train_num 20000


  CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --main_process_port 29501 \
  src/train/new_train.py \
  --config /data/shared/Dongziping/sharedcode/DiffusionWM/newexperiments/ddim_treering_no/04-22_11:46/Tconfig.json \
  --seed 2025 \
  --train_batchsize 1 \
  --num_samples 16 \
  --image_size 512\
  --backbone unet \
  --learning_rate 1e-4 \
  --lr_scheduler no \
  --train_num_steps 50000 \
  --gradient_accumulate_every 16 \
  --ema_decay 0.995 \
  --amp \
  --alpha 1.0 \
  --ffl_w_start_step 0 \
  --exp_name ddim_treering_5000 \
  --data_path /data/shared/Dongziping/sharedcode/DiffForge/data/treering/wm/seed2025 \
  --save_and_sample_every 1 \
  --train_num 5000 \
  --resume_from 15