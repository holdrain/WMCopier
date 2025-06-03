# this file is for inversion forgery

import os
from utils.helpers import str2msg
import argparse
from Watermarkschemes.helpers import target_message_dict, cal_tolerant
import torch
from diffusers import DDIMPipeline, DDIMScheduler
from easydict import EasyDict
from torch.utils.data import DataLoader
from decode import load_models
from dataset import CustomImageFolder
from inversion import pred_latents, sample
from setting import inversion_val_path, model_choices, trained_config_file,beta_schedule_choices
from trainer import load_pretrained_model
from utils.helpers import new_dir, parse_yml, set_seeds
from utils.metrics import psnr_ssim
from Watermarkschemes.helpers import tensor_norm_dict, transforms_dict_inversion, message_length_dict
import time
from datetime import datetime
from torchvision.utils import save_image

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def get_parser():
    parser = argparse.ArgumentParser(description='Inversion forgery parameters')
    
    parser.add_argument('--device', type=str, default='cuda:5',help='Device to run on')
    parser.add_argument('--num_images', type=int, default=1000,help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=10,help='Batch size for sampling')
    
    parser.add_argument('--milestone', type=int, default=20,help='Milestone for model checkpoint')
    parser.add_argument('--method', type=str, default='rivaGan',help='Method type')
    parser.add_argument('--num_level', type=str,help='num for training')
    
    parser.add_argument('--num_inference_steps', type=int, default=100,help='Number of inference steps')

    parser.add_argument('--seed', type=int, default=2025,help='Random seed')
    parser.add_argument('--show',action='store_true',help='test on selected images for show in paper')
    parser.add_argument('--save_sample',action='store_true',help='saving forged images(bool)')
    parser.add_argument('--save_dir',type=str, default='results/rebuttal/attack_results/forged_images')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seeds(args.seed)
    args.config = trained_config_file[args.method][str(args.num_level)]
    args.result_file = f"nipsresults/attack_results/{args.method}.txt"
    Tconfig = EasyDict(parse_yml(args.config))

    # model and pipeline
    model = model_choices[Tconfig.backbone](Tconfig.image_size)
    model.eval()
    model = model.to(args.device)
    load_pretrained_model(model, accelerator=None, 
                         pretrained_ckp=Tconfig.results_folder + f"/ckp/model-{args.milestone}.pt")
    _, decoder = load_models('rivaGan', args.device)
    if not hasattr(Tconfig, 'beta_schedule'):
        Tconfig.beta_schedule = 'linear'
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, trained_betas = beta_schedule_choices[Tconfig.beta_schedule]())
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)


    # Process each dataset
    tolerant_bits = cal_tolerant(message_length_dict[args.method])
    for dataset in ['mscoco','celebahq','imagenet','diffusiondb']:
        if args.save_sample:
            save_dir = new_dir(os.path.join(args.save_dir,args.method,args.num_level,dataset))
        dataset_start_time = time.time()
        data_dir = inversion_val_path[dataset] if args.show else inversion_val_path[dataset]
        ds = CustomImageFolder(data_dir=data_dir,
                             transform=transforms_dict_inversion[args.method],
                             num=args.num_images,
                             random_sample=True)
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        # Forge
        psnr_avg = 0.0
        end_step = args.search_up
        # target_message = target_message_dict[args.method]
        # target_message_tensor = torch.Tensor(str2msg(target_message))

        while psnr_avg < args.psnr_thre:
            end_step = end_step - args.search_step
            if end_step < args.search_lo:
                print("Forge failed!")
                break
            start_step = args.num_inference_steps - end_step
            psnr_sum, ssim_sum = 0.0, 0.0
            bitwise_accuracy_sum = 0.0
            fp_sum = 0
            save_count = 0
            
            for x_0 in dataloader:
                x_0 = x_0.to(args.device)
                intermediate_latents = pred_latents(pipeline, x_0, args.num_inference_steps, 
                                                 end_step, args.device)
                rec_images,_ = sample(pipeline, start_step,intermediate_latents[-1],
                                 args.num_inference_steps, args.device)
                rec_images_norm = tensor_norm_dict[args.method](rec_images)
                rec_images_norm = rec_images_norm.detach().cpu()

                message_list = ['00001011010001101011010010100000','00001100001001011110001101101110','00001101100000101000100001001001',
                  '00110010011010100011011100101111','01001010001111101001001001100000','01101111111000010110011010111100','01111101000001111001011110011100',
                  '11011100100001011000110111100100','11011100110100110010101010000101','11101000111111111011110100010000']

                for r_idx in range(rec_images.shape[0]):
                    if args.method != 'titan':
                        min_diff = 32
                        best_bitwise_accuracy = 0
                        for msg in message_list:
                            msg_tensor = torch.Tensor(str2msg(msg))
                            decoded_message = decoder(rec_images_norm[r_idx].unsqueeze(0))
                            decoded_message = decoded_message.round().clip(0, 1).long()
                            difference = (decoded_message != msg_tensor).float()
                            diff = difference.sum().item()
                            bitwise_accuracy = (1.0 - difference.mean()).item()
                            if diff < min_diff:
                                min_diff = diff
                                best_bitwise_accuracy = bitwise_accuracy
                        if min_diff <= tolerant_bits:
                            fp_sum += 1
                        bitwise_accuracy_sum += best_bitwise_accuracy
                    
                    if args.save_sample:
                        save_count += 1
                        save_image(rec_images[r_idx],os.path.join(save_dir,f"{save_count:05d}.png"),normalize=True)
                    psnr_single, ssim_single = psnr_ssim(x_0[r_idx], rec_images[r_idx])
                    psnr_sum += psnr_single.item()
                    ssim_sum += ssim_single.item()

            # Calculate averages
            bitwise_accuracy_avg = bitwise_accuracy_sum / args.num_images
            psnr_avg = psnr_sum / args.num_images
            ssim_avg = ssim_sum / args.num_images
            print(f"Current Step: {end_step}")
            print(f"Accuracy: {bitwise_accuracy_avg:.4f}, FPR:{fp_sum/args.num_images}, PSNR: {psnr_avg:.4f}")
        if psnr_avg >= args.psnr_thre:
            with open(args.result_file, 'a') as f:
                f.write(f"Num_level:{args.num_level}, Dataset: {dataset}, EndStep: {end_step} - "
                        f"Accuracy: {bitwise_accuracy_avg:.4f}, "
                        f"PSNR: {psnr_avg:.4f}, SSIM: {ssim_avg:.4f}\n")