# this file is for analysis the image and bitacc in each denosing step 

import os
from utils.helpers import str2msg
import argparse
from Watermarkschemes.helpers import target_message_dict, cal_tolerant
import torch
from diffusers import DDIMPipeline, DDIMScheduler
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from decode import load_models
from data.dataset import CustomImageFolder
from inversion import pred_latents, sample ,sample_zp
from setting import inversion_val_path, model_choices, trained_config_file,beta_schedule_choices
from train.trainer import load_pretrained_model
from utils.helpers import new_dir, parse_yml, set_seeds
from utils.metrics import psnr_ssim
from Watermarkschemes.helpers import tensor_norm_dict, transforms_dict_inversion, message_length_dict
import time
from datetime import datetime
from torchvision.utils import save_image
import csv

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def get_parser():
    parser = argparse.ArgumentParser(description='Inversion forgery parameters')
    
    parser.add_argument('--device', type=str, default='cuda:0',help='Device to run on')
    
    parser.add_argument('--milestone', type=int, default=20,help='Milestone for model checkpoint')
    parser.add_argument('--method', type=str, default='hidden',help='Method type')
    parser.add_argument('--num_level', type=str,help='num for training')
    parser.add_argument('--num_inference_steps', type=int, default=100,help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=2025,help='Random seed')
    parser.add_argument('--sample_type', type=str, default='ori',help='Sample type')
    parser.add_argument('--num_images', type=int, default=10,help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=5,help='Batch size for processing')
    parser.add_argument('--eta', type=float, default=0.01,help='eta for sampling')
    parser.add_argument('--beta', type=float, default=0.6,help='beta for sampling')
    parser.add_argument('--M', type=int, default=30,help='M for sampling')
    parser.add_argument('--v', type=float, default=0.01,help='v for sampling')
    parser.add_argument('--all_steps', action='store_true',help='Whether to fix all steps for sampling')
    parser.add_argument('--save_dir', type=str, default='nipsresults/analysis_results',help='Directory to save results')
    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seeds(args.seed)
    args.config = trained_config_file[args.method][str(args.num_level)]
    Tconfig = EasyDict(parse_yml(args.config))
    save_dir = new_dir(os.path.join(args.save_dir, args.method, args.sample_type))
    results_file = os.path.join(args.save_dir, f"results-{args.method}-{args.sample_type}.csv")
    

    # model and pipeline
    model = model_choices[Tconfig.backbone](Tconfig.image_size)
    model.eval()
    model = model.to(args.device)
    load_pretrained_model(model, accelerator=None, 
                         pretrained_ckp=Tconfig.results_folder + f"/ckp/model-{args.milestone}.pt")
    _, decoder = load_models(args.method, args.device)
    
    if not hasattr(Tconfig, 'beta_schedule'):
        Tconfig.beta_schedule = 'linear'
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, trained_betas = beta_schedule_choices[Tconfig.beta_schedule]())
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)

    # Process each dataset
    tolerant_bits = cal_tolerant(message_length_dict[args.method])
    for dataset in ['mscoco']:
        data_dir = inversion_val_path[dataset]
        ds = CustomImageFolder(data_dir=data_dir,
                             transform=transforms_dict_inversion[args.method],
                             num=args.num_images,
                             random_sample=True)
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        # Forge
        end_step = 40
        start_step = args.num_inference_steps - end_step
        psnr_sum, ssim_sum = 0.0, 0.0
        target_message = target_message_dict[args.method]
        if args.method not in ['dwtdct','rivaGan']:
            target_message_tensor = torch.Tensor(str2msg(target_message)).to(args.device)
        else:
            target_message_tensor = torch.Tensor(str2msg(target_message))

        bitwise_accuracy_sum = 0.0
        psnr_sum, ssim_sum = 0.0, 0.0
        fp_sum = 0.0
        with open(results_file, mode='a', newline='') as file:
            for x_0 in dataloader:
                x_0 = x_0.to(args.device)
                intermediate_latents = pred_latents(pipeline, x_0, args.num_inference_steps, 
                                                    end_step, args.device)
                if args.sample_type == 'ori':
                    rec_images,rec_interms = sample(pipeline, start_step,intermediate_latents[-1],
                                                    args.num_inference_steps, args.device, return_interm=True)
                elif args.sample_type == 'zp':
                    rec_images,_ = sample_zp(x_0, pipeline, start_step,intermediate_latents,
                                    args.num_inference_steps, args.device, all_steps=args.all_steps, return_interm=True,eta=args.eta,
                                    beta=args.beta,M=args.M,v=args.v)
                else:
                    raise NotImplementedError(f"Unknown sample type: {args.sample_type}")
                
                rec_images_norm = tensor_norm_dict[args.method](rec_images)
                if args.method in ['dwtdct','rivaGan','rivaGanmul']:
                    rec_images_norm = rec_images_norm.detach().cpu()
    
                for r_idx in range(rec_images.shape[0]):
                    if args.method != 'titan':
                        decoded_message = decoder(rec_images_norm[r_idx].unsqueeze(0))
                        decoded_message = decoded_message.round().clip(0, 1).long()
                        difference = (decoded_message != target_message_tensor).float()
                        if difference.sum().item() <= tolerant_bits:
                            fp_sum += 1
                        bitwise_accuracy = (1.0 - difference.mean()).item()
                        bitwise_accuracy_sum += bitwise_accuracy
                    psnr_single, ssim_single = psnr_ssim(x_0[r_idx], rec_images[r_idx])
                    psnr_sum += psnr_single.item()
                    ssim_sum += ssim_single.item()
            # Calculate averages
            bitwise_accuracy_avg = bitwise_accuracy_sum / args.num_images
            psnr_avg = psnr_sum / args.num_images
            ssim_avg = ssim_sum / args.num_images
            writer = csv.writer(file)
            writer.writerow(["-" * 20])
            writer.writerow(["M", "beta", "eta"])
            writer.writerow([args.M, args.beta, args.eta])
            writer.writerow(["Dataset", "PSNR", "SSIM", "Bitwise Accuracy", "FPR"])
            writer.writerow([dataset, psnr_avg, ssim_avg, bitwise_accuracy_avg, fp_sum/args.num_images])


            