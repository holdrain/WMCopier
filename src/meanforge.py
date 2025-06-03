import os
import torch
import time
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data.dataset import CustomImageFolder
from decode import load_models
from setting import inversion_val_path, train_data_path,images_resolution
from utils.helpers import new_dir, str2msg, set_seeds
from utils.metrics import psnr_ssim
from Watermarkschemes.helpers import target_message_dict, transforms_dict_decode, transforms_dict_encode,cal_tolerant,message_length_dict,tensor_norm_dict
import torchvision.utils as vutils
from torchvision import transforms
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
def get_parser():
    parser = argparse.ArgumentParser(description='Mean forgery attack parameters')
    
    # Basic params
    parser.add_argument('--device', type=str, default='cuda:0',help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=10,help='Batch size for processing')
    parser.add_argument('--method', type=str, default='hidden',help='Watermark method')
    parser.add_argument('--dataset',type=str)
    # Attack params
    parser.add_argument('--val_num_images', type=int, default=1000,help='Number of validation images')
    parser.add_argument('--attack_num_images', nargs='+', type=int, default=[100],help='List of number of images for attack')
    # Other params
    parser.add_argument('--seed', type=int, default=2025,help='Random seed')
    
    return parser


if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    # Set random seed
    set_seeds(args.seed)
    
    # Load decoder
    _, decoder = load_models(args.method, args.device)
    
    # Create output directory and save experiment info
    args.save_dir = ""
    args.result_dir = "results/mean_results"
    
    # Process each attack size
    for attack_num_images in args.attack_num_images:
        dataset_start_time = time.time()
        save_dir = new_dir(os.path.join(args.save_dir,'forged_images',args.method,str(attack_num_images),args.dataset))
        
        # Load datasets
        wm_dataset = CustomImageFolder(train_data_path[args.method], transform=transforms_dict_decode[args.method],num=attack_num_images,random_sample=False)
        cl_dataset = CustomImageFolder("/data/shared/ImageNet-1k/val", transform=transforms_dict_encode[args.method],num=attack_num_images,random_sample=True)
        val_dataset = CustomImageFolder(inversion_val_path[args.dataset],transform=transforms_dict_encode[args.method],num=args.val_num_images,random_sample=True)
        
        wm_dl = DataLoader(wm_dataset, batch_size=args.batch_size)
        cl_dl = DataLoader(cl_dataset, batch_size=args.batch_size)
        val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Calculate mean residual image
        img_size = (1, 3, images_resolution[args.method], images_resolution[args.method])
        residual_image = torch.zeros(img_size, device=args.device)
        aimages_sum = torch.zeros(img_size, device=args.device)
        climages_sum = torch.zeros(img_size, device=args.device)
        tolerant_bits = cal_tolerant(message_length_dict[args.method])

        with tqdm(total=len(wm_dl), desc="calculating mean residual image") as pbar:
            for aimages, climages in zip(wm_dl, cl_dl):
                aimages = aimages.to(args.device)
                climages = climages.to(args.device)
                aimages_sum += aimages.sum(dim=0)
                climages_sum += climages.sum(dim=0)
                pbar.update(1)
            residual_image = (aimages_sum - climages_sum) / attack_num_images
            residual_image_norm = tensor_norm_dict[args.method](residual_image)
            l2_norm_residual = torch.norm(residual_image_norm, p=2)
            print(f"Norm 2 of residual image is: {l2_norm_residual}")
            to_pil = transforms.ToPILImage()
            image_pil = to_pil(residual_image_norm.squeeze(0)) 
            image_pil.convert("L").save(f"results/mean_results/{args.method}_residual.png")

        # Forge and evaluate
        target_message = target_message_dict[args.method]
        if args.method not in ['dwtdct','rivaGan']:
            target_message_tensor = torch.Tensor(str2msg(target_message)).to(args.device)
        else:
            target_message_tensor = torch.Tensor(str2msg(target_message))
        psnr_sum, ssim_sum = 0.0, 0.0
        bitwise_accuracy_sum = 0.0
        fp_sum = 0
        save_count = 0
        save_samples = True  
        
        with tqdm(total=len(val_dl), desc=f"forging images at attack num:{attack_num_images}") as pbar:
            for val_images in val_dl:
                val_images = val_images.to(args.device)
                forged_images = val_images + residual_image
                
                if args.method in ['dwtdct', 'rivaGan','rivaGanmul']:
                    val_images = val_images.detach().cpu()
                    forged_images = forged_images.detach().cpu()  
                for fdx in range(forged_images.shape[0]):
                    if save_samples:
                        save_count += 1
                        vutils.save_image(forged_images[fdx],os.path.join(save_dir,f"{save_count:05d}.png"),normalize=True)
                    decoded_message = decoder(forged_images[fdx].unsqueeze(0))
                    decoded_message = decoded_message.round().clip(0, 1).long()
                    difference = (decoded_message != target_message_tensor).float()
                    bitwise_accuracy = (1.0 - difference.mean()).item()
                    if difference.sum() <= tolerant_bits:
                        fp_sum += 1
                    bitwise_accuracy_sum += bitwise_accuracy
                    psnr_single, ssim_single = psnr_ssim(val_images[fdx], forged_images[fdx])
                    psnr_sum += psnr_single.item()
                    ssim_sum += ssim_single.item()
                pbar.update(1)   
        # Calculate averages
        bitwise_accuracy_avg = bitwise_accuracy_sum / args.val_num_images
        psnr_avg = psnr_sum / args.val_num_images
        ssim_avg = ssim_sum / args.val_num_images
        fpr = fp_sum / args.val_num_images
        
        # Save results
        results = {
            'dataset': args.dataset,
            'psnr': psnr_avg,
            'ssim': ssim_avg,
            'bitwise_accuracy': bitwise_accuracy_avg,
            'fpr': fpr,
        }
        results_path = os.path.join(args.result_dir, f"{args.method}_results.txt")
        with open(results_path, 'a') as f:
            f.write(f"Attack num images: {attack_num_images}\n")
            f.write(f"Results: {results}\n")
            f.write(f"Time taken: {time.time() - dataset_start_time} seconds\n")
            f.write("\n")
        print(f"Results saved to {results_path}")