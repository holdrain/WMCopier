import os
from io import BytesIO
from lpips import LPIPS
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision.utils import save_image
import argparse
import csv

def load_image(url, size=None):
    response = requests.get(url,timeout=0.2,stream=True)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

@torch.no_grad()
def pred_latents(pipe, x_0, num_inference_steps, end_step, device):
    '''
    x_0 is with dimension of (b,c,h,w)
    '''
    intermediate_latents = []
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    latent_t = x_0.clone()
    
    for i in range(1, end_step+1):
        if i > end_step: continue
        t = timesteps[i]
        with torch.no_grad():
            noise_pred = pipe.unet(latent_t, t).sample
            current_t = max(0, t.item() - (1000//num_inference_steps))
            next_t = t 
            alpha_t = pipe.scheduler.alphas_cumprod[current_t]
            alpha_t_next = pipe.scheduler.alphas_cumprod[next_t.item()]
            latent_t = (latent_t - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        intermediate_latents.append(latent_t)
    return intermediate_latents

@torch.no_grad()
def sample(pipe, start_step=0, start_latents=None,
        num_inference_steps=100, device='cuda', return_pil=True, return_interm=False):
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = start_latents.clone()
    latents_list = []
    latents_list.append(latents)
    for i in range(start_step, num_inference_steps):
        t = pipe.scheduler.timesteps[i]
        latents = pipe.scheduler.scale_model_input(latents, t)
        noise_pred = pipe.unet(latents, t).sample
        prev_t = max(1, t.item() - (1000//num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt
        if return_interm:
            latents_list.append(torch.clamp(latents,-1,1).detach().cpu())
    image_tensor = torch.clamp(latents,-1,1)
    return image_tensor,latents_list

@torch.no_grad()
def sample_zp(x_0, pipe, start_step=0, start_latents=None,
        num_inference_steps=100, device='cuda', return_pil=True, 
        return_interm=False, fix=True, all_steps=False, use_lpips=False, likelihood=False, eta=1e-4, beta=1, M=20, v=0.01):
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = start_latents[-1].clone()
    latents_list = []
    likelihood_data = []
    for i in range(start_step, num_inference_steps):
        # denosing
        t = pipe.scheduler.timesteps[i]
        latents = pipe.scheduler.scale_model_input(latents, t)
        prev_t = max(1, t.item() - (1000//num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

        noise_pred = pipe.unet(latents, t).sample
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

        # likelihood
        if likelihood:
            noise = torch.randn_like(latents).to(device)
            noise_flattern = noise.view(noise.size(0),-1)
            noise_pred_flattern = noise_pred.view(noise_pred.size(0),-1)
            likelihood_value = -torch.norm(noise_pred - noise, dim=1) / (latents.shape[1] * latents.shape[2] * latents.shape[3])
            likelihood_data.append((t.item(), likelihood_value.mean().item()))

        # fix
        if fix:
            if all_steps:
                latents = fix_latent(latents,M,eta,beta,v,pipe,device,t,x_0)
            else:
                if i == num_inference_steps - 1:
                    latents = fix_latent(latents,M,eta,beta,v,pipe,device,t,x_0)
        if return_interm:
            latents_list.append(torch.clamp(latents,-1,1).detach().cpu())
    image_tensor = torch.clamp(latents,-1,1)

    if likelihood:
        with open("/home_new/dongziping/DiffusionWM/nipsresults/analysis_results/likelihood_wmdata.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Likelihood"])
            writer.writerows(likelihood_data)
    return image_tensor,latents_list

def fix_latent(latent,M,eta,beta,v,pipe,device,t,x_0):
    latent = latent.clone()
    for j in range(M):
        rn = torch.randn_like(latent)
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        noised_latent = pipe.scheduler.add_noise(latent, rn, t)
        ne_noise_pred = pipe.unet(noised_latent, t).sample
        score = -ne_noise_pred / (1 - alpha_t).sqrt()
        content_grad = latent - x_0
        correction = beta * content_grad - score
        latent = latent - eta * correction
    return latent

def estimate_score(x, pipe, t, alpha_t,device):
    noise = torch.randn_like(x).to(device)
    alpha_prod = scheduler.alphas_cumprod[t_step]
    with torch.no_grad():
        noise_pred = pipe.unet(x, t).sample
    score = -torch.norm(pred_noise - noise, dim=[1, 2, 3])
    return score.item()


def get_parser():
    parser = argparse.ArgumentParser(description='Sampling with smoothing parameters')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for sampling')
    parser.add_argument('--milestone', type=int, default=20,
                        help='Milestone for model checkpoint')
    parser.add_argument('--method', type=str, default='hidden',
                        help='Method type')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to process')
    parser.add_argument('--dataset', type=str, default='celebahq',
                        help='Dataset name')

    parser.add_argument('--smooth_method', type=str, default='ema',
                        choices=['ema', 'window', 'none'],
                        help='Smoothing method to use')
    parser.add_argument('--window_size', type=int, default=3,
                        help='Window size for moving average smoothing')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Alpha value for EMA smoothing')
    
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Interval for saving intermediate images')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
