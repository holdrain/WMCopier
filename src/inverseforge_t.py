# this file is for inversion forgery
import os
import argparse
from tqdm.auto import tqdm
from torchvision import transforms
from diffusers import DDIMPipeline, DDIMScheduler
from easydict import EasyDict
from torch.utils.data import DataLoader
from dataset import CustomImageFolder
from inversion import pred_latents, sample
from setting import model_choices, beta_schedule_choices,images_resolution
from trainer import load_pretrained_model
from utils.helpers import new_dir, parse_yml, set_seeds
from utils.metrics import psnr_ssim
from torchvision.utils import save_image
from datetime import datetime

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'



def get_parser():
    parser = argparse.ArgumentParser(description='Inversion forgery parameters')
    
    parser.add_argument('--device', type=str, default='cuda:0',help='Device to run on')
    parser.add_argument('--dataset', type=str, default='demo',help='Dataset to use for forgery attack')
    parser.add_argument('--num_images', type=int, default=1000,help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default=8,help='Batch size for sampling')
    
    parser.add_argument('--attacker_ckp', type=str, required=True, help='attacker checkpoint path')
    parser.add_argument('--method', type=str, default='hidden',help='watermark algorithm')
    
    # Attacker parameters
    parser.add_argument('--num_inference_steps', type=int, default=100,help='T for DDIM sampling')
    parser.add_argument('--end_step', type=int, default=40,help='Step size for shallow inversion')
    parser.add_argument('--L', type=int, default=100,help='M for refinement')
    parser.add_argument('--beta', type=float, default=100,help='beta for refinement')
    parser.add_argument('--eta', type=float, default=1e-4,help='eta for sampling')
    parser.add_argument('--refine', action='store_true',help='refinement(bool)')
    
    parser.add_argument('--seed', type=int, default=2025,help='Random seed')
    parser.add_argument('--save_dir',type=str, default='results',help='Directory to save forged samples and PSNR results')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seeds(args.seed)
    Tconfig = EasyDict(parse_yml('configs/config.yaml'))

    # model and pipeline
    model = model_choices['unet'](images_resolution[args.method])
    model.eval()
    model = model.to(args.device)
    load_pretrained_model(model, accelerator=None, pretrained_ckp=args.attacker_ckp)

    if not hasattr(Tconfig, 'beta_schedule'):
        Tconfig.beta_schedule = 'linear'
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000, trained_betas = beta_schedule_choices[Tconfig.beta_schedule]())
    pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)


    # dataset and dataloader
    transforms_inversion = transforms.Compose([
        transforms.Resize((images_resolution[args.method], images_resolution[args.method])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = CustomImageFolder( data_dir=args.dataset,
                            transform=transforms_inversion,
                            num=args.num_images,
                            random_sample=True,
                            seed = args.seed)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)


    # log
    save_dir = new_dir(os.path.join(args.save_dir,'forgedimages',args.method))
    psnr_sum, ssim_sum = 0.0, 0.0
    save_count = 0

    # shallow inversion and refinement
    start_step = args.num_inference_steps - args.end_step           
    for x_0 in tqdm(dataloader,desc='forging images...'):
        x_0 = x_0.to(args.device)
        intermediate_latents = pred_latents(pipeline, x_0, args.num_inference_steps, args.end_step, args.device)
        rec_images,_ = sample(x_0, pipeline, start_step, intermediate_latents,
                                args.num_inference_steps, args.device, refine=args.refine, L=args.L, eta=args.eta, beta=args.beta)
                
        rec_images_norm = rec_images * 0.5 + 0.5

        if args.method in ['dwtdct','rivaGan']:
            rec_images_norm = rec_images_norm.detach().cpu()
        for r_idx in range(rec_images.shape[0]):
            save_image(rec_images[r_idx],os.path.join(save_dir,f"{save_count:05d}.png"))
            save_count +=1
            psnr_single, ssim_single = psnr_ssim(x_0[r_idx], rec_images[r_idx])
            psnr_sum += psnr_single.item()
            ssim_sum += ssim_single.item()

        # Calculate averages
        psnr_avg = psnr_sum / args.num_images
        ssim_avg = ssim_sum / args.num_images

        # log
        with open(args.result_file, 'a') as f:
            f.write(f"=== Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write("\n[Configuration]\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n[Metrics]\n")
            f.write(f"{args.dataset}: PSNR={psnr_avg:.4f}, SSIM={ssim_avg:.4f}\n")
            f.write("\n" + "="*60 + "\n\n")
