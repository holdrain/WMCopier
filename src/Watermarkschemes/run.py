import os
from random import sample

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from torchvision import transforms
from dataset import CustomImageFolder
from utils.helpers import (
    generate_random_fingerprints,
    msg2str,
    new_dir,
    str2msg,
)
from utils.yml import load_config
from Watermarkschemes.helpers import target_message_dict, transforms_dict_encode
from Watermarkschemes.model_choice import (
    get_DwtDct,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
)
from setting import inversion_val_path, data_generate_path, images_resolution


def run_stable_signature(opt,device,prompts=None,nowm=False):
    '''stable_signature'''
    if prompts is None:
        prompts = ["Beautiful DSLR Photograph of a penguin on the beach, golden hour"]
    negative_prompt = ["blurry, ugly"]
    target_message = target_message_dict[opt.method]
    if nowm:
        target_message = 'clean'
    output_dir = new_dir(os.path.join(opt.output_dir,opt.method,target_message))
    encoder,_ = get_stablesignature(device,nowm=nowm)
    count = 0
    for pi,prompt in enumerate(prompts):
        idx = 0
        print(f"generating wm images by prompt{pi}")
        with tqdm(total = opt.img_num,desc=f"generating wm images by {opt.method}") as pbar:
            while (idx < opt.img_num):
                encoded_imgs = encoder([prompt] * opt.batch_size,
                                        negative_prompt=negative_prompt * opt.batch_size,
                                        size=images_resolution['stable_signature'])
                for pil in encoded_imgs:
                    enimg_path = os.path.join(output_dir,f"{count:04d}"+opt.filetype)
                    pil.save(enimg_path)
                    idx += 1
                    count += 1
                    pbar.update(1)



def run_hidden(opt,device,message='default'):
    cfgpath = "config/hidden/hidden.yaml"
    cfg = load_config(cfgpath)
    encoder,decoder = get_hiddenmodel(cfg.train,device)
    if opt.test:
        dataset_dir = inversion_val_path[opt.dataset]
    else:
        dataset_dir = data_generate_path[opt.dataset]
    ds = CustomImageFolder(dataset_dir,transform=transforms_dict_encode[opt.method],num=opt.img_num,random_sample=True)
    dl = DataLoader(ds, batch_size = opt.batch_size, shuffle=False, num_workers=0)
    count = 0
    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        for idx,image in enumerate(dl):
            if message == 'random':
                if idx % opt.img_num == 0:
                    target_message = msg2str(generate_random_fingerprints(len(target_message_dict[opt.method])))
            else:
                target_message = target_message_dict[opt.method] 
            output_dir = new_dir(os.path.join(opt.output_dir,opt.method,opt.dataset,target_message))
            image = image.to(device)
            target_message_tensor = torch.tensor(str2msg(target_message)).repeat(image.shape[0],1).to(device)
            encoded_img = encoder(image,target_message_tensor)
            for idx in range(encoded_img.shape[0]):
                enimg_path = os.path.join(output_dir,f"{count:04d}"+opt.filetype)
                vutils.save_image(encoded_img[idx],enimg_path,normalize=True)
                count += 1
                pbar.update(1)


def run_rivagan(opt,device,message='default'):
    if opt.test:
        dataset_dir = inversion_val_path[opt.dataset]
    else:
        dataset_dir = data_generate_path[opt.dataset]
    ds = CustomImageFolder(dataset_dir,transform=transforms_dict_encode[opt.method],num=opt.total_num,random_sample=True)
    dl = DataLoader(ds, batch_size = 1,shuffle=False, num_workers=0)
    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        for idx,image in enumerate(dl):
            if message == 'random':
                if idx % opt.img_num == 0:
                    target_message = msg2str(generate_random_fingerprints(len(target_message_dict[opt.method])))
            else:
                target_message = target_message_dict[opt.method] 
            output_dir = new_dir(os.path.join(opt.output_dir,opt.method,opt.dataset,target_message))
            encoder,decoder = get_rivagan(wm_text=target_message)
            encoded_img = encoder(image)
            enimg_path = os.path.join(output_dir,f"{(idx % opt.img_num):04d}"+opt.filetype)
            vutils.save_image(encoded_img,enimg_path,normalize=True)
            pbar.update(1)
    

def run_dwtdct(opt,device,message='default'):
    if opt.test:
        dataset_dir = inversion_val_path[opt.dataset]
    else:
        dataset_dir = data_generate_path[opt.dataset]
    ds = CustomImageFolder(dataset_dir,transform=transforms_dict_encode[opt.method],num=opt.total_num,random_sample=True)
    dl = DataLoader(ds, batch_size = 1, shuffle=False, num_workers=0)
    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        for idx,image in enumerate(dl):
            if message == 'random':
                # change target message every opt.img_num
                if idx % opt.img_num == 0:
                    target_message = msg2str(generate_random_fingerprints(len(target_message_dict[opt.method])))
            else:
                target_message = target_message_dict[opt.method]
            encoder,decoder = get_DwtDct(wm_text=target_message,wm_type='bits')
            output_dir = new_dir(os.path.join(opt.output_dir,opt.method,opt.dataset,target_message))
            encoded_img = encoder(image)
            enimg_path = os.path.join(output_dir,f"{(idx % opt.img_num):04d}"+opt.filetype)
            vutils.save_image(encoded_img,enimg_path,normalize=True)
            pbar.update(1)

def run_clean(opt,device,message='default'):
    if opt.test:
        dataset_dir = inversion_val_path[opt.dataset]
    else:
        dataset_dir = data_generate_path[opt.dataset]
    ds = CustomImageFolder(dataset_dir,transform=transforms_dict_encode[opt.method],num=opt.total_num,random_sample=True)
    dl = DataLoader(ds, batch_size = 1, shuffle=False, num_workers=0)
    with tqdm(total=ds.__len__(),desc=f"generating wm images by {opt.method}") as pbar:
        for idx,image in enumerate(dl):
            output_dir = new_dir(os.path.join(opt.output_dir,opt.method,opt.dataset,'clean256'))
            encoded_img = image
            enimg_path = os.path.join(output_dir,f"{(idx % opt.img_num):04d}"+opt.filetype)
            vutils.save_image(encoded_img,enimg_path,normalize=True)
            pbar.update(1)

