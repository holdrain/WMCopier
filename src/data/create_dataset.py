import argparse

import torch
import sys
from utils.helpers import set_seeds
from Watermarkschemes.helpers import *
from Watermarkschemes.run import *
from setting import inversion_val_path,train_data_path

def Options():
    parser = argparse.ArgumentParser(description="create dataset for few-shot learning...")
    parser.add_argument('--dataset',type=str, required=True,help='dataset dir')
    parser.add_argument('--method', type=str, required=True, help='watermark scheme')
    parser.add_argument('--img_num', type=int, default=50)
    parser.add_argument('--total_num', type=int, default=10000)
    parser.add_argument('--batch_size',type=int,default=5,help='for batch size')
    parser.add_argument('--size',type=int, default=128,help="size of saved images")
    parser.add_argument('--filetype',choices=['.jpg','.png'],default=".png")
    parser.add_argument('--message',type=str,default='default',choices=['default','random'])
    parser.add_argument('--prompt',type=str,help='prompt for stable_signature')
    parser.add_argument('--nowm',action='store_true',help='clean version for stablesignature')
    parser.add_argument('--test',action='store_true',help='test mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = Options()
    device = torch.device('cuda:5')
    set_seeds(2025)
    if opt.test:
        opt.output_dir = ""
    else:
        opt.output_dir = ""
    if opt.method == 'stable_signature':
        run_stable_signature(opt,device,opt.prompt,nowm=opt.nowm)
    elif opt.method == 'dwtdct':
        run_dwtdct(opt,device,opt.message)
    elif opt.method == 'hidden':
        run_hidden(opt,device,opt.message)
    elif opt.method == 'rivaGan':
        run_rivagan(opt,device,opt.message)
    elif opt.method == 'clean':
        run_clean(opt,device)
    else:
        raise NotImplementedError
    