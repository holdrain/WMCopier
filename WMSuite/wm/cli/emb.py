import argparse

import torch
from wm.core.helpers import *
from wm.core.encode import *

def Options():
    parser = argparse.ArgumentParser(description="create dataset for few-shot learning...")
    parser.add_argument('--dataset',type=str, required=True,help='dataset dir')
    parser.add_argument('--num',type=int,default=5000,help='total number of images to embed watermarks')
    parser.add_argument('--output_dir',type=str,required=True,help='output dir')
    parser.add_argument('--method', type=str, required=True, help='watermark scheme')
    parser.add_argument('--batch_size',type=int,default=5,help='for batch size')
    parser.add_argument('--filetype',choices=['.jpg','.png'],default=".png")
    parser.add_argument('--message_mode',type=str,default='default',choices=['default','random'])
    parser.add_argument('--device',type=str,default='cuda:0',help='device to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = Options()
    set_seeds(2025)
    if opt.method == 'stable_signature':
        run_stable_signature(opt)
    elif opt.method == 'dwtdct':
        run_dwtdct(opt)
    elif opt.method == 'stegastamp':
        run_stegastamp(opt)
    elif opt.method == 'hidden':
        run_hidden(opt)
    elif opt.method == 'rivaGan':
        run_rivagan(opt)
    elif opt.method == 'vine':
        run_vine(opt)
    else:
        raise NotImplementedError
    