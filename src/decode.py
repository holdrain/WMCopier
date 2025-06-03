import argparse
import os

import torch
from tqdm.auto import tqdm

from data.dataset import CustomImageFolder
from utils.helpers import msg2str, set_seeds, str2msg
from utils.yml import load_config
from Watermarkschemes.helpers import target_message_dict, transforms_dict_decode,cal_tolerant,message_length_dict
from Watermarkschemes.model_choice import (
    get_DwtDct,
    get_DwtDctSvd,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
    get_titan,
)


def Options():
    parser = argparse.ArgumentParser(description="create dataset for few-shot learning...")

    parser.add_argument('--forgery_data', type=str, default='', help='data dir')
    parser.add_argument('--method', type=str, default='rivaGan', choices=['dwtdct','dwtdctsvd', 'hidden', 
                                                                      'hidden_combined', 'rivaGan','stable_signature','titan'], help='watermark scheme')
    # parser.add_argument('--output_dir', type=str,default=)
    parser.add_argument('--size', type=int, default=256, help="size of saved images")

    args = parser.parse_args()
    return args

def load_models(method, device):
    if method == 'dwtdctsvd':
        return get_DwtDctSvd(wm_text=target_message_dict[method], wm_type='bits')
    elif method == 'dwtdct':
        return get_DwtDct(wm_text=target_message_dict[method], wm_type='bits')
    elif method in ['hidden', 'hidden_combined']:
        cfgpath = "src/Watermarkschemes/config/hidden/hidden.yaml"
        cfg = load_config(cfgpath)
        return get_hiddenmodel(cfg.train, device)
    elif 'rivaGan' in method:
        return get_rivagan(wm_text=target_message_dict['rivaGan'])
    elif method == 'stable_signature':
        return get_stablesignature(device)
    elif method == 'titan':
        return get_titan()
    elif method == 'treering':
        return None, None
    else:
        raise ValueError(f"Unsupported method: {method}")

def decode_watermark(images_dir, method, decoder, device='cpu'):
    ds = CustomImageFolder(images_dir, transform=transforms_dict_decode[method])
    image_files = ds.filenames
    # print(image_files)
    output_file = os.path.join(images_dir, "decode.txt")
    tolerant_bits = cal_tolerant(message_length_dict[method])
    if method == "titan":
        ps = 0.0
        with tqdm(initial=0, total=len(image_files), desc="decoding watermark from forgery images...") as pbar:
            with open(output_file, 'a') as f:
                f.write("Path\tisgenerated\tconfidence\n")
                for idx in range(len(image_files)):
                    isgenerated,confidence = decoder(image_files[idx])
                    if confidence != "LOW" and isgenerated == 'GENERATED':
                        ps += 1.0
                    f.write(f"{image_files[idx]}\t{isgenerated}\t{confidence}\n")
                    pbar.update(1)
        return ps/len(image_files)
        
    else:
        target_message = target_message_dict[method]
        target_message_tensor = torch.Tensor(str2msg(target_message)).to(device)
        bitwise_accuracy_sum = 0.0
        fp_sum = 0
        with tqdm(initial=0, total=len(image_files), desc="decoding watermark from forgery images...") as pbar:
            with open(output_file, 'a') as f:
                f.write("Path\ttarge message\tdecoded message\tbitacc\n")
                for idx in range(len(image_files)):
                    f_image = ds.__getitem__(idx)
                    f_image = f_image.unsqueeze(0).to(device)
                    decoded_message = decoder(f_image)
                    decoded_message = decoded_message.round().clip(0, 1).long()
                    difference = (decoded_message != target_message_tensor).float()
                    if difference.sum().item() <= tolerant_bits:
                        fp_sum += 1
                    bitwise_accuracy = (1.0 - difference.mean()).item()
                    bitwise_accuracy_sum += bitwise_accuracy
                    
                    f.write(f"{image_files[idx]}\t{target_message}\t{msg2str(decoded_message)}\t{bitwise_accuracy}\n")
                    pbar.update(1)
                bitwise_accuracy_avg = bitwise_accuracy_sum / len(image_files)
                FPR = fp_sum / len(image_files)
                f.write(f"bitwise_accuracy:{round(bitwise_accuracy_avg, 4)}\n")
                f.write(f"fpr:{round(FPR, 4)}\n")
        return bitwise_accuracy_avg,FPR





if __name__ == '__main__':
    set_seeds(2024)
    opt = Options()
    device = torch.device('cuda:2' if opt.method in ['hidden','stable_signature'] else 'cpu')
    _,decoder = load_models(opt.method, device)
    bitwise_accuracy_avg,FPR = decode_watermark(opt.forgery_data, opt.method,decoder,device)
    print(round(bitwise_accuracy_avg, 4),round(FPR, 4))
