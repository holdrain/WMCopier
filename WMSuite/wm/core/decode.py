import os
import torch
from tqdm.auto import tqdm
from datetime import datetime

from WMSuite.wm.core.dataset import CustomImageFolder
from WMSuite.wm.core.helpers import msg2str, str2msg
from WMSuite.wm.core.yml import load_config
from WMSuite.wm.core.helpers import transforms_dict_decode,cal_tolerant,message_length_dict,message_dict,set_seeds
from WMSuite.wm.core.model_choice import (
    get_DwtDct,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
    get_vine,
    get_Stegastamp,
)


def load_models(method, device):
    if method == 'dwtdct':
        return get_DwtDct(wm_text=message_dict[method], wm_type='bits')
    elif method == 'hidden':
        cfgpath = "WMSuite/wm/algorithms/config/hidden/hidden.yaml"
        cfg = load_config(cfgpath)
        return get_hiddenmodel(cfg.train, device)
    elif method == 'stegastamp':
        return get_Stegastamp(device)
    elif method == 'rivaGan':
        return get_rivagan(wm_text=message_dict[method])
    elif method == 'stable_signature':
        return get_stablesignature(device)
    elif method == 'vine':
        return get_vine(device)
    else:
        raise ValueError(f"Unsupported method: {method}")
    

def decode_single_image(image_tensor, target_message_tensor, method, device, decoder, tolerant_bits):
    """
    Decode watermark from a single transformed image tensor.

    Args:
        image_tensor (torch.Tensor): 1xCxHxW tensor already on device
        target_message_tensor (torch.Tensor): ground truth message tensor on device
        method (str): watermark method
        device (str): e.g. "cuda:0" or "cpu"
        decoder: watermark decoder model
        tolerant_bits (int): number of bits tolerated for TP
    
    Returns:
        bitwise_accuracy (float)
        is_tp (bool)
        decoded_message_tensor (torch.Tensor)
    """
    
    # Ensure tensor on correct device
    image_tensor = image_tensor.to(device)

    # Decode
    decoded_message = decoder(image_tensor)
    if method != "stegastamp":
        decoded_message = decoded_message.round().clip(0, 1).long()
    else:
        decoded_message = (decoded_message > 0).long()

    # Compare with target
    diff = (decoded_message != target_message_tensor).float()
    bitwise_accuracy = (1.0 - diff.mean()).item()
    is_watermarked = (diff.sum().item() <= tolerant_bits)

    return bitwise_accuracy, is_watermarked, decoded_message

def decode_from_folder(opt, decoder):
    ds = CustomImageFolder(opt.data_path, transform=transforms_dict_decode[opt.method])
    image_files = ds.filenames
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_file = os.path.join(opt.log_dir, f"results_{ts}.txt")

    tolerant_bits = cal_tolerant(message_length_dict[opt.method], beta=opt.beta)

    if opt.method in ['dwtdct', 'rivaGan']:
        opt.device = 'cpu'

    message = message_dict[opt.method]
    target_message_tensor = torch.Tensor(str2msg(message)).to(opt.device)

    bitwise_acc_sum = 0.0
    tp_sum = 0

    with tqdm(total=len(image_files), desc="decoding watermark from images...") as pbar:
        with open(output_file, 'a') as f:
            f.write("Path\ttarget message\tdecoded message\tbitacc\n")
            
            for idx in range(len(image_files)):
                image_tensor = ds.__getitem__(idx).unsqueeze(0).to(opt.device)

                bitacc, is_tp, decoded_msg = decode_single_image(
                    image_tensor=image_tensor,
                    target_message_tensor=target_message_tensor,
                    method=opt.method,
                    device=opt.device,
                    decoder=decoder,
                    tolerant_bits=tolerant_bits
                )

                bitwise_acc_sum += bitacc
                tp_sum += int(is_tp)

                f.write(f"{image_files[idx]}\t{message}\t{msg2str(decoded_msg)}\t{bitacc}\n")
                pbar.update(1)

            avg_bitacc = bitwise_acc_sum / len(image_files)
            TPR = tp_sum / len(image_files)

            f.write(f"bitwise_accuracy:{round(avg_bitacc, 4)}\n")
            f.write(f"TPR:{round(TPR, 4)}\n")

    return avg_bitacc, TPR
