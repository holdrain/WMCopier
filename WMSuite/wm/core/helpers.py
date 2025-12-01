import torchvision.transforms as transforms
from pathlib import Path
import yaml
import textwrap
import torch
import os
import random
import numpy as np

def cal_tolerant(total_bits, beta=0.05):
    from scipy.stats import binom
    for k in range(total_bits + 1):
        prob = 1 - binom.cdf(k - 1, total_bits, 0.5)  # 计算概率
        if prob <= beta:
            break
    return total_bits - k

def load_prompts(yml_path: str):
    yml_path = Path(yml_path)
    data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))

    raw_prompts = data.get("prompt", "") or ""
    raw_prompts = textwrap.dedent(raw_prompts)
    prompt_list = [line.strip() for line in raw_prompts.splitlines() if line.strip()]

    neg = data.get("negetive_prompt")
    if neg is None:
        neg = data.get("negative_prompt", "")
    negative_prompt = textwrap.dedent(neg or "").strip()

    return prompt_list, [negative_prompt]


def generate_random_fingerprints(fingerprint_size, batch_size=1):
    '''
    return a tensor with dimension of (b,fs) and whose elements are randomly generated 0 or 1
    '''
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z

def msg2str(message):
    string = "".join(str(int(i)) for i in message.view(-1))
    return string

def str2msg(str):
    return torch.tensor([True if el == "1" else False for el in str], dtype=torch.float)

def new_dir(path):
    """
    create a new folder if it not exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# message legnth
message_length_dict = {
    "vine":100,
    "hidden":30,
    "stegastamp":100,
    "dwtdct":32,
    "stable_signature":48,
    "rivaGan":32,
}

# transforms
transforms_dict_encode = {
    "hidden":transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "dwtdct":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "stegastamp":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "stable_signature":transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    "rivaGan":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    'vine':transforms.Compose([
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
}

transforms_dict_decode = {
    "hidden":transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "stegastamp":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "dwtdct":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "stable_signature":transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    "rivaGan":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "vine":transforms.Compose([
        transforms.ToTensor(),
    ]),
}


message_dict = {
    "dwtdct":'00001100001001011110001101101110',
    "stegastamp":'0'*100,
    "vine": '0'*100,
    "hidden":'000010011110100010101100111011',
    "rivaGan":'00001100001001011110001101101110',
    "stable_signature":'111010110101000001010111010011010100010000100111',
}

image_resolution_dict = {
    "hidden":128,
    "dwtdct":256,
    "stegastamp":256,
    "stable_signature":512,
    "rivaGan":256,
    "vine":256,
}
