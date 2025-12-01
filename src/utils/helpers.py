import glob
import json
import math
import os
import random
from argparse import Namespace
from os.path import join

import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image


def ber_on_str(str1, str2):
    assert len(str1) == len(str2), "string1 and string2 must share same length"
    different_bits_count = sum(1 for bit1, bit2 in zip(str1, str2) if bit1 != bit2)
    return float(different_bits_count / len(str1))

def read_txt_file(fpath):
    with open(fpath, "r") as file:
        lines = file.readlines()
    return [line.strip().split(" ")[1] for line in lines]

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

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_networks(i_epoch, path, state_dict):
    torch.save(state_dict, join(path, f"epoch_{i_epoch}_state.pth"))
    print("saving state of training!")
    return


def cal_tolerant(total_bits, beta=0.05):
    from scipy.stats import binom
    for k in range(total_bits + 1):
        prob = 1 - binom.cdf(k - 1, total_bits, 0.5)  # 计算概率
        if prob <= beta:
            break
    return total_bits - k


def save_batch_as_individual_images(tensor_batch, surrogate_sim, output_dir, count):
    assert tensor_batch.dim() == 4, "Input tensor must be a 4D batch tensor"
    batch_size = tensor_batch.size(0)
    output_txt_file = os.path.join(output_dir, "sim.txt")
    with open(output_txt_file, "a") as f:
        for i in range(batch_size):
            tensor = tensor_batch[i]
            sim = surrogate_sim[i]
            image_path = os.path.join(output_dir, f"{count:05d}.jpg")
            vutils.save_image(tensor, image_path)
            f.write(f"{image_path} {sim.item()}\n")
            count += 1


def save_config_to_json(file_path, config):
    config_dict = config.__dict__
    with open(file_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def parse_yml(path):
    if not os.path.exists(path):
        return None

    f = open(path ,"r")
    config = yaml.safe_load(f)
    return config

def combine(args, config):
    dict_args = vars(args)  # Convert args to a dictionary
    for k, v in config.items():
        # Skip if the key is not in args
        if k not in dict_args:
            continue

        # If the value in args is None, keep the value from config
        if dict_args[k] is None:
            continue

        # Check for conflicts and print a message if there is one
        if dict_args[k] != v:
            print(f"[CONFLICT]: `{k}` command-line: `{dict_args[k]}` configuration file: `{v}`\nIgnore value in config file")

    # Update the config dictionary with the values from args (except those that are None)
    config.update({k: v for k, v in dict_args.items() if v is not None})

    # Create a Namespace object with the combined configuration
    comb_dict = config
    comb_args = Namespace(**comb_dict)

    return comb_args



def load_train_state(
    ckp_path, model, optimizer=None, lr_scheduler=None, ema=None,
):
    state_dict = torch.load(ckp_path, map_location="cpu")
    model_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict["model"].items()
    }
    print(model.load_state_dict(model_state_dict,strict=True))
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
        # lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    epoch = state_dict["epoch"]
    if ema is not None:
        ema.load_state_dict(state_dict['ema'])

    return model, optimizer, lr_scheduler, epoch, ema


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.val_batchsize,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=1, cols=1)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def get_lr_scheduler(Tconfig, optimizer, length):
    if Tconfig.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=Tconfig.lr_warmup_steps,
            num_training_steps=(length * Tconfig.num_epochs),
            num_cycles=0.25,
        )
    elif Tconfig.lr_scheduler == "patient":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=Tconfig.lr_gamma, patience=Tconfig.lr_patience
        )
    elif Tconfig.lr_scheduler == "linear":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=Tconfig.lr_step_size, gamma=Tconfig.lr_gamma
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0
        )

    return lr_scheduler



def get_image_count_in_directory(path):
    if not os.path.exists(path):
        return -1

    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_count = sum(
        len(glob.glob(os.path.join(path, pattern))) for pattern in image_patterns
    )
    return image_count
