from functools import partial
import torch
from diffusers import UNet2DModel

seed = 2025

images_resolution = {
    "dwtdct": 256,
    "hidden": 128,
    "rivaGan": 256,
    "stable_signature": 512,
    "titan": 512,
}

target_message_dict = {
    "dwtdct":'00001100001001011110001101101110',
    "hidden":'000010011110100010101100111011',
    "rivaGan":'00001001111010001010110011101100',
    "stable_signature":'111010110101000001010111010011010100010000100111',
}

# model architecture
Unet = partial(
    UNet2DModel,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(
        128,
        128,
        256,
        256,
        512,
        512,
    ),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# beta_schedule
def cosine_beta_schedule(timesteps=1000, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps=1000):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps=1000):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps=1000):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

beta_schedule_choices = {'sigmoid':sigmoid_beta_schedule,'linear':linear_beta_schedule,'cosine':cosine_beta_schedule}
model_choices = {"unet": Unet}
