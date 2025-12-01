import os
import torch
import random
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import transforms

from WMSuite.wm.core.dataset import CustomImageFolder
from WMSuite.wm.core.helpers import (
    generate_random_fingerprints,
    msg2str,
    new_dir,
    str2msg,
    set_seeds,
)
from WMSuite.wm.core.yml import load_config
from WMSuite.wm.core.helpers import message_dict, transforms_dict_encode, load_prompts, image_resolution_dict
from WMSuite.wm.core.model_choice import (
    get_DwtDct,
    get_hiddenmodel,
    get_rivagan,
    get_stablesignature,
    get_Stegastamp,
    get_vine,
)


def encode_single_image(
    image_tensor,
    encoder,
    method,
    device,
    message,
):
    """
    Encode watermark into a single image tensor.

    Args:
        image_tensor (Tensor): CxHxW on CPU
        encoder: encoder model
        method (str): algorithm name
        device (str)
        message (str)

    Returns:
        clean_tensor (Tensor), encoded_tensor (Tensor)
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Methods requiring message
    if method in ["hidden", "stegastamp", "vine"]:
        msg_tensor = torch.tensor(str2msg(message)).unsqueeze(0).to(device)

    # Dispatch encoding
    if method == "hidden":
        encoded = encoder(image_tensor, msg_tensor)
    elif method == "stegastamp":
        encoded = encoder(msg_tensor, image_tensor)
    elif method == "vine":
        encoded = encoder(image_tensor, secret=msg_tensor)
    elif method == "rivagan":
        encoded = encoder(image_tensor)
    elif method == "dwtdct":
        encoded = encoder(image_tensor)
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    return (
        image_tensor.squeeze(0).cpu(),
        encoded.squeeze(0).detach().cpu(),
    )


def save_clean_and_wm(clean_tensor, encoded_tensor, clean_path, wm_path):
    vutils.save_image(encoded_tensor, wm_path, normalize=True)
    vutils.save_image(clean_tensor, clean_path, normalize=True)



def run_stable_signature(opt):
    prompts_list, negative_prompts = load_prompts("wm/algorithms/config/stable_signature/prompt.yml")
    message = message_dict[opt.method]

    print(prompts_list, negative_prompts)

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    encoder_nowm, _ = get_stablesignature(opt.device, nowm=True)
    encoder, _ = get_stablesignature(opt.device, nowm=False)

    count = 0
    while count < opt.num:
        prompts = random.choices(prompts_list, k=opt.batch_size)

        with tqdm(total=opt.num, desc=f"generating wm images by {opt.method}") as pbar:
            set_seeds(count)
            clean_imgs = encoder_nowm(
                prompts,
                negative_prompt=negative_prompts * opt.batch_size,
                size=image_resolution_dict[opt.method],
            )
            set_seeds(count)
            encoded_imgs = encoder(
                prompts,
                negative_prompt=negative_prompts * opt.batch_size,
                size=image_resolution_dict[opt.method],
            )

            for clpil, encpil in zip(clean_imgs, encoded_imgs):
                clean_path = os.path.join(cl_dir, f"{count:04d}{opt.filetype}")
                wm_path = os.path.join(wm_dir, f"{count:04d}{opt.filetype}")
                clpil.save(clean_path)
                encpil.save(wm_path)

                count += 1
                pbar.update(1)



def run_hidden(opt, message_type="default"):
    cfg = load_config("wm/algorithms/config/hidden/hidden.yaml")
    encoder, _ = get_hiddenmodel(cfg.train, opt.device)

    ds = CustomImageFolder(opt.dataset, transform=transforms_dict_encode[opt.method], num=opt.num)
    dl = DataLoader(ds, batch_size=opt.batch_size, num_workers=0)

    message = (
        message_dict[opt.method]
        if message_type == "default"
        else msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
    )

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    with tqdm(total=len(ds), desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for batch in dl:
            batch = batch.to(opt.device)

            msg_tensor = torch.tensor(str2msg(message)).repeat(batch.shape[0], 1).to(opt.device)
            encoded_batch = encoder(batch, msg_tensor)

            for i in range(batch.shape[0]):
                save_clean_and_wm(
                    batch[i].cpu(),
                    encoded_batch[i].cpu(),
                    os.path.join(cl_dir, f"{count:04d}{opt.filetype}"),
                    os.path.join(wm_dir, f"{count:04d}{opt.filetype}"),
                )
                count += 1
                pbar.update(1)


def run_rivagan(opt, message_type="default"):
    ds = CustomImageFolder(opt.dataset, transform=transforms_dict_encode[opt.method], num=opt.num)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    message = (
        message_dict[opt.method]
        if message_type == "default"
        else msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
    )

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    with tqdm(total=len(ds), desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for img in dl:
            encoder, _ = get_rivagan(wm_text=message)
            clean, wm = encode_single_image(
                image_tensor=img.squeeze(0),
                encoder=encoder,
                method="rivagan",
                device='cpu',
                message=message,
            )
            save_clean_and_wm(
                clean, wm,
                os.path.join(cl_dir, f"{count:04d}{opt.filetype}"),
                os.path.join(wm_dir, f"{count:04d}{opt.filetype}")
            )
            count += 1
            pbar.update(1)


def run_dwtdct(opt, message_type="default"):
    ds = CustomImageFolder(opt.dataset, transform=transforms_dict_encode[opt.method], num=opt.num)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    message = (
        message_dict[opt.method]
        if message_type == "default"
        else msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
    )

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    with tqdm(total=len(ds), desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for img in dl:
            encoder, _ = get_DwtDct(wm_text=message, wm_type="bits")
            clean, wm = encode_single_image(
                image_tensor=img.squeeze(0),
                encoder=encoder,
                method="dwtdct",
                device='cpu',
                message=message,
            )
            save_clean_and_wm(
                clean, wm,
                os.path.join(cl_dir, f"{count:04d}{opt.filetype}"),
                os.path.join(wm_dir, f"{count:04d}{opt.filetype}")
            )
            count += 1
            pbar.update(1)


def run_stegastamp(opt, message_type="default"):
    encoder, decoder = get_Stegastamp(opt.device)

    ds = CustomImageFolder(opt.dataset, transform=transforms_dict_encode[opt.method], num=opt.num)
    dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=False)

    message = (
        message_dict[opt.method]
        if message_type == "default"
        else msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
    )

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    with tqdm(total=len(ds), desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for batch in dl:
            batch = batch.to(opt.device)
            msg_tensor = torch.tensor(str2msg(message)).repeat(batch.shape[0], 1).to(opt.device)

            encoded = encoder(msg_tensor, batch)

            for i in range(batch.shape[0]):
                save_clean_and_wm(
                    batch[i].cpu(),
                    encoded[i].cpu(),
                    os.path.join(cl_dir, f"{count:04d}{opt.filetype}"),
                    os.path.join(wm_dir, f"{count:04d}{opt.filetype}"),
                )
                count += 1
                pbar.update(1)


def run_vine(opt, message_type="default"):
    encoder, _ = get_vine(opt.device)

    ds = CustomImageFolder(opt.dataset, transform=transforms_dict_encode[opt.method], num=opt.num)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    message = (
        message_dict[opt.method]
        if message_type == "default"
        else msg2str(generate_random_fingerprints(len(message_dict[opt.method])))
    )

    cl_dir = new_dir(os.path.join(opt.output_dir, opt.method, "clean"))
    wm_dir = new_dir(os.path.join(opt.output_dir, opt.method, message))

    with tqdm(total=len(ds), desc=f"generating wm images by {opt.method}") as pbar:
        count = 0
        for img in dl:
            clean, wm = encode_single_image(
                image_tensor=img.squeeze(0),
                encoder=encoder,
                method="vine",
                device=opt.device,
                message=message,
            )
            save_clean_and_wm(
                clean, wm,
                os.path.join(cl_dir, f"{count:04d}{opt.filetype}"),
                os.path.join(wm_dir, f"{count:04d}{opt.filetype}")
            )
            count += 1
            pbar.update(1)

