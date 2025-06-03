import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import os
from data.dataset import CustomImageFolder
from Watermarkschemes.helpers import transforms_dict_decode,target_message_dict
from decode import load_models
import argparse
from tqdm.auto import tqdm
from utils.helpers import str2msg


def image_distortion(img1, img2, seed, args):
    if args.jpeg_ratio is not None:
        img1.save(f"tmp_a{args.jpeg_ratio}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_a{args.jpeg_ratio}.jpg")
        img2.save(f"tmp_b{args.jpeg_ratio}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_b{args.jpeg_ratio}.jpg")
 
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Distortion')
    parser.add_argument('--device', type=str, default='cuda:4', help='Device to run on')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to process')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--method', type=str, default='rivaGan', help='Method type')
    parser.add_argument('--num_level', type=int, default=5000, help='num for training')
    parser.add_argument('--dataset', type=str, default='diffusiondb', help='Dataset name')
    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)


    args = parser.parse_args()

    authentic_dir = ""
    forged_dir = ""

    if args.method != 'stable_signature':
        a_data_path = os.path.join(authentic_dir,args.method,args.dataset)
    else:
        a_data_path = os.path.join(authentic_dir,args.method)
    b_data_path = os.path.join(forged_dir,args.method,str(args.num_level),args.dataset)

    a_data = CustomImageFolder(a_data_path,transform=None,random_sample=True,num=args.num_images)
    b_data = CustomImageFolder(b_data_path,transform=None,random_sample=True,num=args.num_images)

    # load decoder
    _, decoder = load_models(args.method, args.device)
    target_message = target_message_dict[args.method]
    if args.method not in ['dwtdct','rivaGan']:
        target_message_tensor = torch.Tensor(str2msg(target_message)).to(args.device)
    else:
        target_message_tensor = torch.Tensor(str2msg(target_message))

    a_bit_acc_sum,b_bit_acc_sum = 0,0

    with tqdm(total=args.num_images,desc=f"distorting...") as pbar:
        for idx in range(args.num_images):
            seed = idx + args.seed
            a_img = a_data.__getitem__(idx)
            b_img = b_data.__getitem__(idx)
            a_img_d, b_img_d = image_distortion(a_img,b_img,seed,args)
            a_img_d_tensor = transforms_dict_decode[args.method](a_img_d)
            b_img_d_tensor = transforms_dict_decode[args.method](b_img_d)

            a_img_d_tensor = a_img_d_tensor.to(target_message_tensor.device).unsqueeze(0)
            b_img_d_tensor = b_img_d_tensor.to(target_message_tensor.device).unsqueeze(0)

            a_message = decoder(a_img_d_tensor)
            a_message = a_message.round().clip(0, 1).long()
            a_difference = (a_message != target_message_tensor).float()
            a_bitwise_accuracy = (1.0 - a_difference.mean()).item()
            a_bit_acc_sum += a_bitwise_accuracy
            
            b_message = decoder(b_img_d_tensor)
            b_message = b_message.round().clip(0, 1).long()
            b_difference = (b_message != target_message_tensor).float()
            b_bitwise_accuracy = (1.0 - b_difference.mean()).item()
            b_bit_acc_sum += b_bitwise_accuracy

            pbar.update(1)
    
    a_bit_acc_avg = a_bit_acc_sum / args.num_images
    b_bit_acc_avg = b_bit_acc_sum / args.num_images
    print(f"Average bitwise accuracy for authentic images: {a_bit_acc_avg}")
    print(f"Average bitwise accuracy for forged images: {b_bit_acc_avg}")
    print(f"Average bitwise accuracy difference: {a_bit_acc_avg - b_bit_acc_avg}")


    result_dir = ""
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"bit_acc_{args.method}.txt")

    with open(result_file, "a") as f:
        f.write("-"*20)
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Num Level: {args.num_level}\n")
        f.write(f"JPEG Quality: {args.jpeg_ratio}\n")
        f.write(f"Gaussian Blur Radius: {args.gaussian_blur_r}\n")
        f.write(f"Gaussian Noise STD: {args.gaussian_std}\n")
        f.write(f"Number of Images: {args.num_images}\n")
        f.write(f"Authentic Bit Acc: {a_bit_acc_avg:.4f}\n")
        f.write(f"Forged Bit Acc: {b_bit_acc_avg:.4f}\n")
        f.write(f"Bit Acc Difference: {a_bit_acc_avg - b_bit_acc_avg:.4f}\n")



