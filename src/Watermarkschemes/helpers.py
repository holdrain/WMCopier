import torchvision.transforms as transforms
import torch

def cal_tolerant(total_bits, beta=0.05):
    from scipy.stats import binom
    for k in range(total_bits + 1):
        prob = 1 - binom.cdf(k - 1, total_bits, 0.5)
        if prob <= beta:
            break
    return total_bits - k

# message legnth
message_length_dict = {
    "clean":0,
    "hidden":30,
    "dwtdct":32,
    "stable_signature":48,
    "rivaGan":32,
    'titan':0,
    "rivaGanmul":32,
    "treering":0,  
}

# transforms
transforms_dict_encode = {
    'clean':transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "hidden":transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "dwtdctsvd":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "dwtdct":transforms.Compose([
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
    "rivaGanmul":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "wam":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'titan':transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        ]),
    'treering':transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ]),
}

transforms_dict_decode = {
    "hidden":transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "dwtdct":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "dwtdctsvd":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "stable_signature":transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    "rivaGan":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "rivaGanmul":transforms.Compose([
        transforms.ToTensor(),
    ]),
    "titan":transforms.Compose([
        transforms.ToTensor(),
        ]),
    "wam":transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}



tensor_norm_dict = {
    "hidden": lambda x: x,
    "dwtdct": lambda x: x * 0.5 + 0.5,
    "dwtdctsvd": lambda x: x * 0.5 + 0.5,
    "stable_signature": lambda x: ((x * 0.5 + 0.5) - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(x.device)) / 
                                  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(x.device),
    "rivaGan": lambda x: x * 0.5 + 0.5,
    "rivaGanmul": lambda x: x * 0.5 + 0.5,
    "titan": lambda x: x * 0.5 + 0.5,
    "treering": lambda x: x * 0.5 + 0.5,
}

transforms_dict_inversion= {
    "hidden":transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "dwtdct":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "stable_signature":transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "rivaGan":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "rivaGanmul":transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "titan":transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'treering':transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

target_message_dict = {                            
    "dwtdct":'00001100001001011110001101101110', # 00001100001001011110001101101110(finetune),00001100001001011110001101101110
    "hidden":'000011000010010111100011011011', # 000011000010010111100011011011(finetune) 000010011110100010101100111011
    "rivaGan":'00001001111010001010110011101100', # 00001100001001011110001101101110(finetune),00001001111010001010110011101100
    "stable_signature":'111010110101000001010111010011010100010000100111',
}