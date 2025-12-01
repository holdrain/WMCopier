import torch

from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf

from WMSuite.wm.algorithms.models.dwtdct.arch import InvisibleWatermarker
from WMSuite.wm.algorithms.models.hidden.arch import Hidden
from WMSuite.wm.algorithms.models.hidden.noise_layers.noiser import Noiser
from WMSuite.wm.algorithms.models.stablesignature.utils_model import load_model_from_config
from WMSuite.wm.algorithms.models.stegastamp.models import StegaStampEncoder, StegaStampDecoder
from WMSuite.wm.algorithms.models.vine.vine.src.stega_encoder_decoder import CustomConvNeXt
from WMSuite.wm.algorithms.models.vine.vine.src.vine_turbo import VINE_Turbo
from accelerate.utils import set_seed

def get_DwtDct(wm_text,wm_type):
    'the input of encoder and decoder should be from 0 to 1'
    Dwtdctwatermarker = InvisibleWatermarker(wm_text,'dwtDct',wm_type)
    encoder = Dwtdctwatermarker.encode
    decoder = Dwtdctwatermarker.decode
    return encoder,decoder

def get_vine(device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    encoder = VINE_Turbo.from_pretrained('Shilin-LU/VINE-B-Enc')
    encoder.to(device)

    decoder = CustomConvNeXt.from_pretrained('Shilin-LU/VINE-B-Dec')
    decoder.to(device)
    
    return encoder,decoder

def get_Stegastamp(device):
    encoder_path = "WMSuite/wm/algorithms/checkpoints/stegastamp/AFHQ_cat2dog_256x256_encoder.pth"
    decoder_path = "WMSuite/wm/algorithms/checkpoints/stegastamp/AFHQ_cat2dog_256x256_decoder.pth"
    IMAGE_RESOLUTION = 256  
    IMAGE_CHANNELS = 3

    encoder_state_dict = torch.load(encoder_path,map_location='cpu',weights_only=False)
    decoder_state_dict = torch.load(decoder_path,map_location='cpu',weights_only=False)
    FINGERPRINT_SIZE = encoder_state_dict["secret_dense.weight"].shape[-1]
    print(f'Stegastamp FINGERPRINT_SIZE: {FINGERPRINT_SIZE}')
    encoder = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    decoder = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder,decoder

def get_hiddenmodel(cfg,device):
    '''
    return encoder and unwrapped decoder(Hidden)
    and the input of encoder and decoder should be from -1 to 1
    '''
    noiser = Noiser(cfg.noise,device)
    hidden_net = Hidden(cfg,device,noiser)
    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint,map_location='cpu',weights_only=False)

        hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'],strict=True)
        hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
        print("loading pretrained model success!")
    else:
        print("init model parameters randomly!")
    encoder = hidden_net.encoder_decoder.encoder
    decoder = hidden_net.encoder_decoder.decoder
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder,decoder


def get_stablesignature(device,nowm=False):
    
    ldm_config = "WMSuite/wm/algorithms/config/stable_signature/v2-inference.yaml"
    ldm_ckpt = "WMSuite/wm/algorithms/checkpoints/stable_signature/v2-1_512-ema-pruned.ckpt"
    print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
    config = OmegaConf.load(f"{ldm_config}")
    ldm_ae = load_model_from_config(config, device, ldm_ckpt)
    ldm_aef = ldm_ae.first_stage_model
    ldm_aef.eval()
    if not nowm:
        state_dict = torch.load("WMSuite/wm/algorithms/checkpoints/stable_signature/sd2_decoder.pth",map_location='cpu',weights_only=False)
        unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)

    # huggingface stable diffusion path
    model = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model).to(device)
    pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))

    @torch.no_grad()
    def encoder(prompts,negative_prompt,size):
        '''
            prompts should be a list of strings
            return a list of pil image
        '''
        pil_images = pipe(prompts,negative_prompt=negative_prompt,height=size,width=size).images
        return pil_images

    @torch.no_grad()
    def decoder(x):
        '''
            x should be watermarked images tensor
            return watermark message
        '''
        msg_extractor = torch.jit.load("WMSuite/wm/algorithms/checkpoints/stable_signature/dec_48b_whit.torchscript.pt",map_location='cpu').to(device)
        msg = msg_extractor(x) # b c h w -> b k
        msg = (msg>0).float()
        return msg

    return encoder,decoder

def get_rivagan(wm_text):
    Rivaganwatermarker = InvisibleWatermarker(wm_text,'rivaGan')
    encoder = Rivaganwatermarker.encode
    decoder = Rivaganwatermarker.decode
    return encoder,decoder
    
    