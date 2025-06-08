import sys
import boto3
import botocore
import torch

from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf

from Watermarkschemes.models.dwtdct.arch import InvisibleWatermarker
from Watermarkschemes.models.hidden.arch import Hidden
from Watermarkschemes.models.hidden.noise_layers.noiser import Noiser
from Watermarkschemes.models.stablesignature.utils_model import load_model_from_config


def get_DwtDct(wm_text,wm_type):
    'the input of encoder and decoder should be from 0 to 1'
    Dwtdctwatermarker = InvisibleWatermarker(wm_text,'dwtDct',wm_type)
    encoder = Dwtdctwatermarker.encode
    decoder = Dwtdctwatermarker.decode
    return encoder,decoder

def get_DwtDctSvd(wm_text,wm_type):
    'the input of encoder and decoder should be from 0 to 1'
    Dwtdctwatermarker = InvisibleWatermarker(wm_text,'dwtDctSvd',wm_type)
    encoder = Dwtdctwatermarker.encode
    decoder = Dwtdctwatermarker.decode
    return encoder,decoder



def get_hiddenmodel(cfg,device):
    '''
    return encoder and unwrapped decoder(Hidden)
    and the input of encoder and decoder should be from -1 to 1
    '''
    noiser = Noiser(cfg.noise,device)
    hidden_net = Hidden(cfg,device,noiser)
    if cfg.checkpoint is not None:
        checkpoint = torch.load(cfg.checkpoint,map_location='cpu')

        hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'],strict=True)
        hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
        # hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
        # hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
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

    ldm_config = "/home_new/dongziping/DiffusionWM/src/Watermarkschemes/config/stable_signature/v2-inference.yaml"
    ldm_ckpt = "/home_new/dongziping/DiffusionWM/src/Watermarkschemes/checkpoints/stable_signature/v2-1_512-ema-pruned.ckpt"
    print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
    config = OmegaConf.load(f"{ldm_config}")
    ldm_ae = load_model_from_config(config, device, ldm_ckpt)
    ldm_aef = ldm_ae.first_stage_model
    ldm_aef.eval()
    if not nowm:
        state_dict = torch.load("/home_new/dongziping/DiffusionWM/src/Watermarkschemes/checkpoints/stable_signature/sd2_decoder.pth",map_location='cpu')
        unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)

    model = "/data/shared/AI_4/Huggingface/stable-diffusion-v2-1"
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
        msg_extractor = torch.jit.load("/data/shared/Dongziping/sharedcode/stable_signature/models/dec_48b_whit.torchscript.pt",map_location='cpu').to(device)
        msg = msg_extractor(x) # b c h w -> b k
        msg = (msg>0).float()
        return msg

    return encoder,decoder

def get_rivagan(wm_text):
    Rivaganwatermarker = InvisibleWatermarker(wm_text,'rivaGan')
    encoder = Rivaganwatermarker.encode
    decoder = Rivaganwatermarker.decode
    return encoder,decoder

def get_titan():
    def decode(img_path):
        bedrock_runtime = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
        try:
            with open(img_path, "rb") as image_file:
                input_image_iguana = image_file.read()
            
            response = bedrock_runtime.detect_generated_content(
                foundationModelId="amazon.titan-image-generator-v1",
                content={
                    "imageContent": {"bytes": input_image_iguana}
                }
            )

            is_generated = response.get("detectionResult")
            confidence = response.get("confidenceLevel")
            
            return is_generated, confidence

        except (botocore.exceptions.EndpointConnectionError, botocore.exceptions.BotoCoreError) as e:
            print(f"Network error occurred: {e}")
            return None, None
    return None,decode

def get_wam(device):
    # load model
    exp_dir = "Watermarkschemes/checkpoints/watermark_anything"
    json_path = os.path.join(exp_dir, "params.json")
    ckpt_path = os.path.join(exp_dir, 'checkpoint.pth')
    wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()
    return wam.embed,wam.detect

if __name__ == '__main__':
    _,decoder = get_titan()
    x = decoder("/data/shared/Dongziping/sharedcode/DiffusionWM/results/distored_results_ori/titan/mscoco/Blur/p0_00025.png")
    
    