from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import kornia

def psnr_ssim(batch_a, batch_b):
    '''
    Automatically handles image tensors in either [0,1] or [-1,1] range
    input: image tensors with values in either [0,1] or [-1,1]
    output: PSNR and SSIM values
    '''
    # Detect input range and normalize to [0,1] if needed
    min_val,max_val = batch_a.min(), batch_a.max()
    if min_val < 0 or max_val > 1: 
        if min_val >= -1 and max_val <= 1: 
            batch_a = (batch_a + 1) * 0.5
            batch_b = (batch_b + 1) * 0.5
        else:  
            batch_a = (batch_a - min_val) / (max_val - min_val)
            batch_b = (batch_b - min_val) / (max_val - min_val)

    # Add batch dimension if needed
    if batch_a.dim() == 3: 
        batch_a = batch_a.unsqueeze(0)
    if batch_b.dim() == 3: 
        batch_b = batch_b.unsqueeze(0)

    # Calculate PSNR
    psnr_wm2co = kornia.metrics.psnr(
        batch_a.detach().clamp(0, 1),
        batch_b.detach().clamp(0, 1),
        1
    )

    # Calculate SSIM
    ssim_wm2co = kornia.metrics.ssim(
        batch_a.detach().clamp(0, 1),
        batch_b.detach().clamp(0, 1),
        window_size=11,
    ).mean()
    
    return psnr_wm2co, ssim_wm2co