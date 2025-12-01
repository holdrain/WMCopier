import torch

# inversion to get intermediate latents
@torch.no_grad()
def pred_latents(pipe, x_0, num_inference_steps, end_step, device):
    '''
    x_0 is with dimension of (b,c,h,w)
    '''
    intermediate_latents = []
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)
    latent_t = x_0.clone()
    
    for i in range(1, end_step+1):
        if i > end_step: continue
        t = timesteps[i]
        with torch.no_grad():
            noise_pred = pipe.unet(latent_t, t).sample
            current_t = max(0, t.item() - (1000//num_inference_steps))
            next_t = t 
            alpha_t = pipe.scheduler.alphas_cumprod[current_t]
            alpha_t_next = pipe.scheduler.alphas_cumprod[next_t.item()]
            latent_t = (latent_t - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        intermediate_latents.append(latent_t)
    return intermediate_latents

# denosing with refinement
@torch.no_grad()
def sample(x_0, pipe, start_step=0, start_latents=None,
        num_inference_steps=100, device='cuda',
        return_interm=False, refine=True, all_steps=False, eta=1e-4, beta=1, L=20):
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = start_latents[-1].clone()
    latents_list = []
    for i in range(start_step, num_inference_steps):

        t = pipe.scheduler.timesteps[i]
        latents = pipe.scheduler.scale_model_input(latents, t)
        prev_t = max(1, t.item() - (1000//num_inference_steps))
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

        noise_pred = pipe.unet(latents, t).sample
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

        # refinement
        if refine:
            if all_steps:
                latents = refine_latent(latents,L,eta,beta,pipe,t,x_0)
            else:
                if i == num_inference_steps - 1:
                    latents = refine_latent(latents,L,eta,beta,pipe,t,x_0)
        if return_interm:
            latents_list.append(torch.clamp(latents,-1,1).detach().cpu())

    image_tensor = torch.clamp(latents,-1,1)
    return image_tensor,latents_list

# refine the latent with score correction and content gradient
def refine_latent(latent,L,eta,beta,pipe,t,x_0):
    latent = latent.clone()
    for _ in range(L):
        rn = torch.randn_like(latent)
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        noised_latent = pipe.scheduler.add_noise(latent, rn, t)
        ne_noise_pred = pipe.unet(noised_latent, t).sample
        score = -ne_noise_pred / (1 - alpha_t).sqrt()
        content_grad = latent - x_0
        correction = beta * content_grad - score
        latent = latent - eta * correction
    return latent


