import torch

from utils import sdxl_text_conditioning, sdxl_tokenize_one, sdxl_tokenize_two

default_num_train_timesteps = 1000


@torch.no_grad()
def make_sigmas(beta_start=0.00085, beta_end=0.012, num_train_timesteps=default_num_train_timesteps, device=None):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32, device=device) ** 2

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # TODO - would be nice to use a direct expression for this
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

    return sigmas


@torch.no_grad()
def sdxl_diffusion_loop(
    prompts, images, unet, text_encoder_one, text_encoder_two, controlnet=None, adapter=None, sigmas=None, timesteps=None, x_T=None, micro_conditioning=None, guidance_scale=5.0, generator=None
):
    encoder_hidden_states, pooled_encoder_hidden_states = sdxl_text_conditioning(
        text_encoder_one,
        text_encoder_two,
        sdxl_tokenize_one(prompts).to(text_encoder_one.device),
        sdxl_tokenize_two(prompts).to(text_encoder_two.device),
    )

    if x_T is None:
        x_T = torch.randn((1, 4, 1024 // 8, 1024 // 8), dtype=torch.float32, device=unet.device, generator=generator)
        x_T = x_T * ((sigmas.max() ** 2 + 1) ** 0.5)

    if sigmas is None:
        sigmas = make_sigmas()

    if timesteps is None:
        timesteps = torch.linspace(0, sigmas.numel(), 50, dtype=torch.long, device=unet.device)

    if micro_conditioning is None:
        micro_conditioning = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.long, device=unet.device)

    if adapter is not None:
        down_block_additional_residuals = adapter(images)
    else:
        down_block_additional_residuals = None

    if controlnet is not None:
        controlnet_cond = images
    else:
        controlnet_cond = None

    eps_theta = lambda x_t, t, sigma: sdxl_eps_theta(
        x_t=x_t,
        t=t,
        sigma=sigma,
        unet=unet,
        encoder_hidden_states=encoder_hidden_states,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        guidance_scale=guidance_scale,
        controlnet=controlnet,
        controlnet_cond=controlnet_cond,
        down_block_additional_residuals=down_block_additional_residuals,
    )

    x_0 = ode_solver_diffusion_loop(eps_theta=eps_theta, timesteps=timesteps, sigmas=sigmas, x_T=x_T)

    return x_0


@torch.no_grad()
def sdxl_eps_theta(
    x_t,
    t,
    sigma,
    unet,
    encoder_hidden_states,
    pooled_encoder_hidden_states,
    micro_conditioning,
    guidance_scale,
    controlnet=None,
    controlnet_cond=None,
    down_block_additional_residuals=None,
):
    # TODO - how does this not effect the ode we are solving
    scaled_x_t = x_t / ((sigma**2 + 1) ** 0.5)

    if guidance_scale > 1.0:
        scaled_x_t = torch.concat([scaled_x_t, scaled_x_t])

    if controlnet is not None:
        controlnet_out = controlnet(
            x_t=scaled_x_t,
            t=t,
            encoder_hidden_states=encoder_hidden_states,
            micro_conditioning=micro_conditioning,
            pooled_encoder_hidden_states=pooled_encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )

        down_block_additional_residuals = controlnet_out["down_block_res_samples"]
        mid_block_additional_residual = controlnet_out["mid_block_res_sample"]
        add_to_down_block_inputs = controlnet_out.get("add_to_down_block_inputs", None)
        add_to_output = controlnet_out.get("add_to_output", None)
    else:
        mid_block_additional_residual = None
        add_to_down_block_inputs = None
        add_to_output = None

    eps_hat = unet(
        x_t=scaled_x_t,
        t=t,
        encoder_hidden_states=encoder_hidden_states,
        micro_conditioning=micro_conditioning,
        pooled_encoder_hidden_states=pooled_encoder_hidden_states,
        down_block_additional_residuals=down_block_additional_residuals,
        mid_block_additional_residual=mid_block_additional_residual,
        add_to_down_block_inputs=add_to_down_block_inputs,
        add_to_output=add_to_output,
    )

    if guidance_scale > 1.0:
        eps_hat_uncond, eps_hat = eps_hat.chunk(2)

        eps_hat = eps_hat_uncond + guidance_scale * (eps_hat - eps_hat_uncond)

    return eps_hat


@torch.no_grad()
def ode_solver_diffusion_loop(eps_theta, timesteps, sigmas, x_T):
    x_t = x_T

    for i in range(len(timesteps) - 1, -1, -1):
        t = timesteps[i]

        sigma = sigmas[i]

        eps_hat = eps_theta(x_t=x_t, t=t, sigma=sigma)

        if i == 0:
            x_0_hat = x_t - sigma * eps_hat
        else:
            # first order euler
            dt = sigmas[i - 1] - sigma

            # TODO - note which specific ode this is the solution to
            dx_by_dt = (x_t - sigma * eps_hat) / sigma

            x_t_minus_1 = x_t + dx_by_dt * dt

            x_t = x_t_minus_1

            # TODO add optional second order

    return x_0_hat
