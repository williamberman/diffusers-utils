import torch

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
