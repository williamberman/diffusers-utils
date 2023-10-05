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


_with_tqdm = False


def set_with_tqdm(it):
    global _with_tqdm

    _with_tqdm = it


@torch.no_grad()
def rk_ode_solver_diffusion_loop(eps_theta, timesteps, sigmas, x_T, rk_steps_weights):
    x_t = x_T

    iter_over = range(len(timesteps) - 1, -1, -1)

    if _with_tqdm:
        from tqdm import tqdm

        iter_over = tqdm(iter_over)

    for i in iter_over:
        t = timesteps[i].unsqueeze(0)
        sigma = sigmas[t]

        if i == 0:
            eps_hat = eps_theta(x_t=x_t, t=t, sigma=sigma)
            x_0_hat = x_t - sigma * eps_hat
        else:
            dt = sigmas[timesteps[i - 1]] - sigma

            dx_by_dt = torch.zeros_like(x_t)
            dx_by_dt_cur = torch.zeros_like(x_t)

            for rk_step, rk_weight in rk_steps_weights:
                dt_ = dt * rk_step
                t_ = t + dt_
                x_t_ = x_t + dx_by_dt_cur * dt_
                eps_hat = eps_theta(x_t=x_t_, t=t_, sigma=sigma)
                # TODO - note which specific ode this is the solution to and
                # how input scaling does/doesn't effect the solution
                # dx_by_dt_cur = (x_t_ - sigma * eps_hat) / sigma
                dx_by_dt_cur = eps_hat
                dx_by_dt += dx_by_dt_cur * rk_weight

            x_t_minus_1 = x_t + dx_by_dt * dt

            x_t = x_t_minus_1

    return x_0_hat


euler_ode_solver_diffusion_loop = lambda *args, **kwargs: rk_ode_solver_diffusion_loop(*args, **kwargs, rk_steps_weights=[[0, 1]])

heun_ode_solver_diffusion_loop = lambda *args, **kwargs: rk_ode_solver_diffusion_loop(*args, **kwargs, rk_steps_weights=[[0, 0.5], [1, 0.5]])

rk4_ode_solver_diffusion_loop = lambda *args, **kwargs: rk_ode_solver_diffusion_loop(*args, **kwargs, rk_steps_weights=[[0, 1 / 6], [1 / 2, 1 / 3], [1 / 2, 1 / 3], [1, 1 / 6]])
