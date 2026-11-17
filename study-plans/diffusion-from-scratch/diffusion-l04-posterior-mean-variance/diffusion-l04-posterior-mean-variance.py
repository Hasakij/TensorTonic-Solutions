import torch

def posterior_mean_variance(x0: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor, betas: torch.Tensor) -> dict:
    """
    Returns a dict with the posterior "mean" (same shape as x0) and the per-sample "variance" and "log_variance" of shape (B,).
    """
    alphas = 1 - betas
    ab = torch.cumprod(alphas, dim=0)
    ab_prev = torch.cat([torch.ones_like(ab[:1]), ab[:-1]])
    
    variance = ((1 - ab_prev) / (1 - ab)) * betas
    log_variance = torch.log(torch.cat([variance[1:2], variance[1:]]))

    coef_x0 = betas * (torch.sqrt(ab_prev)) / (1 - ab)
    coef_xt = (1 - ab_prev) * torch.sqrt(alphas) / (1 - ab)

    vshape = (-1, *([1] * (x0.dim() - 1)))
    mean = coef_x0[timesteps].view(vshape) * x0 + coef_xt[timesteps].view(vshape) * x_t

    return {
        "mean": mean,
        "variance": variance[timesteps],
        "log_variance": log_variance[timesteps]
    }
    
