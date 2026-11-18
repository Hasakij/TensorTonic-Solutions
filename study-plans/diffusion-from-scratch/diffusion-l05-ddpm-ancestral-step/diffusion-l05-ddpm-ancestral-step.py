import torch

def ddpm_step(x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor, variance_type: str) -> torch.Tensor:
    """
    Returns the sample x_{t-1} after one DDPM reverse step, with the shape and dtype of x_t.
    """
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    alpha_t = alphas[t]
    ab_t = alpha_bars[t]
    ab_prev = alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
    
    beta_t = betas[t]

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - ab_t)) * predicted_noise)
    if variance_type == "fixed_large":
        var = beta_t
    elif variance_type == "fixed_small":
        var = beta_t * (1.0 - ab_prev) / (1 - ab_t)
    else:
        raise ValueError(f"Wrong variance type {variance_type}")
    if t == 0:
        return mean
    else:
        return mean + torch.sqrt(var) * noise
