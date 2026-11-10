import torch

def forward_diffusion_sample(x0: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Returns the noisy samples x_t with the same shape and dtype as x0.
    """
    alpha_bar_t = alpha_bars[timesteps].to(x0.dtype)
    alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x0.dim() -1)))
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
