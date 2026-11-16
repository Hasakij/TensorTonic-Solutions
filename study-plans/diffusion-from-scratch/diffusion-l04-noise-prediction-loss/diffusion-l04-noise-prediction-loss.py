import torch

def noise_prediction_loss(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss as a Python float under "loss" and the per-sample losses of shape (B,) under "per_sample_loss".
    """
    B = x0.shape[0]
    D = x0[0].numel()
    T = alpha_bars.shape[0]

    x0_flat = x0.reshape(B, D)
    noise_flat = noise.reshape(B, D)
    
    ab = alpha_bars[timesteps].to(x0.dtype)[:, None] # (B, 1)
    x_t = torch.sqrt(ab) * x0_flat + torch.sqrt(1 - ab) * noise_flat  # (B, D)
    
    t_frac = timesteps.to(x0.dtype)[:, None] / T # (B, 1)
    eps_hat = x_t @ weight + t_frac * time_weight + bias # (B, D)
    
    per_sample_loss = ((eps_hat - noise_flat) **2).mean(dim=1) # (B, )
    loss = per_sample_loss.mean()
    return {
        "loss": loss.item(),
        "per_sample_loss": per_sample_loss
    }
    
