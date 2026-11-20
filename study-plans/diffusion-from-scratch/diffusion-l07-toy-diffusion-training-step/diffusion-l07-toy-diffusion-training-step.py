import math
import torch
import torch.nn.functional as F

def toy_diffusion_training_step(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, params: dict, embed_dim: int, learning_rate: float, max_grad_norm: float) -> dict:
    """
    Returns a dict with "loss" (float), "grad_norm" (float, before clipping) and "params" (dict of updated tensors).
    """
    ab = alpha_bars[timesteps].unsqueeze(1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    half = embed_dim // 2
    i = torch.arange(half, dtype=x0.dtype, device=x0.device)
    f = torch.exp(-math.log(10000) * i / half)
    angles = timesteps[:, None].to(x0.dtype) * f[None, :]
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if embed_dim % 2 != 0:
        zeros = torch.zeros(emb.shape[0], 1, dtype=x0.dtype, device=x0.device)
        emb = torch.cat([emb, zeros], dim=-1)

    weights = {key: value.detach().clone().requires_grad_(True) for key, value in params.items()}
    hidden = F.silu(torch.cat([x_t, emb], dim=1) @ weights["W1"] + weights["b1"])
    eps_hat = hidden @ weights["W2"] + weights["b2"]
    loss = ((eps_hat - noise) ** 2).mean()
    loss.backward()

    keys = ["W1", "b1", "W2", "b2"]
    grad_norm = torch.sqrt(sum((weights[key].grad ** 2).sum() for key in keys)).item()
    clip = min(1.0, max_grad_norm / (grad_norm + 1e-6))

    with torch.no_grad():
        new_params = {key: weights[key] - learning_rate * clip * weights[key].grad for key in keys}

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm,
        "params": new_params
    }

    