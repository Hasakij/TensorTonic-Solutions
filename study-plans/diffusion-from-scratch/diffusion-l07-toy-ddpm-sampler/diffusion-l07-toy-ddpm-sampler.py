import math
from typing import Optional

import torch
import torch.nn.functional as F
def toy_ddpm_sample(x_T: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor, params: dict, embed_dim: int, clip_value: Optional[float]) -> dict:
    """
    Returns a dict with "samples" of shape (N, D) and "pred_x0" of shape (T, N, D).
    """
    dtype = x_T.dtype
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    num_steps = betas.shape[0]
    half = embed_dim // 2
    i = torch.arange(half, dtype=dtype)
    f = torch.exp(-math.log(10000) * i / half)
    
    x = x_T.clone()
    preds_x0 = []
    for step, t in enumerate(range(num_steps -1, -1, -1)):
        args = t * f
        emb = torch.cat([torch.cos(args), torch.sin(args), torch.zeros(embed_dim % 2, dtype=dtype)])
        emb = emb.expand(x.shape[0], -1)
        
        eps_hat = F.silu(torch.cat([x, emb], dim=1) @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]
        ab_t = alpha_bars[t]
        ab_prev = alpha_bars[t-1] if t > 0 else torch.ones((), dtype=dtype)
        pred_x0 = (x - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        if clip_value is not None:
            pred_x0 = torch.clamp(pred_x0, -clip_value, clip_value)
        preds_x0.append(pred_x0)

        mean = (betas[t] * torch.sqrt(ab_prev) / (1 - ab_t)) * pred_x0 + ((1 - ab_prev) * torch.sqrt(alphas[t]) / (1 - ab_t)) * x

        if t > 0:
            beta_tilde = (betas[t] * (1 - ab_prev)) / (1 - ab_t)
            x = mean + torch.sqrt(beta_tilde) * noise[step]
        else:
            x = mean

    return {
        "samples": x,
        "pred_x0": torch.stack(preds_x0)
    }