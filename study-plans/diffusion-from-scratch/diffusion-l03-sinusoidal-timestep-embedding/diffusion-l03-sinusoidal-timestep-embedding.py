import torch

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
    """
    Returns the float32 sinusoidal embedding of shape (B, dim).
    """
    half = dim // 2
    B = timesteps.shape[0]
    
    i = torch.arange(half, dtype=torch.float32)
    freqs = torch.exp(-torch.log(torch.tensor(max_period,  dtype=torch.float32)) * i / half)

    angles = timesteps[:, None].float() * freqs[None, :]

    parts = [torch.cos(angles), torch.sin(angles)]

    if dim % 2 == 1:
        emb = torch.cat([torch.cos(angles), torch.sin(angles), torch.zeros(B, 1, dtype=torch.float32)], dim=1)
    else:
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    return emb