import torch

def flow_matching_loss(x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with "x_t", "target", "per_sample_loss" (tensors) and "loss" (Python float).
    """
    t_col = t[:,None]
    x_t = (1 - t_col) * x_data + t_col * x_noise
    target = x_noise - x_data
    v = x_t @ weight + t_col * time_weight + bias
    per_sample_loss = ((v - target) ** 2).mean(dim=1)
    loss = per_sample_loss.mean().item()
    return {
        "x_t": x_t,
        "target": target,
        "per_sample_loss": per_sample_loss,
        "loss": loss
    }
