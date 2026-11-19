import torch

def denoising_score_matching_loss(x0: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss "loss" (Python float) and "per_sample_loss" (tensor of shape (B,)).
    """
    sigma = sigmas.unsqueeze(-1)
    x_noise = x0 + sigma * noise

    score = (x_noise @ weight + bias) / sigma
    target_score = -noise / sigma
    sqr_error = (score - target_score).pow(2).sum(dim=1)
    per_sample_loss = 0.5 * sigmas.pow(2) * sqr_error

    return {
        "loss": per_sample_loss.mean().item(),
        "per_sample_loss": per_sample_loss
    }
