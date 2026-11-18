import torch

def ddim_step(x_t: torch.Tensor, predicted_noise: torch.Tensor, t: int, t_prev: int, alpha_bars: torch.Tensor, eta: float, noise: torch.Tensor, clip_sample: bool) -> dict:
    """
    Returns a dict with "x_prev" and "pred_x0", each with the shape and dtype of x_t.
    """
    a_t = alpha_bars[t]
    a_prev = alpha_bars[t_prev] if t_prev >= 0 else 1

    pred_x0 = (x_t - torch.sqrt(1 - a_t) * predicted_noise) / torch.sqrt(a_t)
    eps = predicted_noise
    if clip_sample:
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        eps = (x_t - torch.sqrt(a_t) * pred_x0) / (torch.sqrt(1 - a_t))
    sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(max(1 - a_t / a_prev, 0))
    x_prev = torch.sqrt(torch.tensor(a_prev)) * pred_x0 + torch.sqrt(max(1 - a_prev - sigma ** 2, 0)) * eps + sigma * noise
    return {
        "x_prev": x_prev,
        "pred_x0": pred_x0
    }
