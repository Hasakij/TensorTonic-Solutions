import torch

def min_snr_weighted_loss(per_sample_mse: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, gamma: float, prediction_type: str) -> dict:
    """
    Returns a dict with the per-sample "snr" and "weights" of shape (B,) and the weighted batch "loss" as a Python float.
    """
    alpha_bar_t = alpha_bars[timesteps]
    snr = alpha_bar_t / (1 - alpha_bar_t)
    capped = torch.clamp(snr, max=gamma)
    if prediction_type == "epsilon":
        weights = capped / snr
    elif prediction_type == "x0":
        weights = capped
    elif prediction_type == "v":
        weights = capped / (snr + 1)
    else:
        raise ValueError(f"Wrong prediction type: {prediction_type}")
    loss = torch.mean(weights * per_sample_mse).item()
    return {
        "snr": snr,
        "weights": weights,
        "loss": loss
    }
