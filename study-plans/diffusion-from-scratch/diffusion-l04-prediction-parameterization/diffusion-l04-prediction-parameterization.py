import torch

def convert_prediction(model_output: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, prediction_type: str) -> dict:
    """
    Returns a dict with the "x0", "epsilon" and "v" predictions, each with the shape and dtype of x_t.
    """
    if prediction_type not in ("epsilon", "x0", "v"):
        raise ValueError(f"Wrong prediction type: {prediction_type}")
    alpha_bar_t = alpha_bars[timesteps].view(-1, *[1] * (x_t.dim() - 1))
    a = torch.sqrt(alpha_bar_t)
    b = torch.sqrt(1 - alpha_bar_t)

    if prediction_type == "epsilon":
        epsilon = model_output.clone()
        x0 = (x_t - b * epsilon) / a
        v = a * epsilon - b * x0
    elif prediction_type == "x0":
        x0 = model_output.clone()
        epsilon = (x_t - a * x0) / b
        v = a * epsilon - b * x0
    else:
        v = model_output.clone()
        x0 = a * x_t - b * v
        epsilon = b * x_t + a * v
    return {
        "x0": x0,
        "epsilon": epsilon,
        "v": v
    }
