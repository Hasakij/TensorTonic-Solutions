import numpy as np

def cosine_noise_schedule(num_steps: int, s: float, max_beta: float) -> dict:
    """
    Returns a dict with float64 arrays "betas" and "alpha_bars", each of shape (num_steps,).
    """
    u = np.arange(num_steps + 1, dtype=np.float64)
    t = np.arange(num_steps, dtype=np.float64)
    arg = ((u / num_steps + s) / (1 + s)) * np.pi / 2
    f = np.cos(arg) ** 2
    raw_betas = 1 - f[1:] / f[:-1]
    betas = np.minimum(raw_betas, max_beta)
    alpha_bars = np.cumprod(1 - betas)
    return {
        "betas": betas,
        "alpha_bars": alpha_bars
    }
