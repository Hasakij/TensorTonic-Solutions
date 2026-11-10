import numpy as np

def rescale_zero_terminal_snr(betas: np.ndarray) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alpha_bars" and "snr" of the rescaled schedule, each of shape (T,).
    """
    alpha_bars = np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))
    root = np.sqrt(alpha_bars)
    
    root_first = root[0]
    root_last = root[-1]
    
    root = (root - root_last) * root_first / (root_first - root_last)
    new_alpha_bars = root ** 2
    
    new_alphas = np.empty_like(new_alpha_bars)
    new_alphas[0] = new_alpha_bars[0]
    new_alphas[1:] = new_alpha_bars[1:] / new_alpha_bars[:-1]
    
    new_betas = 1.0 - new_alphas
    
    snr = new_alpha_bars / (1.0 - new_alpha_bars)
    return {
        "betas": new_betas,
        "alpha_bars": new_alpha_bars,
        "snr": snr
    }