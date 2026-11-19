import numpy as np

def annealed_langevin_dynamics(x_init: np.ndarray, sigmas: np.ndarray, steps_per_sigma: int, step_size: float, noise: np.ndarray, data_mean: np.ndarray, data_var: float) -> np.ndarray:
    """
    Returns the final samples after annealed Langevin dynamics, a float64 array of shape (N, D).
    """
    x = np.array(x_init, dtype=np.float64)
    L = len(sigmas)
    sigma_last_sq = sigmas[-1] ** 2

    for i in range(L):
        sigma = sigmas[i]
        alpha = step_size * sigma ** 2 / sigma_last_sq
        for j in range(steps_per_sigma):
            z = noise[i * steps_per_sigma + j]
            score = -(x - data_mean) / (data_var + sigma**2)
            x = x + (alpha / 2) * score + np.sqrt(alpha) * z
    return x
        