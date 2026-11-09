import numpy as np

def diagonal_gaussian_kl(mean_p: np.ndarray, log_var_p: np.ndarray, mean_q: np.ndarray, log_var_q: np.ndarray) -> np.ndarray:
    """
    Returns KL(p || q) summed over the last axis, as a float64 array with the leading shape of the inputs.
    """
    var_p = np.exp(log_var_p)
    var_q = np.exp(log_var_q)
    per_dim = 0.5 * (log_var_q - log_var_p + (var_p + (mean_p - mean_q) ** 2) / var_q - 1.0)
    return np.sum(per_dim, axis=-1)
