import numpy as np

def gaussian_mixture_score(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """
    Returns the score grad_x log p(x) of the isotropic Gaussian mixture at every row of x, as a float64 array of shape (N, D).
    """
    N, D = x.shape
    K = means.shape[0]

    diff = x[:, None, :] - means[None, :, :]     # (N, K ,D)
    sq = np.sum(diff ** 2, axis=-1)
    log_comp = (-D / 2) * np.log(2 * np.pi * variances) - sq / (2 * variances)

    log_joint = np.log(weights) + log_comp       # (N, K)
    m = np.max(log_joint, axis=1, keepdims=True) # (N, 1)
    log_p = m + np.log(np.sum(np.exp(log_joint - m), axis=1, keepdims=True)) # (N, 1)

    resp = np.exp(log_joint - log_p)

    score = (np.sum(resp[:, :, None] * (means[None, :, :] - x[:, None, :]) / variances[None, :, None], axis=1))
    return score
    
