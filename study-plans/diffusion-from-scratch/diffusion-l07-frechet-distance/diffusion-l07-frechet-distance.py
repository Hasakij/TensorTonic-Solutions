import numpy as np

def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Returns the squared Frechet distance between Gaussian fits of the two feature sets as a Python float.
    """
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)

    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    mean_term = np.sum((mean_a - mean_b) ** 2)
    
    sigma_a = np.atleast_2d(np.cov(a, rowvar=False, ddof=1))
    sigma_b = np.atleast_2d(np.cov(b, rowvar=False, ddof=1))

    values, vectors = np.linalg.eigh(sigma_a)
    values = np.clip(values, 0, None)
    sigma_a_sqrt = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    M = sigma_a_sqrt @ sigma_b @ sigma_a_sqrt

    lambda_M = np.linalg.eigvalsh(M)
    lambda_M = np.clip(lambda_M, 0, None)
    cross = np.sum(np.sqrt(lambda_M))

    trace_term = np.trace(sigma_a) + np.trace(sigma_b)

    d2 = mean_term + trace_term - 2 * cross
    return d2
    
    
