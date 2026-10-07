import numpy as np

def unet_bottleneck(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                    kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 bottleneck features in NHWC layout.
    """
    def conv_relu(inp, kernel, bias):
        B, H, W, C_in = inp.shape
        k = kernel.shape[0]
        C_out = kernel.shape[3]

        p = k // 2
        inp_padded = np.pad(inp, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        res = np.zeros((B, H, W, C_out), dtype=np.float64)

        for i in range(H):
            for j in range(W):
                patch = inp_padded[:, i:i+k, j:j+k, :]
                res[:, i, j, :] = np.tensordot(patch, kernel, axes=((1, 2, 3), (0, 1, 2))) + bias
        return np.maximum(0.0, res)

    # Conv + Bias + ReLU
    h1 = conv_relu(x, kernel1, bias1)

    # Conv + Bias + ReLU
    y = conv_relu(h1, kernel2, bias2)
    return y.astype(np.float64)
    