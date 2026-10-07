import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray,
                       W_up: np.ndarray, b_up: np.ndarray,
                       kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 decoder features in NHWC layout.
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

    # Upsample: repeat each spatial value twice
    # H, W -> 2H, 2W
    x_up = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)

    # Channel projection
    U = np.maximum(0.0, np.dot(x_up, W_up) + b_up)

    # Center-crop skip tensor to match U dimensions
    _, h_u, w_u, _ = U.shape
    _, h_s, w_s, _ = skip.shape

    h_offset = (h_s - h_u) // 2
    w_offset = (w_s - w_u) // 2

    # Slicing  the center region
    s_cropped = skip[:, h_offset:h_offset+h_u, w_offset:w_offset+w_u, :]

    # Concatenate skip channels and projected decoder channels
    combined = np.concatenate([s_cropped, U], axis=-1)

    # Apply two same-padded convolutions with ReLU
    conv1 = conv_relu(combined, kernel1, bias1)
    y = conv_relu(conv1, kernel2, bias2)
    
    return y.astype(np.float64)