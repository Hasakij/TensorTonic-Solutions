import numpy as np

def unet_forward(x: np.ndarray, weights: dict) -> np.ndarray:
    """
    Returns float64 per-pixel segmentation logits in NHWC layout.
    """
    def conv_relu(inp, kernel, bias, apply_relu=True):
        B, H, W, C_in = inp.shape
        k = kernel.shape[0]
        C_out = kernel.shape[3]
        p = k // 2

        padded = np.pad(inp, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
        res = np.zeros((B, H, W, C_out), dtype=np.float64)

        for i in range(H):
            for j in range(W):
                patch = padded[:, i:i+k, j:j+k, :]
                res[:, i, j, :] = np.tensordot(patch, kernel, axes=((1, 2, 3), (0, 1, 2 ))) + bias
        return np.maximum(0.0, res) if apply_relu else res

    # Encoder
    c1 = conv_relu(x, weights['enc_kernel1'], weights['enc_bias1'])
    skip = conv_relu(c1, weights['enc_kernel2'], weights['enc_bias2'])

    # Max Pooling 2x2
    B, H, W, C = skip.shape
    pooled = skip.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))
    

    # Bottleneck
    b1 = conv_relu(pooled, weights['bridge_kernel1'], weights['bridge_bias1'])
    bridge = conv_relu(b1, weights['bridge_kernel2'], weights['bridge_bias2'])
    
    # Decoder: Upsample + Project
    up = np.repeat(np.repeat(bridge, 2, axis=1), 2, axis=2)
    U = np.maximum(0.0, np.dot(up, weights['W_up']) + weights['b_up'])

    # Skip connection: Crop + Concat
    _, h_u, w_u, _ = U.shape
    _, h_s, w_s, _ = skip.shape
    h_off, w_off = (h_s - h_u) // 2, (w_s - w_u) // 2

    skip_cropped = skip[:, h_off:h_off+h_u, w_off:w_off+w_u, :]
    combined = np.concatenate([skip_cropped, U], axis=-1)

    # Decoder convolutions
    d1 = conv_relu(combined, weights['dec_kernel1'], weights['dec_bias1'])
    d2 = conv_relu(d1, weights['dec_kernel2'], weights['dec_bias2'])

    # Output layer
    logits = d2 @ weights['W_out'] + weights['b_out']
    return logits.astype(np.float64)
    
        
        