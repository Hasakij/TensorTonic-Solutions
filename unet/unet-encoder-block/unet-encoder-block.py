import numpy as np

def unet_encoder_block(x: np.ndarray, kernel1: np.ndarray, bias1: np.ndarray,
                       kernel2: np.ndarray, bias2: np.ndarray) -> dict:
    """
    Returns pooled and skip as float64 arrays in a dictionary.
    """
    def conv_relu(inp, kernel, bias):
        
        # Extract input dimensions: batch, height, width, input channels
        B, H, W, C_in = inp.shape
        # Extract kernel dimensions: size (k x k) nad output channels
        k = kernel.shape[0] # kernel size
        C_out = kernel.shape[3] # number of filters

        # Calculate padding size for "same" padding strategy
        p = k // 2
        
        # Apply zero padding to  height and width axes only (axes 1 and 2)
        # to maintain spatial resolution after convolution
        inp_padded = np.pad(inp, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')

        # Empty output array
        res = np.zeros((B, H, W, C_out), dtype=np.float64)

        # Spatial sliding window iteration
        for i in range(H):
            for j in range(W):
                # Extract a k x k patch across all batches and input channels
                patch = inp_padded[:, i:i+k, j:j+k, :]
                
                # Vectorized convolution and add bias
                res[:, i, j, :] = np.tensordot(patch, kernel, axes=((1, 2, 3), (0, 1, 2))) + bias

        # Apply ReLU
        return np.maximum(0.0, res)

    # 2 convolutions
    # Extract initial features and changes channel depth
    conv1 = conv_relu(x, kernel1, bias1)
    # Refines features
    skip = conv_relu(conv1, kernel2, bias2)

    # Max pooling to reduce spatial resolution by half for the contracting path
    B, H_s, W_s, C_s = skip.shape
    # Reshape and take maximum to perform pooling
    pooled = skip.reshape(B, H_s // 2, 2, W_s // 2, 2, C_s).max(axis=(2, 4))
    
    return {
        "pooled": pooled.astype(np.float64),
        "skip": skip.astype(np.float64)
    }