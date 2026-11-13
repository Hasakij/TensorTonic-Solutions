import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x: torch.Tensor, params: dict, num_groups: int, num_heads: int) -> torch.Tensor:
    """
    Returns x plus the attention output, shape (B, C, H, W) with the dtype of x.
    """
    B, C, H, W = x.shape
    N = H * W
    d = C // num_heads

    # GN + tokenization
    h = F.group_norm(x, num_groups, params["norm_weight"], params["norm_bias"], eps=1e-5)
    Z = h.flatten(2).transpose(1,2) # (B, C, H, W) -> (B, H*W, C)

    # Q, K, V
    Q = F.linear(Z, params["q_weight"], params["q_bias"])
    K = F.linear(Z, params["k_weight"], params["k_bias"])
    V = F.linear(Z, params["v_weight"], params["v_bias"])

    # split heads
    # (B, N, C) -> (B, N, n, d) - > (B, n, N, d)
    Q = Q.view(B, N, num_heads, d).transpose(1, 2)
    K = K.view(B, N, num_heads, d).transpose(1, 2)
    V = V.view(B, N, num_heads, d).transpose(1, 2)

    # attention per head
    # (B, n, N, d) -> (B, n, d, N)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    O = attn @ V # (B, n, N, d)

    # concatenation
    # (B, n, N, d) -> (B, N, n, d) -> (B, N, C)
    O = O.transpose(1, 2).reshape(B, N, C)

    # out proj
    # (B, N, C) -> (B, C, N) -> (B, C, H, W)
    O = F.linear(O, params["out_weight"], params["out_bias"])
    O = O.transpose(1, 2).reshape(B, C, H, W)
    return x + O
    
    
    
    