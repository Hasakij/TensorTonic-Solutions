import torch
import torch.nn.functional as F
def dit_block_adaln_zero(x: torch.Tensor, conditioning: torch.Tensor, params: dict, patch_size: int) -> dict:
    """
    Returns a dict with "tokens" and "output", each a tensor of shape (B, N, D) with the dtype of x.
    """
    B, C, H, W = x.shape
    D = params["embed_weight"].shape[1]
    p = patch_size

    W_e = params["embed_weight"].T.reshape(D, C, p, p)
    z = F.conv2d(x, W_e, params["embed_bias"], stride=p) # (B, D, H/p, W/p)
    z = z.flatten(2).transpose(1, 2)                     # (B, N, D)
    z = z + params["pos_embed"]
    
    mod = F.silu(conditioning) @ params["ada_weight"] + params["ada_bias"] # (B, 3D)
    s_shift, s_scale, g = mod.chunk(3, dim=-1) # (B, D)
    s_shift = s_shift[:, None, :]              # (B, 1, D)
    s_scale = s_scale[:, None, :]
    g = g[:, None, :]

    h = F.layer_norm(z, (D,), eps=1e-6) # (B, N, D)
    h = h * (1 + s_scale) + s_shift
    h = h @ params["mlp_weight"] + params["mlp_bias"]
    out = z + g * h
    
    return {
        "tokens": z,
        "output": out
    }
