import torch
import torch.nn.functional as F

def time_residual_block(x: torch.Tensor, time_embedding: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the block output of shape (B, C_out, H, W) with the dtype of x.
    """
    t = F.linear(F.silu(time_embedding), params['time_weight'], params['time_bias'])
    
    h = F.group_norm(x, num_groups, params['norm1_weight'], params['norm1_bias'], eps=1e-5)
    h = F.conv2d(F.silu(h), params['conv1_weight'], params['conv1_bias'], padding=1)
    h = h + t[:, :, None, None]

    h = F.group_norm(h, num_groups, params['norm2_weight'], params['norm2_bias'], eps=1e-5)
    out_conv2 = F.conv2d(F.silu(h), params['conv2_weight'], params['conv2_bias'], padding=1)

    if "skip_weight" in params:
        skip = F.conv2d(x, params['skip_weight'], params['skip_bias'])
    else:
        skip = x
    return skip + out_conv2
    
    
