import math
import torch
import torch.nn.functional as F

def res_block(h, e, params, prefix, num_groups):
    # GN -> SiLU -> conv1
    z = F.group_norm(h, num_groups, params[f"{prefix}_norm1_weight"], params[f"{prefix}_norm1_bias"], eps=1e-5)
    z = F.conv2d(F.silu(z), params[f"{prefix}_conv1_weight"], params[f"{prefix}_conv1_bias"], padding=1)
    t = F.linear(F.silu(e), params[f"{prefix}_time_weight"], params[f"{prefix}_time_bias"])
    z = z + t[:, :, None, None]

    # GN -> SiLU -> conv2
    z = F.group_norm(z, num_groups, params[f"{prefix}_norm2_weight"], params[f"{prefix}_norm2_bias"], eps=1e-5)
    z = F.conv2d(F.silu(z), params[f"{prefix}_conv2_weight"], params[f"{prefix}_conv2_bias"], padding=1)

    if f"{prefix}_skip_weight" in params:
        skip = F.conv2d(h, params[f"{prefix}_skip_weight"], params[f"{prefix}_skip_bias"])
    else:
        skip = h
        
    return skip + z
    
    
def tiny_unet_forward(x: torch.Tensor, timesteps: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the network output of shape (B, C_out, H, W) with the dtype of x.
    """

    B, C_in, H, W = x.shape
    E = params["time_linear1_weight"].shape[1]
    C = params["conv_in_weight"].shape[0]
    C_out = params["conv_out_weight"].shape[0]
    
    i = torch.arange(E // 2, dtype=x.dtype, device=x.device)
    freqs = torch.exp((-2 * i / E) * math.log(10000.0))
    angles = timesteps[:, None].to(x.dtype) * freqs[None, :]
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    lin1 = F.linear(emb, params["time_linear1_weight"], params["time_linear1_bias"])
    e = F.linear(F.silu(lin1), params["time_linear2_weight"], params["time_linear2_bias"])
    
    h = F.conv2d(x, params["conv_in_weight"], params["conv_in_bias"], padding=1)
    h1 = res_block(h, e, params, "down_res", num_groups)

    d = F.conv2d(h1, params["downsample_weight"], params["downsample_bias"], stride=2, padding=1)
    m = res_block(d, e, params, "mid_res", num_groups)
    m = F.interpolate(m, scale_factor=2, mode="nearest")
    u = F.conv2d(m, params["upsample_weight"], params["upsample_bias"], padding=1)

    cat = torch.cat([u, h1], dim=1)
    r = res_block(cat, e, params, "up_res", num_groups)
    r = F.group_norm(r, num_groups, params["out_norm_weight"], params["out_norm_bias"], eps=1e-5)
    out = F.conv2d(F.silu(r), params["conv_out_weight"], params["conv_out_bias"], padding=1)

    return out
    
