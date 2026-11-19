import torch

def karras_euler_sample(initial_noise: torch.Tensor, num_steps: int, sigma_min: float, sigma_max: float, rho: float, data_mean: float, data_std: float, method: str) -> dict:
    """
    Returns a dict with the noise levels "sigmas" of shape (num_steps + 1,) and the final "samples" with the shape of initial_noise.
    """
    ramp = torch.linspace(0, 1, num_steps,dtype = initial_noise.dtype, device = initial_noise.device)
    sigmas = (sigma_max**(1/rho) + ramp * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
    x = sigma_max * initial_noise
    
    def D(x, sigma):
        return (data_std**2 * x + sigma**2 * data_mean) / (data_std**2 + sigma**2)

    for i in range(num_steps):
        s_i, s_next = sigmas[i], sigmas[i+1]
        d = (x - D(x, s_i)) / s_i
        x_euler = x + (s_next - s_i) * d
        if method == "heun" and s_next > 0:
            d_huen = (x_euler - D(x_euler, s_next)) / s_next
            x = x + (s_next - s_i) * (d + d_huen) / 2
        else:
            x = x_euler
    return {
        "sigmas": sigmas,
        "samples": x
    }
            