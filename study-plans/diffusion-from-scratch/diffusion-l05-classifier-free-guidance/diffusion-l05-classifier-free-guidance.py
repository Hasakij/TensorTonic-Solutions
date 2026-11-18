import torch

def classifier_free_guidance(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, guidance_scale: float, guidance_rescale: float) -> torch.Tensor:
    """
    Returns the guided prediction with the shape and dtype of cond_pred.
    """
    guided = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    if guidance_rescale <= 0:
        return guided
    dims = tuple(range(1, guided.ndim))
    s_c = cond_pred.std(dim=dims, keepdim=True, correction=0)
    s_g = guided.std(dim=dims, keepdim=True, correction=0)
    r = guided * (s_c / s_g)
    
    result = guidance_rescale * r + (1 - guidance_rescale) * guided
    return result
        
