import torch
from torch import Tensor
from typing import List, Optional, Union

def mask_image_from_size(x: Tensor, xs: Optional[Union[Tensor, List[int]]] = None) -> Tensor:
    """Create a mask for a batch of images based on their sizes.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        xs: Optional list or tensor of valid widths for each image
        
    Returns:
        Binary mask tensor of shape (N, 1, H, W)
    """
    if xs is None:
        return torch.ones((x.size(0), 1, x.size(2), x.size(3)), 
                        dtype=torch.float32, 
                        device=x.device)
    
    # Convert list to tensor if needed
    if isinstance(xs, list):
        xs = torch.tensor(xs, device=x.device)
    
    # Create width position tensor
    w_pos = torch.arange(x.size(3), device=x.device).view(1, -1)
    w_pos = w_pos.expand(x.size(0), -1)
    
    # Create mask based on valid widths
    mask = w_pos < xs.view(-1, 1)
    return mask.unsqueeze(1).float()

class MaskImageFromSize(torch.nn.Module):
    """Module wrapper for mask_image_from_size function."""

    def forward(self, x: Tensor, xs: Optional[Union[Tensor, List[int]]] = None) -> Tensor:
        return mask_image_from_size(x, xs)
