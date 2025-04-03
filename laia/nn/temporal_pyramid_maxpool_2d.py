import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union

from laia.data import PaddedTensor


def temporal_maxpool_2d(x: Tensor, output_size: Union[int, Tuple[int, int], List[int]]) -> Tensor:
    """Applies temporal max pooling over an input signal.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        output_size: Desired output size (H_out, W_out)
        
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, list):
        output_size = tuple(output_size)
    return F.adaptive_max_pool2d(x, output_size)


class TemporalPyramidMaxPool2d(torch.nn.Module):
    """Temporal pyramid max pooling layer."""

    def __init__(self, levels: List[int], height: int = 1):
        """Initialize the pooling layer.
        
        Args:
            levels: List of temporal levels for pooling
            height: Height of the output feature maps
        """
        super().__init__()
        self.levels = levels
        self.height = height

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Output tensor containing pooled features at different temporal levels
        """
        features = []
        for level in self.levels:
            pooled = temporal_maxpool_2d(x, (self.height, level))
            features.append(pooled)
        return torch.cat(features, dim=3)
