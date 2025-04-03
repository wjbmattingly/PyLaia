import torch
from typing import List, Optional, Tuple, Union

from laia.data import PaddedTensor
from laia.nn.adaptive_pool_2d import adaptive_maxpool_2d


class PyramidMaxPool2d(torch.nn.Module):
    """Pyramid max pooling layer that supports PaddedTensor inputs."""

    def __init__(self, levels: List[int], vertical: bool = False):
        """Initialize the pooling layer.
        
        Args:
            levels: List of pooling levels
            vertical: If True, pool vertically, otherwise horizontally
        """
        super().__init__()
        self.levels = levels
        self.vertical = vertical

    def forward(self, x: Union[torch.Tensor, PaddedTensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor or PaddedTensor of shape (N, C, H, W)
            
        Returns:
            Tensor containing concatenated pooled features
        """
        if isinstance(x, PaddedTensor):
            x, xs = x.data, x.sizes
        else:
            xs = None

        n, c = x.size()[:2]
        features = []
        
        for level in self.levels:
            output_sizes = (level, 1) if self.vertical else (1, level)
            pooled = adaptive_maxpool_2d(
                batch_input=x,
                output_sizes=output_sizes,
                batch_sizes=xs
            )
            features.append(pooled.view(n, c * level))

        return torch.cat(features, dim=1)
