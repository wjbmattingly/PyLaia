import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union

from laia.data import PaddedTensor


def adaptive_avgpool_2d(batch_input: Tensor, output_sizes: Union[int, Tuple[int, int], List[int]], batch_sizes: Optional[Tensor] = None) -> Tensor:
    """Applies 2D adaptive average pooling over an input signal.
    
    Args:
        batch_input: Input tensor of shape (N, C, H, W)
        output_sizes: Desired output size (H_out, W_out)
        batch_sizes: Optional tensor of valid sizes for each input
        
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    if isinstance(output_sizes, int):
        output_sizes = (output_sizes, output_sizes)
    elif isinstance(output_sizes, list):
        output_sizes = tuple(output_sizes)

    if batch_sizes is None:
        return F.adaptive_avg_pool2d(batch_input, output_sizes)
    
    outputs = []
    for n in range(batch_input.size(0)):
        h, w = map(int, batch_sizes[n])
        x = batch_input[n, :, :h, :w].contiguous()
        outputs.append(F.adaptive_avg_pool2d(x, output_sizes))
    return torch.stack(outputs)


def adaptive_maxpool_2d(batch_input: Tensor, output_sizes: Union[int, Tuple[int, int], List[int]], batch_sizes: Optional[Tensor] = None) -> Tensor:
    """Applies 2D adaptive max pooling over an input signal.
    
    Args:
        batch_input: Input tensor of shape (N, C, H, W)
        output_sizes: Desired output size (H_out, W_out)
        batch_sizes: Optional tensor of valid sizes for each input
        
    Returns:
        Output tensor of shape (N, C, H_out, W_out)
    """
    if isinstance(output_sizes, int):
        output_sizes = (output_sizes, output_sizes)
    elif isinstance(output_sizes, list):
        output_sizes = tuple(output_sizes)

    if batch_sizes is None:
        return F.adaptive_max_pool2d(batch_input, output_sizes)
    
    outputs = []
    for n in range(batch_input.size(0)):
        h, w = map(int, batch_sizes[n])
        x = batch_input[n, :, :h, :w].contiguous()
        outputs.append(F.adaptive_max_pool2d(x, output_sizes))
    return torch.stack(outputs)


class AdaptivePool2d(torch.nn.Module):
    """Base class for adaptive pooling that supports PaddedTensor inputs."""

    def __init__(self, output_sizes: Union[int, Tuple[int, int], List[int]], func: callable):
        super().__init__()
        self._output_sizes = output_sizes
        self._func = func
        self._fixed_size = isinstance(output_sizes, int) or (
            output_sizes[0] is not None and output_sizes[1] is not None
        )

    @property
    def output_sizes(self):
        return self._output_sizes

    def forward(self, x: Union[Tensor, PaddedTensor]) -> Union[Tensor, PaddedTensor]:
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        y = self._func(batch_input=x, output_sizes=self.output_sizes, batch_sizes=xs)
        if xs is None or self._fixed_size:
            return y
        ys = xs.clone()
        dim = int(self.output_sizes[0] is None)
        ys[:, dim] = self.output_sizes[dim]
        return PaddedTensor.build(y, ys)


class AdaptiveAvgPool2d(AdaptivePool2d):
    """Adaptive average pooling layer that supports PaddedTensor inputs."""
    
    def __init__(self, output_size: Union[int, Tuple[int, int], List[int]]):
        super().__init__(output_sizes=output_size, func=adaptive_avgpool_2d)


class AdaptiveMaxPool2d(AdaptivePool2d):
    """Adaptive max pooling layer that supports PaddedTensor inputs."""
    
    def __init__(self, output_size: Union[int, Tuple[int, int], List[int]]):
        super().__init__(output_sizes=output_size, func=adaptive_maxpool_2d)
