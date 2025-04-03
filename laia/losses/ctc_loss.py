import itertools
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_packed_sequence

import laia.common.logging as log
from laia.losses.loss import Loss

_logger = log.get_logger(__name__)


def transform_batch(batch):
    """Transform a batch for CTC loss computation.
    
    Args:
        batch: Input batch, either a PackedSequence or a Tensor
        
    Returns:
        tuple: (x, xs) where x is the input tensor and xs is the sequence lengths
    """
    if isinstance(batch, torch.nn.utils.rnn.PackedSequence):
        x, xs = pad_packed_sequence(batch)
    elif isinstance(batch, torch.Tensor):
        x, xs = batch, [batch.size(0)] * batch.size(1)
    else:
        raise NotImplementedError(f"Not implemented for type {type(batch)}")
    return x, xs


def get_valids_and_errors(
    xs: List[int], y: List[List[int]]
) -> Tuple[List[int], List[int]]:
    """Check for sequences which are too short to produce the given
    target, according to CTC model. These could produce infinite
    losses or potential buffer overflows in CTC.
    """
    assert len(xs) == len(y)

    def count_minimum_frames(y: List[int]) -> int:
        return len(y) + sum(y[i] == y[i - 1] for i in range(1, len(y)))

    check = [xs[i] >= count_minimum_frames(y[i]) for i in range(len(y))]
    return (
        # Indices of OK samples
        [i for i, valid in enumerate(check) if valid],
        # Indices of the samples with errors regarding their ref length
        [i for i, valid in enumerate(check) if not valid],
    )


class CTCLoss(Loss):
    """
    Attributes:
      reduction (string): Specifies the type of reduction to
        perform on the minibatch costs: 'none' | 'mean' | 'sum'.
        With 'none': no reduction is done and a tensor with the cost of each
        example in the minibatch is returned,
        'mean': the mean of the per-example losses is returned,
        'sum': the sum of all per-example losses is returned.
        Default: 'mean'.
      average_frames (bool): Specifies whether the loss of each
        sample should be divided by its number of frames. Default: ``False''.
      blank (int): Index of the blank label. Default: 0.
    """

    def __init__(
        self, reduction: str = "mean", average_frames: bool = False, blank: int = 0
    ):
        super().__init__()
        assert reduction in (
            "none",
            "mean",
            "sum",
        ), f"Reduction {reduction} is not supported"
        self.reduction = reduction
        self.average_frames = average_frames
        assert blank >= 0, "Blank index must be >= 0"
        self.blank = blank

    def forward(
        self, x: torch.Tensor, y: List[List[int]], **kwargs: Dict
    ) -> Optional[torch.Tensor]:
        x, xs = transform_batch(x)
        assert len(y) == x.size(1), "Batch size does not match"

        valid_indices, err_indices = get_valids_and_errors(xs, y)
        if err_indices:
            if kwargs.get("batch_ids", None) is not None:
                assert isinstance(kwargs["batch_ids"], (list, tuple))
                err_indices = [kwargs["batch_ids"][i] for i in err_indices]
            _logger.warning(
                "The following samples in the batch were ignored "
                "for the loss computation: {}",
                err_indices,
            )
        if not valid_indices:
            _logger.warning("All samples in the batch were ignored!")
            return

        # prepare tensors of the correct type
        x = torch.nn.functional.log_softmax(x, dim=-1)
        
        # Get the device of the input tensor
        device = x.device
        
        # Convert xs to tensor if it's not already
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.int, device=device)
        else:
            xs = xs.to(dtype=torch.int, device=device)
            
        # Convert ys to tensor
        ys = torch.tensor([len(y_n) for y_n in y], dtype=torch.int, device=device)
        
        # Convert y to tensor
        y = torch.tensor(list(itertools.chain.from_iterable(y)), dtype=torch.int, device=device)

        # keep valid indices
        valid_indices = torch.tensor(valid_indices, device=device)
        x = x.index_select(1, valid_indices)  # Select valid batch items
        xs = xs.index_select(0, valid_indices)
        ys = ys.index_select(0, valid_indices)

        # Compute CTC loss
        losses = torch.nn.functional.ctc_loss(
            log_probs=x,
            targets=y,
            input_lengths=xs,
            target_lengths=ys,
            blank=self.blank,
            reduction="none",
            zero_infinity=True,
        )

        if self.average_frames:
            losses = losses / xs.to(losses)

        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()
