import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import laia.common.logging as log

_logger = log.get_logger(__name__)


class ProgressBarGPUStats(pl.callbacks.DeviceStatsMonitor):
    """Monitor GPU stats during training and add them to the progress bar."""

    def __init__(self, memory_utilization: bool = True, gpu_utilization: bool = True):
        super().__init__()
        self._gpu_ids = None
        self.memory_utilization = memory_utilization
        self.gpu_utilization = gpu_utilization

    def setup(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        super().setup(trainer, *args, **kwargs)
        if trainer.strategy.root_device.type == "cuda":
            self._gpu_ids = ",".join(map(str, trainer.device_ids))

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", *args, **kwargs) -> None:
        if not self._gpu_ids:
            return

        stats = {}
        if self.memory_utilization:
            memory_usage = self.get_memory_usage()
            if memory_usage:
                stats["GPU mem"] = f"{memory_usage:.1f}GB"

        if self.gpu_utilization:
            gpu_usage = self.get_gpu_utilization()
            if gpu_usage:
                stats["GPU util"] = f"{gpu_usage:.1f}%"

        if stats:
            trainer.progress_bar_metrics["gpu_stats"] = stats

    def get_memory_usage(self) -> Optional[float]:
        """Get GPU memory usage in GB."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            return memory_allocated
        except:
            return None

    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            return None
