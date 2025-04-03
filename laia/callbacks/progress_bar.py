import sys
from collections import defaultdict
from logging import INFO
from typing import Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar as PLProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from tqdm.auto import tqdm

import laia.common.logging as log
from laia.callbacks.meters import Timer
import math


def _format_num(n: Union[int, float]) -> str:
    """Format number for display."""
    if math.isinf(n):
        return "∞" if n > 0 else "-∞"
    return f"{n:.3g}"


class ProgressBar(PLProgressBar):
    """Custom progress bar for PyLaia training."""

    def __init__(
        self,
        refresh_rate: int = 1,
        ncols: Optional[int] = 120,
        dynamic_ncols: bool = True,
        process_position: int = 0,
    ):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)
        self.ncols = ncols
        self.dynamic_ncols = dynamic_ncols
        self.running_sanity = None
        self.level = INFO
        self.format = defaultdict(
            self.format_factory,
            {
                "loss": "{:.4f}",
                "cer": "{:.1%}",
                "wer": "{:.1%}",
            },
        )
        # lightning merges tr + va into a "main bar".
        # we want to keep it separate so we have to time it ourselves
        self.tr_timer = Timer()
        self.va_timer = Timer()

    @staticmethod
    def format_factory():
        return "{}"

    def get_metrics(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> Dict[str, Union[int, float, str]]:
        items = super().get_metrics(trainer, model)
        return {k: _format_num(v) if isinstance(v, (int, float)) else v for k, v in items.items()}

    def init_sanity_tqdm(self) -> Tqdm:
        bar = super().init_sanity_tqdm()
        bar.set_description("Validation sanity check")
        return bar

    def init_train_tqdm(self) -> tqdm:
        return tqdm(
            desc="TR",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
            smoothing=0,
        )

    def init_validation_tqdm(self) -> tqdm:
        return tqdm(
            desc="VA",
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=False,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
        )

    def init_test_tqdm(self) -> tqdm:
        return tqdm(
            desc="Decoding",
            position=(2 * self.process_position + 1),
            disable=self.is_disabled,
            leave=True,
            ncols=self.ncols,
            dynamic_ncols=self.dynamic_ncols,
            file=sys.stderr,
        )

    def on_epoch_start(self, trainer, *args, **kwargs):
        # skip parent
        super(PLProgressBar, self).on_epoch_start(trainer, *args, **kwargs)
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(self.total_train_batches)
        self.main_progress_bar.set_description_str(f"TR - E{trainer.current_epoch}")

    def on_train_epoch_start(self, *args, **kwargs):
        super().on_train_epoch_start(*args, **kwargs)
        self.tr_timer.reset()

    def on_validation_epoch_start(self, trainer, *args, **kwargs):
        super().on_validation_start(trainer, *args, **kwargs)
        if trainer.running_sanity_check:
            self.val_progress_bar.set_description_str("VA sanity check")
        else:
            self.tr_timer.stop()
            self.va_timer.reset()
            self.val_progress_bar.set_description_str(f"VA - E{trainer.current_epoch}")

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # skip parent to avoid two postfix calls
        super(PLProgressBar, self).on_train_batch_end(
            trainer, *args, **kwargs
        )
        if self._should_update(self.train_batch_idx, self.total_train_batches):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(
                refresh=True,
                running_loss=trainer.progress_bar_dict["loss"],
                **trainer.progress_bar_metrics.get("gpu_stats", {}),
            )

    def set_postfix(self, pbar, prefix):
        l = len(prefix)
        postfix = {}
        for k, v in self.trainer.progress_bar_dict.items():
            if k.startswith(prefix):
                postfix[k[l:]] = self.format[k[l:]].format(v)
            elif not k.startswith("tr_") and not k.startswith("va_"):
                postfix[k] = v
        pbar.set_postfix(postfix, refresh=True)

    @staticmethod
    def fix_format_dict(
        pbar: tqdm, prefix: Optional[str] = None, timer: Optional[Timer] = None
    ) -> Dict:
        format_dict = pbar.format_dict
        if timer is not None:
            log.debug(
                f"{prefix} - lightning-elapsed={format_dict['elapsed']} elapsed={timer.value}"
            )
            format_dict["elapsed"] = timer.value
        # remove the square blocks, they provide no info
        format_dict["bar_format"] = (
            "{desc}: {percentage:.0f}% {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_noinv:.2f}it/s{postfix}]"
        )
        return format_dict

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        super().on_train_epoch_end(trainer, *args, **kwargs)
        if self.is_enabled:
            # add metrics to training bar
            self.set_postfix(self.main_progress_bar, "tr_")
            # override training time
            format_dict = self.fix_format_dict(
                self.main_progress_bar, "TR", self.tr_timer
            )
            # log training bar
            log.log(self.level, tqdm.format_meter(**format_dict))

            if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch:
                return
            # add metrics to training bar.
            # note: this is here instead of in `on_validation_epoch_end`
            # because `val_loop` gets called before `on_train_epoch_end` so the VA
            # bar would get printed before the TR bar. see:
            # https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#hook-lifecycle-pseudocode
            self.set_postfix(self.val_progress_bar, "va_")
            # override validation time
            format_dict = self.fix_format_dict(
                self.val_progress_bar, "VA", self.va_timer
            )
            # log validation bar
            log.log(self.level, tqdm.format_meter(**format_dict))

    def on_validation_batch_end(self, *args, **kwargs):
        # skip parent
        super(PLProgressBar, self).on_validation_batch_end(*args, **kwargs)
        if self._should_update(self.val_batch_idx, self.total_val_batches):
            self._update_bar(self.val_progress_bar)

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        super().on_validation_epoch_end(trainer, *args, **kwargs)
        if self.is_enabled:
            if trainer.running_sanity_check:
                self.val_progress_bar.refresh()
                log.log(
                    self.level,
                    tqdm.format_meter(**self.fix_format_dict(self.val_progress_bar)),
                )
            else:
                self.va_timer.stop()

    def on_validation_end(self, *args, **kwargs):
        # skip parent to avoid postfix call
        super(PLProgressBar, self).on_validation_end(*args, **kwargs)
        self.val_progress_bar.close()

    def on_test_end(self, *args, **kwargs):
        self.test_progress_bar.clear()
        super().on_test_end(*args, **kwargs)
