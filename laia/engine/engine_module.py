from typing import Any, Callable, Iterator, Optional, Tuple

import pytorch_lightning as pl
import torch

from laia.common.arguments import OptimizerArgs, SchedulerArgs
from laia.common.types import Loss as LossT
from laia.engine.engine_exception import exception_catcher
from laia.losses.loss import Loss
from laia.utils import check_tensor


class EngineModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: Optional[OptimizerArgs] = None,
        scheduler: Optional[SchedulerArgs] = None,
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.batch_input_fn = batch_input_fn
        self.batch_target_fn = batch_target_fn
        self.batch_id_fn = batch_id_fn
        self.batch_y_hat = None
        
        # Save optimizer and scheduler configs
        if optimizer is not None:
            self.lr = optimizer.learning_rate
        else:
            self.lr = None  # For inference only
            
        self.save_hyperparameters()

    def configure_optimizers(self):
        """Configure optimizers for training."""
        optimizer_name = self.hparams.optimizer.name.lower()
        learning_rate = self.hparams.optimizer.learning_rate

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate
            )
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=learning_rate
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        if not self.hparams.scheduler:
            return optimizer

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.scheduler.factor,
                patience=self.hparams.scheduler.patience,
            ),
            "monitor": self.hparams.scheduler.monitor,
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def prepare_batch(self, batch: Any) -> Tuple[Any, Any]:
        if self.batch_input_fn and self.batch_target_fn:
            return self.batch_input_fn(batch), self.batch_target_fn(batch)
        if isinstance(batch, tuple) and len(batch) == 2:
            x, y = batch
            # Ensure x is a tensor if it's a list
            if isinstance(x, list):
                x = torch.stack(x)
            return x, y
        return batch, None

    def check_tensor(self, batch: Any, batch_y_hat: Any) -> None:
        # note: only active when logging level <= DEBUG
        check_tensor(
            tensor=batch_y_hat.data,
            logger=__name__,
            msg=(
                "Found {abs_num} ({rel_num:.2%}) infinite values in the model output "
                "at epoch={epoch}, batch={batch}, global_step={global_step}"
            ),
            epoch=self.current_epoch,
            batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
            global_step=self.global_step,
        )

    def exception_catcher(self, batch: Any) -> Iterator[None]:
        return exception_catcher(
            self.batch_id_fn(batch) if self.batch_id_fn else batch,
            self.current_epoch,
            self.global_step,
        )

    def compute_loss(self, batch: Any, batch_y_hat: Any, batch_y: Any) -> LossT:
        with self.exception_catcher(batch):
            kwargs = {}
            if isinstance(self.criterion, Loss) and self.batch_id_fn:
                kwargs = {"batch_ids": self.batch_id_fn(batch)}
            batch_loss = self.criterion(batch_y_hat, batch_y, **kwargs)
            if batch_loss is not None:
                if not torch.isfinite(batch_loss).all():
                    raise ValueError("The loss is NaN or ± inf")
            return batch_loss

    def training_step(self, batch: Any, *_, **__):
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            batch_y_hat = self.model(batch_x)
        self.check_tensor(batch, batch_y_hat)
        batch_loss = self.compute_loss(batch, batch_y_hat, batch_y)
        if batch_loss is None:
            return
        self.log(
            "tr_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": batch_loss, "batch_y_hat": batch_y_hat}

    def validation_step(self, batch: Any, *_, **__):
        batch_x, batch_y = self.prepare_batch(batch)
        with self.exception_catcher(batch):
            batch_y_hat = self.model(batch_x)
        self.check_tensor(batch, batch_y_hat)
        batch_loss = self.compute_loss(batch, batch_y_hat, batch_y)
        if batch_loss is None:
            return
        self.log(
            "va_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": batch_loss, "batch_y_hat": batch_y_hat}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
