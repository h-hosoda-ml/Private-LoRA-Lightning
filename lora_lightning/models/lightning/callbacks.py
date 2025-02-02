import os
import shutil
from typing import Literal

import torch
from lightning.pytorch.callbacks import Callback
from lightning import Trainer, LightningModule
from lightning.fabric.utilities.rank_zero import rank_zero_only

from lora_lightning.logging import logger


class LiveCheckpointCallback(Callback):
    def __init__(
        self,
        dir_path: str,
        monitor: str | None = None,
        mode: Literal["max", "min"] = "min",
        save_each_epoch: bool = False,
        save_weight_only: bool = True,
        save_last: bool = True,
    ):
        self.dir_path = dir_path
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best_model_path = None
        self.last_model_path = None
        self._last_step = -1
        self._last_value = None
        self.save_weight_only = save_weight_only
        self.save_each_epoch = save_each_epoch

    def _store_checkpoint(self, trainer: Trainer, ckpt_path: str):
        trainer.save_checkpoint(ckpt_path, weights_only=self.save_weight_only)

    def _save_best(self, trainer: Trainer, metric_value: torch.Tensor):
        if metric_value is None:
            raise ValueError("No value to save.. Something has gone wrong!")

        self._delete_best_path()
        monitor = self.monitor.replace("/", "-")

        self.best_model_path = os.path.join(
            self.dir_path,
            f"best_mode_{self.mode}_metric_{monitor}_value_{metric_value:.4f}_step_{self._last_step}.ckpt",
        )
        self._store_checkpoint(trainer, self.best_model_path)

    @rank_zero_only
    def _delete_best_path(self):
        if self.best_model_path and os.path.exists(self.best_model_path):
            if os.path.isdir(self.best_model_path):
                shutil.rmtree(self.best_model_path)
            else:
                os.remove(self.best_model_path)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.save_each_epoch:
            filename = f"epoch_{trainer.current_epoch}.ckpt"
            ckpt_path = os.path.join(self.dir_path, filename)
            self._store_checkpoint(trainer, ckpt_path)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.save_last:
            ckpt_path = os.path.join(self.dir_path, "last.ckpt")
            self._store_checkpoint(trainer, ckpt_path)

    def on_log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        metric_name: str,
        metric_value: torch.Tensor,
        **kwargs,
    ):
        if not self.monitor:
            return

        if metric_name != self.monitor:
            return

        if trainer.global_step == 0:
            return

        last_step = trainer.global_step

        if last_step == self._last_step:
            return

        old_value = metric_value.clone()
        sync_dist = kwargs.get("sync_dist", False)

        if sync_dist:
            is_dist_initialized = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            world_size = (
                torch.distributed.get_world_size() if is_dist_initialized else 1
            )
            if is_dist_initialized and world_size > 1:
                assert isinstance(
                    metric_value, torch.Tensor
                ), "sync_dist=True requires a scalar value"

                metric_value = metric_value.to(torch.float32)
                # NOTE: 全GPUの合計を計算
                torch.distributed.all_reduce(metric_value)
                metric_value = metric_value / world_size

        do_save = False
        self._last_step = last_step

        if self.mode == "min":
            do_save = self._last_value is None or metric_value < self._last_value
        else:
            do_save = self._last_value is None or metric_value > self._last_value

        if do_save:
            self._save_best(trainer, metric_value)
            self._last_value = metric_value
