import os

import torch
from lightning import LightningModule

from transformers import PreTrainedModel

from lora_lightning.logging import logger
from lora_lightning.models.utils import compute_loglike_loss


class LightningEfficientCkpt(LightningModule):
    def __init__(self, model_obj: PreTrainedModel | None = None, **kwargs):
        super().__init__()
        self.model_object = model_obj
        self.save_hyperparameters()

        # NOTE: Trueの場合、以前Loadしたcheckpointの情報を保持する
        self.save_if_loaded_from_ckpt = kwargs.get("save_if_loaded_from_ckpt", True)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **model_kwargs):
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        ckpt["hyper_parameters"].update(**model_kwargs)

        model = cls(**ckpt["hyper_parameters"])
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    def load_state_dict(self, state_dict: dict, **kwargs):
        # 読み込んだstate_dictの情報を格納
        self._params_from_checkpoint = (
            set(state_dict.keys) if self.save_if_loaded_from_ckpt else set()
        )
        for name, _ in self.state_dict().items():
            if name in state_dict:
                logger.info(f"StateDictから {name} の読み込みを行います")

        return super().load_state_dict(state_dict, strict=False)

    def on_save_checkpoint(self, checkpoint):
        self._delete_non_trainable_params(checkpoint["state_dict"])

    def on_load_checkpoint(self, checkpoint):
        print("Loading checkpoint...")

        load_result = self.load_state_dict(checkpoint["state_dict"])

        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def _delete_non_trainable_params(self, state_dict: dict):
        if not hasattr(self, "_params_from_checkpoint"):
            # NOTE: load_state_dictでcheckpointを読み込んだ場合に存在するattr
            self._params_from_checkpoint = set()

        if not hasattr(self, "trainable_param_names"):
            self.trainable_param_names = [
                name for name, param in self.named_parameters() if param.requires_grad
            ]

        deleted = []
        for key in state_dict.keys():
            if not (key in self.trainable_param_names) and not (
                key in self._params_from_checkpoint
            ):
                # 訓練可能ではない or checkpointからloadしていない場合
                del state_dict[key]
                deleted.append(key)

        logger.info(f"State Dictから {len(deleted)} 個のパラメータを削除")


class LigntningTrainingMixin:
    @property
    def inference_outputs(self):
        if not hasattr(self, "_inference_outputs"):
            self._inference_outputs = []
        return self._inference_outputs

    @property
    def best_val_result(self):
        if not hasattr(self, "_best_val_result"):
            self._best_val_result = None
        return self._best_val_result

    @property.setter
    def best_val_result(self, value):
        self._best_val_result = value

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss  # NOTE: Huggingface のモデルを使用することを前提

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/lr", self.optimizers()[0].param_groups[0]["lr"], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        # NOTE: Huggingface モデルでは 訓練時でない時に .lossが存在しない
        loss = compute_loglike_loss(outputs.logits, batch["labels"], reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self.inference_outputs.append(loss.detach())

        return mean_loss

    def on_validation_epoch_end(self) -> None:
        self.log_loss(split="val")
        self._inference_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        # NOTE: Huggingface モデルでは 訓練時でない時に .lossが存在しない
        loss = compute_loglike_loss(outputs.logits, batch["labels"], reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self.inference_outputs.append(loss.detach())

        return mean_loss

    def on_test_epoch_end(self) -> None:
        self.log_loss(split="test")
        self.inference_outputs.clear()

    def log_loss(self, split="val"):
        outputs = self.inference_outputs
        losses = torch.cat([out for out in outputs], dim=0)

        self.log(
            f"{split}/loss",
            losses.mean(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if split == "val":
            if self.best_val_result is None:
                self.best_val_result = losses.mean()
            else:
                # 改善した時のみ
                # NOTE: minimize方向を想定
                if losses.mean() < self.best_val_result:
                    self.best_val_result = losses.mean()
                    self.log(
                        f"{split}/best_loss",
                        losses.mean(),
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )

    def setup(self, stage):
        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            batch_size = train_loader.batch_size
            total_batches = len(train_loader)

            if train_loader.drop_last:
                # 不完全なバッチが切り捨てられている場合
                self.num_train_examples = total_batches * batch_size
            else:
                last_batch_samples = len(train_loader.dataset) % batch_size
                if last_batch_samples == 0:
                    last_batch_samples = batch_size

                self.num_train_examples = (
                    total_batches - 1
                ) * batch_size + last_batch_samples

            logger.info(f"Train Exampleの数: {self.num_train_examples}")

    def configure_optimizers(self):
        from lora_lightning.models.get_optimizer import get_optimizer_and_scheduler

        args = self.hparams

        (optimizer, schedular), self.trainable_param_names = (
            get_optimizer_and_scheduler(
                self, args, num_train_examples=self.num_train_examples
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": schedular,
                "interval": "step",  # NOTE: Stepごとにscheduler.step()
            },
        }