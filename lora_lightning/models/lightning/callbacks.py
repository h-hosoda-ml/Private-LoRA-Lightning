from lightning.pytorch.callbacks import Callback

from lora_lightning.logging import logger


class DataSizeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        logger.info("on_fit_startが呼ばれました")

        train_loader = trainer.train_dataloader
        if train_loader is not None:
            batch_size = train_loader.batch_size
            total_batches = len(train_loader)

            if train_loader.drop_last:
                # 不完全なバッチが切り捨てられている場合
                setattr(pl_module, "num_train_examples", total_batches * batch_size)
            else:
                last_batch_samples = len(train_loader.dataset) % batch_size
                if last_batch_samples == 0:
                    last_batch_samples = batch_size

                setattr(
                    pl_module,
                    "num_train_examples",
                    (total_batches - 1) * batch_size + last_batch_samples,
                )

            logger.info(f"Train Exampleの数: {pl_module.num_train_examples}")
        else:
            logger.info("train_dataloaderが見つかりませんでした")
