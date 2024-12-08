from typing import Type
from dataclasses import asdict

from transformers import set_seed

from lightning import Trainer

from lora_lightning.logging import logger, setup_logging
from lora_lightning.arguments import TrainingArgs
from lora_lightning.datamodule.base import get_datamodule
from lora_lightning.models.lightning.expert_module import ExpertModule
from lora_lightning.models.lightning.loggers import get_pl_loggers
from lora_lightning.models.lightning.callbacks import DataSizeCallback


def train_lora(args: TrainingArgs, model_class: Type[ExpertModule]):
    set_seed(args.seed)

    # Loggerの出力先を指定
    setup_logging(args.output_dir, reset_log=True)

    logger.info("Loggerの起動")
    logger.info(f"入力引数: {asdict(args)}")

    # Lightning Logger
    loggers = get_pl_loggers(args)

    # Lightning DataModule
    dm = get_datamodule(args)

    # Lightning Module
    module = model_class(args)

    trainer = Trainer(
        devices=1,
        accelerator=args.accelerator,
        logger=loggers,
        num_sanity_val_steps=0,
        max_epochs=args.num_train_epochs,
        max_steps=args.total_steps + 1 if args.total_steps != -1 else -1,
        strategy=args.compute_strategy if args.compute_strategy else "auto",
        precision=(
            int(args.precision) if args.precision in ["16", "32"] else args.precision
        ),
        log_every_n_steps=10,
    )

    trainer.fit(module, dm)


if __name__ == "__main__":
    train_lora(args=TrainingArgs.parse(), model_class=ExpertModule)
