import os
import json

import torch

import lightning.pytorch as L
from lightning.pytorch.loggers import Logger

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs


def get_pl_loggers(args: TrainingArgs) -> list[Logger]:
    loggers = []

    add_simple_logger(loggers, args)
    return loggers


def add_simple_logger(loggers: list, args: TrainingArgs):
    loggers.append(SimpleLogger(args.output_dir))


class SimpleLogger(Logger):
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "mertics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    @property
    def name(self):
        return "SimpleLogger"

    @property
    def version(self):
        return "0.1"

    def log_metrics(self, metrics, step):
        lines = []
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            lines.append({"name": k, "value": v, "step": step})

        try:
            with open(self.output_file, "a+") as f_out:
                for l in lines:
                    f_out.write(json.dumps(l) + "\n")
        except Exception as e:
            logger.error(f"Failed to log merics: {e}")

    def log_hyperparams(self, params):
        logger.info(f"Logging hyperparameters: {params}")
