import os
from typing import Type, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from transformers import AutoTokenizer

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs
from lora_lightning.datamodule.utils import get_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_datamodule(args: TrainingArgs):
    from lora_lightning.datamodule.osaka import OsakaDataModule

    if "osaka" in args.dataset:
        dm = OsakaDataModule(args=args)
    else:
        raise ValueError(f"未対応のdataset名です: {args.dataset}")

    return dm


@dataclass
class DefaultCollator:
    tokenizer: AutoTokenizer = None
    padding: bool = "max_length"
    max_length: int = 128
    return_tensors: str = "pt"

    def __call__(self, batch):
        instructions = [b["instruction"] for b in batch]
        inputs = [b["input"] for b in batch]
        labels = [b["labels"] for b in batch]

        prompts = self.generate_prompt(instructions, inputs)
        output_batch = self.prepare_inputs(prompts, labels)

        return output_batch

    def generate_prompt(self, instructions: list[str], inputs: list[str]) -> list[str]:
        prompts = []

        for instruction, input in zip(instructions, inputs):
            prompts.append(
                f"[INST] {instruction}\n\n{input} [/INST]"
                if input
                else f"[INST] {instruction} [/INST]"
            )
        return prompts

    def prepare_inputs(self, prompts: list[str], labels: list[str]):
        output_batch = {}

        tokenized_prompts = self.tokenizer(
            prompts,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )
        tokenized_labels = self.tokenizer(
            labels,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )

        output_batch["input_ids"] = tokenized_prompts["input_ids"]
        output_batch["attention_mask"] = tokenized_prompts["attention_mask"]
        output_batch["labels"] = tokenized_labels["input_ids"]

        return output_batch


class DataModule(LightningDataModule, ABC):
    collate_class: Type = DefaultCollator

    def __init__(self, args: TrainingArgs, for_generation: bool = True):
        super().__init__()

        self.args = args
        self.tokenizer = get_tokenizer(args, for_generation=for_generation)
        self.setup_dataset()

    @property
    def collate_fn(self) -> Callable:
        return self.collate_class(tokenizer=self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        logger.info(f"Batch size: {self.args.train_batch_size}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collate_fn,
        )

    @abstractmethod
    def setup_dataset(self):
        pass
