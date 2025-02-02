import os
from typing import Type, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule

from transformers import AutoTokenizer

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
    tokenizer: AutoTokenizer
    padding: bool | str = True
    max_input_length: int = 1024
    max_output_length: int = 128
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "gpt"
    for_generation: bool = False
    task_to_id: dict | None = None
    add_eos_to_targets: bool = False

    def __call__(self, batch):
        sources = [
            (
                b["input"] + self.tokenizer.pad_token + b["instruction"]
                if b["input"]
                else b["instruction"]
            )
            for b in batch
        ]
        labels = [b["output"] for b in batch]

        output_batch = self.prepare_inputs_for_gpt_family(sources, labels)

        return output_batch

    def prepare_inputs_for_gpt_family(self, sources, labels):
        output_batch = {}
        # Add eos token
        sources, labels = self.add_space_and_eos(sources, labels)

        if self.for_generation:
            tokenized_sources = self.tokenizer(
                sources,
                max_length=self.max_input_length,
                add_special_tokens=False,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            tokenized_labels = self.tokenizer(
                labels,
                max_length=self.max_output_length,
                add_special_tokens=False,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
            output_batch["input_ids"] = tokenized_sources["input_ids"]
            output_batch["attention_mask"] = tokenized_sources["attention_mask"]
            output_batch["labels"] = tokenized_labels["input_ids"]
            return output_batch

        if self.max_input_length > 0:
            if self.tokenizer.truncation_side == "left":
                tokenized_labels = self.tokenizer(
                    labels,
                    max_length=self.max_input_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                )
            else:
                tokenized_sources = self.tokenizer(
                    sources,
                    max_length=self.max_input_length,
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    truncation=True,
                )

            tok_sources_plus_labels = self.tokenizer(
                [i + t for i, t in zip(sources, labels)],
                max_length=self.max_input_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
            )
        else:
            tokenized_sources = self.tokenizer(
                sources,
                padding="longest",
                return_tensors=self.return_tensors,
            )
            tok_sources_plus_labels = self.tokenizer(
                [i + t for i, t in zip(sources, labels)],
                padding="longest",
                return_tensors=self.return_tensors,
            )

        targets = tok_sources_plus_labels["input_ids"].clone()
        targets = torch.masked_fill(
            targets,
            ~tok_sources_plus_labels["attention_mask"].bool(),
            self.label_pad_token_id,
        )

        mask = torch.zeros(
            tok_sources_plus_labels["attention_mask"].shape[0],
            tok_sources_plus_labels["attention_mask"].shape[1] + 1,
        )

        # mask targets positions corresponding to the inputs
        if self.tokenizer.truncation_side == "left":
            labels_len = tokenized_labels["attention_mask"].int().sum(-1)
            pad_tokens = tok_sources_plus_labels["attention_mask"].shape[
                1
            ] - tok_sources_plus_labels["attention_mask"].int().sum(-1)

            if self.tokenizer.padding_side == "left":
                offset = -labels_len - 1
            else:
                offset = torch.clamp(
                    -pad_tokens - labels_len - 1, min=-self.max_input_length, max=0
                )
        else:
            input_len = tokenized_sources["attention_mask"].int().sum(-1)
            pad_tokens = tok_sources_plus_labels["attention_mask"].shape[
                1
            ] - tok_sources_plus_labels["attention_mask"].int().sum(-1)

            # handle right padding here!
            if self.tokenizer.padding_side == "left":
                offset = torch.clamp(pad_tokens + input_len, max=self.max_input_length)
            else:
                offset = input_len

        mask[(torch.arange(mask.shape[0]), offset)] = 1
        mask = mask.cumsum(dim=1).bool()
        mask = mask[:, :-1]
        targets = torch.masked_fill(targets, ~mask, self.label_pad_token_id)

        output_batch["input_ids"] = tok_sources_plus_labels["input_ids"]
        output_batch["attention_mask"] = tok_sources_plus_labels["attention_mask"]
        output_batch["labels"] = targets
        return output_batch

    def add_space_and_eos(self, sources, labels):
        import copy

        sources_ = copy.deepcopy(sources)
        labels_ = copy.deepcopy(labels)

        for i in range(len(sources_)):
            if self.tokenizer.custom_merges_space and sources_[i][-1] == " ":
                sources_[i] = sources_[i][:-1]
                labels_[i] = " " + labels_[i]

            if (
                sources_[i][-1] not in [" ", "\n"]
                and len(labels_[i]) > 0
                and labels_[i][0] not in [" ", "\n"]
            ):
                labels_[i] = " " + labels_[i]

        # adds the eos token
        labels_ = [
            l + ((" " + self.tokenizer.eos_token) if self.add_eos_to_targets else "")
            for l in labels_
        ]
        return sources_, labels_


class DataModule(LightningDataModule, ABC):
    collate_class: Type = DefaultCollator

    def __init__(self, args: TrainingArgs):
        super().__init__()

        self.args = args
        self.tokenizer = get_tokenizer(args)
        self.setup_dataset()

    @property
    def collate_fn(self) -> Callable:
        return self.collate_class(
            tokenizer=self.tokenizer, for_generation=self.args.for_generation
        )

    def train_dataloader(self) -> DataLoader:
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
