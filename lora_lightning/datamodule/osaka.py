import os
import json

from torch.utils.data import Dataset

from lora_lightning.logging import logger
from lora_lightning.datamodule.base import DataModule
from lora_lightning.datamodule.utils import split_dataset


class OsakaInstructionDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        return {
            "instruction": item["instruction"],
            "input": item["input"],
            "labels": item["output"],
        }


class OsakaDataModule(DataModule):
    def setup_dataset(self):
        if self.args.dataset is None:
            raise ValueError("データセットを明記してください")

        file_name = (
            self.args.dataset
            if self.args.dataset.endswith(".json")
            else self.args.dataset + ".json"
        )
        try:
            with open(os.path.join(self.args.data_dir, file_name)) as f:
                dataset_json: list[dict] = json.load(f)
        except Exception:
            logger.warning("データセットの読み込みに失敗しました")
            raise

        dataset = OsakaInstructionDataset(dataset_json)

        self.train_dataset, self.val_dataset = split_dataset(
            dataset, self.args.train_ratio
        )
