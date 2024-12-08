import torch
import torch.nn as nn

from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.utils import model_loader_helper


class BaseModel(nn.Module):
    def __init__(self, config: TrainingArgs, model_obj=None, **kwargs):
        super().__init__()

        self.load_in_4bit = config.get("load_in_4bit", False)
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.device_map = config.get("device_map", "cpu")
        self.precision = config.get("precision", "bf16")

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "`load_in_4bit` か `load_in_8bit` どちらか片方のみを指定してください"
            )

        self.model = (
            model_loader_helper(
                config.model,
                bf16=self.precision == "bf16",
                fp16=self.precision == "fp16",
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map,
            )
            if model_obj is None
            else model_obj
        )

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype
