import torch
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutput

from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.utils import model_loader_helper


class BaseModel(nn.Module):
    def __init__(self, config: TrainingArgs, model_obj=None, **kwargs):
        super().__init__()

        self.load_in_4bit = config.load_in_4bit or False
        self.load_in_8bit = config.load_in_8bit or False
        self.device_map = config.device_map or "cpu"
        self.precision = config.precision or "bf16"

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

    def forward(
        self, input_ids, attention_mask=None, labels=None, **kwargs
    ) -> CausalLMOutput:
        outputs = self.model.forward(
            input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )
        return outputs
