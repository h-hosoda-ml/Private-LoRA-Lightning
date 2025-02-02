import torch.nn as nn

from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.base import BaseModel

from lora_lightning.models.modifier.base import Modifier
from lora_lightning.models.modifier.lora import LoRA, QvkLoRA


class ExpertModel(BaseModel):
    def __init__(
        self, config: TrainingArgs, model_obj: nn.Module | None = None, **kwargs
    ):
        super().__init__(config, model_obj=model_obj, **kwargs)

        if config.model_modifier == "lora":
            LoRA.modify_transformer(self.model, config)
        elif config.model_modifier == "qvk_lora":
            QvkLoRA.modify_transformer(self.model, config)

    @property
    def modifiers(self):
        modifiers = []
        for _, module in self.model.named_modules():
            if isinstance(module, Modifier):
                modifiers.append(module)
        return modifiers
