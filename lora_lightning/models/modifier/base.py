import re

import torch.nn as nn
from transformers import PreTrainedModel

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs


class Modifier(nn.Module):
    @classmethod
    def modify_transformer(cls, model_obj: PreTrainedModel, config: TrainingArgs):
        return modify_with_adapter(model_obj, config, cls)


def modify_with_adapter(
    model_obj: PreTrainedModel, config: TrainingArgs, adapter_klass
):
    for m_name, module in dict(model_obj.named_modules()).items():
        # モジュール名の一致
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                # レイヤー名の一致
                if re.fullmatch(config.modify_layers, c_name):
                    logger.info(f"Patching {m_name}.{c_name}...")

                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )

    return model_obj
