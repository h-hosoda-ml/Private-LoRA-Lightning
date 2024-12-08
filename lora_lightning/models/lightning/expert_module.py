from transformers import PreTrainedModel

from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.expert_model import ExpertModel
from lora_lightning.models.lightning.base import (
    LightningEfficientCkpt,
    LigntningTrainingMixin,
)


class ExpertModule(LightningEfficientCkpt, LigntningTrainingMixin):
    def __init__(
        self, config: TrainingArgs, model_obj: PreTrainedModel | None = None, **kwargs
    ):
        super().__init__(model_obj, **kwargs)

        self.model = ExpertModel(config=config, model_obj=model_obj, **kwargs)
