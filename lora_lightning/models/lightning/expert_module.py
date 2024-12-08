from transformers import PreTrainedModel

from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.expert_model import ExpertModel
from lora_lightning.models.lightning.base import (
    LightningEfficientCkpt,
    LigntningTrainingMixin,
)


class ExpertModule(LigntningTrainingMixin, LightningEfficientCkpt):
    def __init__(
        self, args: TrainingArgs, model_obj: PreTrainedModel | None = None, **kwargs
    ):
        LightningEfficientCkpt.__init__(self, model_obj, **vars(args))

        self.model = ExpertModel(config=args, model_obj=model_obj, **kwargs)
