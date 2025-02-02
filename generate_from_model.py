from typing import Type
from dataclasses import asdict

import torch
from transformers import set_seed

from lora_lightning.logging import logger, setup_logging
from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.lightning.expert_module import ExpertModule
from lora_lightning.datamodule.base import get_datamodule


state_dict_file = "/workdir/repos/Private-LoRA-Lightning/output/last.ckpt"


def main(args: TrainingArgs, model_class: Type[ExpertModule]):
    set_seed(args.seed)

    setup_logging(args.output_dir, reset_log=True)

    logger.info("Loggerの起動")
    logger.info(f"入力引数: {asdict(args)}")

    dm = get_datamodule(args)
    module = model_class(args)

    state_dict = torch.load(state_dict_file)["state_dict"]
    module.load_state_dict(state_dict)

    if module.training:
        module.eval()

    with torch.no_grad():
        for batch in dm.train_dataloader():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            outputs = module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_return_sequences=1,
                repetition_penalty=1.2,
            )

            prompt = dm.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = dm.tokenizer.decode(outputs[0], skip_special_tokens=True)[
                len(prompt) :
            ]

            logger.info(f"Prompt: {prompt}\n\n")
            logger.info(f"Output: {generated_text}\n\n\n\n")


if __name__ == "__main__":
    main(args=TrainingArgs.parse(), model_class=ExpertModule)
