import re
import math

import torch.optim as optim
from transformers import PreTrainedModel

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs


def get_optimizer(
    model: PreTrainedModel, args: TrainingArgs
) -> tuple[optim.Optimizer, set[str]]:
    optim_name = args.optimizer
    trainable_param_names = set()
    trainable_params = []

    for name, param in model.named_parameters():
        if re.fullmatch(args.trainable_param_names, name):
            trainable_param_names.add(name)
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    for name in sorted(trainable_param_names):
        logger.info(f"Training Param: {name}")

    # TODO: Per-Parameter options の適用
    # 参考: https://pytorch.org/docs/stable/optim.html#per-parameter-options
    if optim_name.lower() == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    elif optim_name.lower() == "adam":
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(trainable_params, lr=args.learning_rate)
    else:
        raise ValueError(f"無効なoptimizer名です: {optim_name}")

    return optimizer, trainable_param_names


def get_optimizer_and_scheduler(
    model: PreTrainedModel, args: TrainingArgs, num_train_examples: int
):
    from lora_lightning.models.get_scheduler import get_scheduler
    from lora_lightning.models.utils import get_global_batch_size

    optimizer, trainable_param_names = get_optimizer(model, args)
    global_bs = get_global_batch_size(args.train_batch_size)

    if args.total_steps == -1:
        if args.num_train_epochs == -1:
            raise ValueError(
                "total_steps or num_train_epochs のどちらかは設定の必要があります"
            )
        args.total_steps = (
            math.ceil(num_train_examples / global_bs) * args.num_train_epochs
        )

    if args.warmup_steps == -1 or args.warmup_proportion > 0.0:
        logger.info(
            f"warmup_steps を warmup_proportion: {args.warmup_proportion} を元に計算します"
        )

        args.warmup_steps = int(args.warmup_proportion * args.total_steps)

    logger.info("Optimizerの設定")
    logger.info(f"Total Steps: {args.total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Schedular: {args.scheduler}")

    optimizer, trainable_param_names = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    return (optimizer, scheduler), trainable_param_names
