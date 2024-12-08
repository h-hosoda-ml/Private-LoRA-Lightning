import math

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from lora_lightning.arguments import TrainingArgs


def get_scheduler(optimizer: optim.Optimizer, args: TrainingArgs):
    scheduler_name = args.scheduler

    if scheduler_name == "linear_and_cos_with_warmup":
        return get_linear_and_cos_schedule_with_warmup(
            optimizer, args.warmup_steps, args.total_steps
        )
    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, args.total_steps)
    else:
        raise ValueError(f"無効なscheduler名です: {scheduler_name}")


def get_linear_and_cos_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine annealing
            prog = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
