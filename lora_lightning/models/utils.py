import torch
import torch.distributed
import torch.nn as nn

from lora_lightning.logging import logger


def model_loader_helper(
    model_name,
    device_map="auto",
    bf16=True,
    fp16=False,
    load_in_4bit=False,
    load_in_8bit=False,
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Specify either 'load_in_4bit' or 'load_in_8bit' or neither.")

    from transformers import (
        PreTrainedModel,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # set dtype
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model_name, PreTrainedModel):
        return model_name.train()  # 訓練モードへ

    if "pythia" in model_name or "Llama" in model_name:
        model_object: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"Model name {model_name} is not available")

    logger.info("Freezing parameters of the base model by default.")

    for _, param in model_object.named_parameters():
        param.requires_grad = False

    return model_object.train()


def get_global_batch_size(
    batch_size: int,
):
    """Computes the global batch size."""
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    global_bs = batch_size * world_size

    return global_bs


def compute_loglike_loss(logits: torch.Tensor, labels: torch.Tensor, reduction="none"):
    batch_size = logits.size(0)
    vocab_size = logits.size(-1)
    # labels = labels.squeeze(-1)  # NOTE: labelsを(batch_size, seq_length)へ

    # 生成モデルを想定しており、特殊トークンの部分を除く
    shift_logits = logits[:, :-1, :].contiguous()  # 最後を除く
    shift_labels = labels[:, 1:].contiguous()  # 最初を除く

    loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fn(shift_logits, shift_labels)

    if reduction == "none":
        loss = loss.view((batch_size, -1))
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss

    return loss
