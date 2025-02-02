from torch.utils.data import Dataset, random_split

from transformers import AutoTokenizer

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs


def tokenizer_merges_space(tokenizer):
    test1 = "this"
    test2 = " this"

    return len(tokenizer(test1)["input_ids"]) == len(tokenizer(test2)["input_ids"])


def get_tokenizer(args: TrainingArgs):
    return get_tokenizer_with_args(
        model_name=args.model,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        for_generation=args.for_generation,
    )


def get_tokenizer_with_args(
    model_name: str,
    padding_side: str = "right",
    truncation_side="right",
    for_generation: bool = False,
):
    if "pythia" or "gpt-neox" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
    elif "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
        tokenizer.pad_token_id = 0
    else:
        raise ValueError(f"モデル名: {model_name} は現在対応していません。")

    tokenizer.padding_side = padding_side
    logger.info(f"TokenizerのPadding Sideを {padding_side} に設定...")

    tokenizer.truncation_side = truncation_side
    logger.info(f"TokenizerのTruncation Sideを {truncation_side} に設定...")

    if for_generation:
        if padding_side == "right":
            logger.warning(
                "Padding sideが'right'ですが、generation mode が Trueになっています!"
            )

        logger.info("for_generation が Trueのため、padding_side を 'left' に設定します")
        tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        logger.info("pad_token_idが存在しないため、eos_token_idに設定します。")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.custom_merges_space = tokenizer_merges_space(tokenizer)

    return tokenizer


# TODO: testにも分割できるような柔軟な設計に
def split_dataset(dataset: Dataset, train_ratio: float):
    total_size = len(dataset)

    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset
