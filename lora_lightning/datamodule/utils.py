from torch.utils.data import Dataset, random_split

from transformers import AutoTokenizer

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs


def get_tokenizer(args: TrainingArgs, for_generation):
    return get_tokenizer_with_args(
        model_name=args.model,
        padding_side=args.padding_side,
        for_generation=for_generation,
    )


def get_tokenizer_with_args(
    model_name: str,
    padding_side: str = "right",
    for_generation: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

    logger.info(f"TokenizerのPadding Sideを {padding_side} に設定...")
    tokenizer.padding_side = padding_side

    if "llama-2" in model_name.lower():
        # pad_token を EOS -> UNK
        tokenizer.pad_token = tokenizer.unk_token

        if for_generation:
            if padding_side == "right":
                logger.warning(
                    "Padding side が 'right'に設定されていますが, 生成モデルを想定しています！"
                )

            logger.info(
                "for_generation が Trueのため、padding_side を 'left' に設定します"
            )
            tokenizer.padding_side = "left"
    # TODO: 他のモデルに対してもTokenizerの設定を行う
    # elif "pythia" in model_name.lower():
    else:
        raise ValueError(f"モデル名: {model_name} は現在対応していません。")

    return tokenizer


# TODO: testにも分割できるような柔軟な設計に
def split_dataset(dataset: Dataset, train_ratio: float):
    total_size = len(dataset)

    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset
