import os
import ast
import json
import argparse
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    # model arguments
    model: str = None
    device_map: str = "cpu"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # output directories
    data_dir: str = os.getenv("TRAIN_DIR", "/tmp/")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")

    # Training Config
    scheduler: str = "linear_decay_with_warmup"
    checkpoint: str = None
    learning_rate: float = 1e-3
    warmup_proportion: float = 0.06
    train_batch_size: int = 32
    trainable_param_names: str = ".*"
    optimizer: str = "adamw"
    num_train_epochs: int = 5
    eval_every_n_epoch: int = 1
    seed: int = 42
    precision: str = "32"

    # LoRA Config
    lora_rank: int = 4
    lora_dropout: float = 0.05
    modify_layers: str = None

    @classmethod
    def parse(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config_files", required=False)
        args = parser.parse_args()

        extra_kwargs = {}

        if args.config_files:
            for filename in args.config_files.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(
                        os.getenv("CONFIG_PATH", default="configs"), filename
                    )

                if not os.path.exists(filename) and ".json" not in filename:
                    filename = filename + ".json"

                file_kwargs = json.load(open(filename))
                cls.process_kwargs(
                    file_kwargs,
                    eval=True,
                )
                extra_kwargs.update(file_kwargs)

        config = cls(**extra_kwargs)

        return config

    @classmethod
    def process_kwargs(cls, kwargs: dict, eval: bool = True) -> None:
        for k, v in kwargs.items():

            if not hasattr(cls, k):
                raise ValueError(f"{k} is not in the config")

            # python objectに変換
            # NOTE: evalよりも安全なast.literal_evalを採用
            if eval:
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    v = v
            else:
                v = v

            if type(v) == str and "$" in v:
                from string import Template

                v = Template(v).substitute(os.environ)

            kwargs[k] = v
