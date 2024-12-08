import math

import torch
import torch.nn as nn

from lora_lightning.logging import logger
from lora_lightning.arguments import TrainingArgs
from lora_lightning.models.modifier.base import Modifier


class LoRA(Modifier):
    def __init__(self, config: TrainingArgs, layer: nn.Linear):
        super().__init__()
        self.config = config
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.dropout = config.lora_dropout
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.init_b_random = config.lora_init_b_random
        self.training_steps = 0.0
        self.scaling = self.alpha / self.rank
        self.forward_fn = None
        self.layer = layer

        if self.dropout > 0.0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            # 恒等変換
            self.dropout_layer = lambda x: x

        if hasattr(layer, "weight"):
            self.weight = layer.weight

        if hasattr(layer, "bias"):
            self.bias = layer.bias

        self.create_for_layer(layer)  # LoRAの作成
        self.reset_parameters()  # 初期化
        self.merged_with_layer = False  # LoRAを元行列にmergeしたか

    def create_for_layer(self, layer: nn.Linear):
        """LoRA行列を作成"""
        if isinstance(layer, nn.Linear):
            # LoRA行列
            self.lora_a = nn.Parameter(
                torch.empty(layer.in_features, self.rank).to(device=layer.weight.device)
            )
            self.lora_b = nn.Parameter(
                torch.empty(self.rank, layer.out_features).to(
                    device=layer.weight.device
                ),
            )
            # forward関数
            self.forward_fn = self.forward_linear_
        else:
            raise NotImplementedError("LoRA only supports nn.Linear layers.")

    def forward_linear_(self, input, **kwargs):
        """LoRAを適用させたForward関数"""
        output = self.layer(input)
        if self.merged_with_layer:
            return output
        else:
            input_lora = input.to(self.lora_a.dtype)
            input_lora = self.dropout_layer(input_lora)
            adapter_out = (
                torch.matmul(torch.matmul(input_lora, self.lora_a), self.lora_b)
                * self.scaling
            )
            return output + adapter_out.to(input.dtype)

    def reset_parameters(self):
        """パラメータの初期化処理 一様分布にしている。"""
        gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-std, std)

        # ensure that initially, adding the adapter does not change the output
        if self.init_b_random:
            logger.warning("LoRAの初期値が0ではない可能性があります")
            with torch.no_grad():
                self.lora_b.uniform_(-std, std)
        else:
            # NOTE: 推奨
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)
