from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ModelConfig:
    name: str
    default_lr: float
    eval_batch_size: int
    default_dpo_lr: Optional[float] = None
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adamw"
    default_dpo_optimizer: str = "rmsprop"
    vllm_tensor_parallel: int = 1


MODEL_CONFIGS = [
    ModelConfig(
        name="meta-llama/Meta-Llama-3-70B",
        default_lr=1e-4,
        default_dpo_lr=1e-6,
        eval_batch_size=1,
        model_parallel=True,
        vllm_tensor_parallel=2,
        gradient_checkpointing=True,
        custom_kwargs={
            "torch_dtype": torch.bfloat16,
        },
        default_optimizer="adafactor",
    ),
    ModelConfig(
        name="google/gemma-2b",
        default_lr=5e-4,
        default_dpo_lr=5e-6,  # actually not used
        eval_batch_size=1,
        model_parallel=True,
        gradient_checkpointing=True,
    ),
    ModelConfig(
        name="mistralai/Mistral-7B-v0.1",
        default_lr=1e-4,
        default_dpo_lr=1e-6,
        eval_batch_size=1,
        model_parallel=True,
        gradient_checkpointing=True,
    ),
]

MODELS_DICT: Dict[str, ModelConfig] = {model_config.name: model_config for model_config in MODEL_CONFIGS}
