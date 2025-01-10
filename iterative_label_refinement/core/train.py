import os
import time
import pickle
from typing import Callable, Optional, Tuple

import torch
import datasets
import numpy as np

from vllm import LLM
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.core.common import clear_mem
from iterative_label_refinement.core.models import ModelConfig
from iterative_label_refinement.core.eval import eval_model_vllm


def get_free_memory_ratio() -> float:
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    free_memory_ratio = free_memory / total_memory
    return free_memory_ratio


def wait_until_gpu_memory_free(threshold: float) -> None:
    threshold = threshold
    while True:
        free_memory_ratio = get_free_memory_ratio()
        if free_memory_ratio > threshold:
            logger.info(f"GPU memory free ratio is now {free_memory_ratio:.2f}. Proceeding.")
            break
        else:
            logger.info(f"GPU memory free ratio is {free_memory_ratio:.2f}. Waiting...")
            time.sleep(5)  # wait for 5 seconds before checking again


def train_model(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    log_every: int = 10,
    minibatch_size: int = 8,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_warmup",
    optimizer_name: str = "adamw",
    save_path: Optional[str] = None,
    save_frequency: float = 0,
    warmup_ratio: float = 0.05,
    use_wandb: bool = True,
    seed: int = 0,
    data_seed: int = 0,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"
    assert minibatch_size == 1, "minibatch size must be 1 for now"
    assert lr_schedule in ["cosine", "linear", "cosine_warmup"], "invalid lr schedule"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()

    if optimizer_name.lower() == "adamw":
        optimizer = "adamw_torch"
    elif optimizer_name.lower() == "adafactor":
        optimizer = "adafactor"
    elif optimizer_name.loswer() == "rmsprop":
        optimizer = "rmsprop"
    else:
        raise ValueError(f"invalid optimizer {optimizer_name}, must be adamw or adafactor or rmsprop")

    training_args = TrainingArguments(
        seed=seed,
        bf16=True,
        optim=optimizer,
        learning_rate=lr,
        data_seed=data_seed,
        save_only_model=True,
        output_dir=save_path,
        logging_dir=save_path,
        logging_steps=log_every,
        num_train_epochs=epochs,
        save_steps=save_frequency,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
        per_device_train_batch_size=minibatch_size,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy="no" if save_frequency == 0 else "steps",
        gradient_accumulation_steps=batch_size // minibatch_size,
        warmup_ratio=warmup_ratio if "warmup" in lr_schedule else 0,
        lr_scheduler_type="cosine" if "cosine" in lr_schedule else lr_schedule,
    )

    def data_collate_func(x):
        """Only for minibatch_size=1"""
        input_ids = torch.tensor([x[0]["input_ids"]])
        attention_mask = torch.tensor([x[0]["attention_mask"]])
        labels = torch.tensor([x[0]["labels"]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collate_func,
    )

    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(save_path)
    logger.info(f"Final model saved to: {save_path}")


def train_model_dpo(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    log_every: int = 10,
    minibatch_size: int = 8,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_warmup",
    optimizer_name: str = "adamw",
    save_path: Optional[str] = None,
    save_frequency: float = 0,
    warmup_ratio: float = 0.05,
    use_wandb: bool = True,
    seed: int = 0,
    data_seed: int = 0,
    tokenizer: Optional[AutoTokenizer] = None,
    dpo_loss_type: str = "sigmoid",
    dpo_beta: float = 0.1,
) -> None:
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"
    assert minibatch_size == 1, "minibatch size must be 1 for now"
    assert lr_schedule in ["cosine", "linear", "cosine_warmup"], "invalid lr schedule"

    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()

    if optimizer_name.lower() == "adamw":
        optimizer = "adamw_torch"
    elif optimizer_name.lower() == "adafactor":
        optimizer = "adafactor"
    elif optimizer_name.lower() == "rmsprop":
        optimizer = "rmsprop"
    else:
        raise ValueError(f"invalid optimizer {optimizer_name}, must be adamw or adafactor or rmsprop")

    training_args = DPOConfig(
        beta=dpo_beta,
        seed=seed,
        bf16=True,
        optim=optimizer,
        learning_rate=lr,
        data_seed=data_seed,
        save_only_model=True,
        output_dir=save_path,
        logging_dir=save_path,
        logging_steps=log_every,
        num_train_epochs=epochs,
        save_steps=save_frequency,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
        per_device_train_batch_size=minibatch_size,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy="no" if save_frequency == 0 else "steps",
        gradient_accumulation_steps=batch_size // minibatch_size,
        warmup_ratio=warmup_ratio if "warmup" in lr_schedule else 0,
        lr_scheduler_type="cosine" if "cosine" in lr_schedule else lr_schedule,
        model_adapter_name="train_adapter",
        ref_adapter_name="ref_adapter",
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=2048,
        max_target_length=2048,
        loss_type=dpo_loss_type,
    )

    def data_collate_func(x):
        """Bypassing the default data collator to handle the custom data format."""
        res = {}
        for k in x[0]:
            if k.endswith("_input_ids_"):
                res[k[:-1]] = torch.tensor([x[0][k]])
            elif k.endswith("_attention_mask_"):
                res[k[:-1]] = torch.tensor([x[0][k]])
            elif k.endswith("_logps"):
                res[k] = torch.tensor([x[0][k]])
            elif k.endswith("_labels_"):
                res[k[:-1]] = torch.tensor([x[0][k]], dtype=torch.long)
        return res

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=data_collate_func,
    )

    trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(save_path)
    logger.info(f"Final model saved to: {save_path}")


def train_and_save_model(
    *,
    model_config: ModelConfig,
    train_ds: datasets.Dataset,
    test_ds: datasets.Dataset,
    ds_eval_func: Callable,
    inference_ds: Optional[datasets.Dataset] = None,
    batch_size: int,
    lr: float,
    epochs: int,
    max_len: int,
    eval_batch_size: Optional[int] = None,
    minibatch_size_per_device: Optional[int] = None,
    save_path: Optional[str] = None,
    force_retrain: bool = False,
    train_with_dropout: bool = False,
    lr_schedule: str = "constant",
    optimizer_name: str = "adamw",
    save_frequency: float = 0,
    generate_weak_labels: bool = True,
    num_samples: int = 1,
    label_temp: float = 0.0,
    inference_on_train: bool = False,
    warmup_ratio: float = 0.05,
    force_eval: bool = False,
    use_wandb: bool = True,
    seed: int = 0,
    lora_rank: Optional[int] = None,
    gpu_usage: float = 0.9,
    data_seed: int = 0,
    compute_weak_label_acc: bool = True,
    compute_test_acc: bool = True,
    do_dpo: bool = False,
    round_id: int = 0,
    tokenizer: Optional[AutoTokenizer] = None,
    dpo_path: Optional[str] = None,
    dpo_loss_type: str = "sigmoid",
    dpo_beta: float = 0.1,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(name: str, load_from_last_round: bool = False, **kwargs) -> Tuple[torch.nn.Module, bool]:
        if (  # checks possible model save paths
            os.path.exists(os.path.join(save_path, "model.safetensors.index.json"))
            or os.path.exists(os.path.join(save_path, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(save_path, "train_adapter", "adapter_model.safetensors"))
        ) and not force_retrain:
            logger.info(f"already trained, load with vLLM for inference... {save_path}")
            return None, True
        config = AutoConfig.from_pretrained(name, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(name, config=config, device_map="auto", **kwargs)
        if load_from_last_round:
            assert lora_rank is not None, "currently only support LoRA for DPO"
            assert dpo_path is not None, "dpo_path must be provided"
            logger.info(f"Loading DPO starting point from {dpo_path}")
            model = PeftModel.from_pretrained(
                model,
                dpo_path,
                is_trainable=True,
                adapter_name="train_adapter",
            )
            model.load_adapter(dpo_path, adapter_name="ref_adapter")  # DPO's reference model
        elif lora_rank is not None:
            peft_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear")
            model.enable_input_require_grads()  # seems to be required for LoRA with gradient checkpointing
            model = get_peft_model(model, peft_config)
        return model, False

    already_trained = False
    model, already_trained = maybe_load_model(model_config.name, load_from_last_round=do_dpo and round_id, **custom_kwargs)
    minibatch_size = minibatch_size_per_device

    test_results = None
    if not already_trained:
        start = time.time()
        training_params = dict(
            model=model,
            ds=train_ds,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            gradient_checkpointing=gradient_checkpointing,
            minibatch_size=minibatch_size,
            train_with_dropout=train_with_dropout,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            save_path=save_path,
            save_frequency=save_frequency,
            warmup_ratio=warmup_ratio,
            use_wandb=use_wandb,
            seed=seed,
            data_seed=data_seed,
            tokenizer=tokenizer,
        )
        if do_dpo and round_id:
            training_params["dpo_loss_type"] = dpo_loss_type
            training_params["dpo_beta"] = dpo_beta
            train_model_dpo(**training_params)
        else:
            train_model(**training_params)
        logger.info(f"Model training took {time.time() - start} seconds")

        model.to("cpu")
        del model
        clear_mem()

    # wait until other processes are done
    wait_until_gpu_memory_free(gpu_usage)

    # reload the model with vLLM
    if lora_rank is not None:
        model = LLM(
            model=model_config.name,
            tokenizer=model_config.name,
            dtype="bfloat16",
            tensor_parallel_size=model_config.vllm_tensor_parallel,
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=gpu_usage,
            max_model_len=4096,
        )
    else:
        model = LLM(
            model=save_path,
            tokenizer=model_config.name,
            dtype="bfloat16",
            tensor_parallel_size=model_config.vllm_tensor_parallel,
            gpu_memory_utilization=gpu_usage,
            max_model_len=4096,
        )

    lora_path = save_path if lora_rank is not None else None
    if do_dpo and round_id:
        lora_path = os.path.join(save_path, "train_adapter")

    if force_eval:
        logger.info("Evaluation started:")
        start = time.time()
        if not compute_test_acc:
            ds_eval_func = lambda q, x, y: -1.0
        test_results = eval_model_vllm(
            model,
            test_ds,
            max_len,
            ds_eval_func,
            eval_batch_size,
            num_samples=num_samples,
            temperature=0,  # use consistent testing temperature
            lora_path=lora_path,
        )
        logger.info(f"Evaluation took {time.time() - start} seconds")

    inference_results = None
    if inference_ds and generate_weak_labels:
        if not compute_weak_label_acc:
            ds_eval_func = lambda q, x, y: -1.0
        inference_results = eval_model_vllm(
            model,
            inference_ds,
            max_len,
            ds_eval_func,
            eval_batch_size,
            temperature=label_temp,
            lora_path=lora_path,
            num_samples=6 if do_dpo else 1,
        )

    train_inference_results = None
    if inference_on_train:
        train_inference_results = eval_model_vllm(
            model,
            train_ds,
            max_len,
            ds_eval_func,
            eval_batch_size,
            temperature=label_temp,
            lora_path=lora_path,
        )

    with open(os.path.join(save_path, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "avg_acc_test": float(np.mean([r["acc"] for r in test_results])) if test_results else None,
                "avg_acc_inference": float(np.mean([r["acc"] for r in inference_results] if inference_results else [])),
                "test_results": test_results if test_results else [],
                "inference_results": inference_results if inference_results else [],
                "train_inference_results": train_inference_results if train_inference_results else [],
            },
            f,
        )

    clear_mem()

    return test_results, inference_results
