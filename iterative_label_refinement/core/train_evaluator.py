import glob
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, Optional

import fire
import torch
import datasets
import evaluate
import numpy as np
from tqdm import tqdm
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from iterative_label_refinement.core.logger import logger, set_log_level
from iterative_label_refinement.core.common import clear_mem, get_tokenizer
from iterative_label_refinement.core.train import ModelConfig

MODEL_CONFIGS = [
    ModelConfig(
        name="google/gemma-2b",
        default_lr=1e-6,
        eval_batch_size=1,
        model_parallel=True,
        gradient_checkpointing=True,
    ),
    ModelConfig(
        name="mistralai/Mistral-7B-v0.1",
        default_lr=1e-6,
        eval_batch_size=1,
        model_parallel=True,
        gradient_checkpointing=True,
    ),
]
MODELS_DICT: Dict[str, ModelConfig] = {model_config.name: model_config for model_config in MODEL_CONFIGS}


def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    return "-".join(f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items()))


def eval(model: torch.nn.Module, ds: datasets.Dataset) -> list:
    model.eval()
    results = []
    with torch.no_grad():
        for x in tqdm(ds, desc="Evaluating"):
            sys.stdout.flush()
            input_ids = torch.tensor([x["input_ids"]]).to(model.device)
            atten_msk = torch.tensor([x["attention_mask"]]).to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=atten_msk)
            logits = outputs.logits
            logits = torch.squeeze(logits)
            label = torch.tensor(x["label"]).to(model.device)
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == label).float().mean().item()
            results.append({"acc": acc})
    return results


def train(
    model: torch.nn.Module,
    ds: datasets.Dataset,
    ds_test: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    log_every: int = 10,
    minibatch_size: int = 8,
    gradient_checkpointing: bool = False,
    epochs: int = 1,
    lr_schedule: str = "cosine_anneal",
    optimizer_name: str = "adam",
    save_path: str = None,
    warmup_ratio: float = 0.05,
    use_wandb: bool = True,
    seed: float = 0,
    save_full_model: bool = False,
    lora_rank: int = 0,
    log_level: str = "info",
) -> None:
    set_log_level(log_level)

    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"

    if optimizer_name.lower() == "adamw":
        optimizer = "adamw_torch"
    elif optimizer_name.lower() == "adafactor":
        optimizer = "adafactor"
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adamw or adafactor"

    training_args = TrainingArguments(
        seed=seed,
        bf16=True,
        optim=optimizer,
        learning_rate=lr,
        save_only_model=True,
        output_dir=save_path,
        logging_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        label_names=["labels"],
        logging_steps=log_every,
        num_train_epochs=epochs,
        overwrite_output_dir=True,
        report_to="wandb" if use_wandb else "none",
        per_device_train_batch_size=minibatch_size,
        per_device_eval_batch_size=minibatch_size,
        gradient_checkpointing=gradient_checkpointing,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps=batch_size // minibatch_size,
        warmup_ratio=warmup_ratio if "warmup" in lr_schedule else 0,
        lr_scheduler_type="cosine" if "cosine" in lr_schedule else lr_schedule,
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    if save_full_model and lora_rank:
        model = model.merge_and_unload()

    logger.info("Saving model...")
    model.save_pretrained(save_path)
    logger.info(f"Final model saved to: {save_path}")

    clear_mem()

    return model


def main(
    ds_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    model_size: str = "google/gemma-2b",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    minibatch_size_per_device: Optional[float] = None,
    results_folder: str = None,
    lr_schedule: str = "cosine_anneal_warmup",
    sweep_subfolder: str = "evaluators",
    use_wandb: bool = False,
    save_model: bool = True,
    warmup_ratio: float = 0.05,
    lora_rank: int = 0,
    save_full_model: bool = True,
    seed: int = 0,
) -> None:
    if results_folder is None:
        results_folder = os.environ.get("ILR_SAVE_PATH")
        if results_folder is None:
            raise ValueError("results_folder not provided and ILR_SAVE_PATH not set")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    logger.info(f"Saving to {os.path.join(results_folder, sweep_subfolder)}")

    if not lora_rank:
        save_full_model = True

    if lora_rank and not save_full_model:
        raise NotImplementedError("Saving adapter only is not supported. HuggingFace seems to have a bug with LoRA when training SequenceClassification models.")

    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    model_config = MODELS_DICT[model_size]

    if lr is None:
        lr = model_config.default_lr

    if optim is None:
        optim = model_config.default_optimizer

    config = {
        "batch_size": batch_size,
        "ds_name": ds_name,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        "seed": seed,
        "lr_schedule": lr_schedule,
        "warmup_ratio": warmup_ratio,
        "lora_rank": lora_rank,
        "save_full_model": save_full_model,
    }

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    if os.path.exists(ds_name):
        dataset = datasets.load_from_disk(ds_name)
    else:
        logger.info("Dataset not found locally, loading from HuggingFace...")
        dataset = datasets.load_dataset(ds_name)
    ds_train, ds_test = dataset["train"], dataset["test"]
    logger.info(f"len(ds_train): {len(ds_train)}, len(ds_test): {len(ds_test)}")

    config_name = get_config_foldername(config)
    logger.info(f"config_name: {config_name}")

    save_path = os.path.join(results_folder, sweep_subfolder, config_name)

    # Tokenize method
    tokenizer = get_tokenizer(model_config.name)

    def process_function(x):
        q, a = x["question"], x["answer"]
        inputs = tokenizer(q)
        inputs["label"] = 1 if "accept" in a else 0
        return inputs

    # Tokenize datasets
    ds_train = ds_train.map(process_function, batched=False)
    ds_test = ds_test.map(process_function, batched=False)

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def load_model(name, **kwargs):
        if os.path.exists(os.path.join(save_path, "model.safetensors.index.json")) or os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
            if lora_rank:
                model = AutoPeftModelForSequenceClassification.from_pretrained(save_path, device_map="auto")
            else:
                model = AutoModelForSequenceClassification.from_pretrained(save_path, device_map="auto", num_labels=2, **kwargs)
            logger.info(f"Loaded PEFT model from {save_path}")
            model = model.merge_and_unload()
            model.save_pretrained(save_path)
            return model, True
        model = AutoModelForSequenceClassification.from_pretrained(name, device_map="auto", num_labels=2, **kwargs)
        if lora_rank:
            peft_config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, target_modules="all-linear", task_type=TaskType.SEQ_CLS)  # , modules_to_save=['score.weight'])
            model.enable_input_require_grads()  # seems to be required for LoRA with gradient checkpointing
            model = get_peft_model(model, peft_config)
        return model, False

    already_trained = False
    model, already_trained = load_model(model_config.name, **custom_kwargs)
    minibatch_size = minibatch_size_per_device

    if not already_trained:
        model = train(
            model,
            ds_train,
            ds_test,
            batch_size,
            lr=lr,
            epochs=epochs,
            gradient_checkpointing=gradient_checkpointing,
            minibatch_size=minibatch_size,
            lr_schedule=lr_schedule,
            optimizer_name=optim,
            save_path=save_path,
            warmup_ratio=warmup_ratio,
            seed=seed,
            use_wandb=use_wandb,
            save_full_model=save_full_model,
            lora_rank=lora_rank,
        )

    logger.info("Final evaluation:")
    start = time.time()
    test_results = eval(model, ds_test)
    logger.info(f"Evaluation took {time.time() - start} seconds")

    # final test accuracy
    acc = np.mean([x["acc"] for x in test_results])
    res_dict = {"accuracy": acc}
    logger.info(f"Final test accuracy: {acc}")

    with open(os.path.join(save_path, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "avg_acc_test": float(np.mean([r["acc"] for r in test_results])),
                "test_results": test_results,
            },
            f,
        )

    with open(os.path.join(save_path, "results_summary.json"), "w") as f:
        json.dump(res_dict, f, indent=4)

    with open(os.path.join(save_path, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # delete model if not saving
    if not save_model:
        files_to_remove = glob.glob(os.path.join(save_path, "*safetensors*"))
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                logger.info(f"Removed {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")

    # delete intermediate checkpoints
    files_to_remove = glob.glob(os.path.join(save_path, "checkpoint-*"))
    for file_path in files_to_remove:
        files = os.listdir(file_path)
        for f in files:
            try:
                os.remove(os.path.join(file_path, f))
                logger.info(f"Removed {os.path.join(file_path, f)}")
            except Exception as e:
                logger.error(f"Failed to remove {os.path.join(file_path, f)}: {e}")
        try:
            os.rmdir(file_path)
            logger.info(f"Removed {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")


if __name__ == "__main__":
    fire.Fire(main)
