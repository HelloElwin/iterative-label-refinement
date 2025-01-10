import glob
import json
import os
import random
from typing import Optional

import fire
import torch
import numpy as np
from dotenv import load_dotenv
from datasets import load_from_disk

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.core.models import MODELS_DICT
from iterative_label_refinement.core.train import train_and_save_model
from iterative_label_refinement.core.common import get_config_foldername, get_tokenizer
from iterative_label_refinement.core.datasets import (
    DATASET_EVAL_FUNC,
    DATASET_MAX_LEN,
    VALID_DATASETS,
    load_dataset,
    tokenize_dataset,
)


def main(
    seed: int = 0,
    data_seed: int = 0,
    epochs: int = 2,
    batch_size: int = 32,
    ds_name: str = "gsm8k",
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    model_size: str = "google/gemma-2b",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    force_retrain: bool = False,
    max_len: Optional[int] = None,
    train_with_dropout: bool = False,
    minibatch_size_per_device: int = 1,  # per device
    results_folder: Optional[str] = None,
    lr_schedule: str = "cosine_warmup",
    method: Optional[str] = None,
    weak_labels_path: Optional[str] = None,
    sweep_subfolder: str = "default",
    num_samples: int = 1,
    use_wandb: bool = False,
    save_frequency: float = 0,
    half_data: int = 0,  # 1 or 2: train with the first or second half of weak labels
    round_id: int = 0,
    save_model: bool = False,
    generate_weak_labels: bool = True,
    label_temp: float = 0.0,
    inference_on_train: bool = False,
    warmup_ratio: float = 0.05,
    force_eval: Optional[bool] = None,
    lora_rank: int = 64,
    gpu_usage: float = 0.9,
    train_with_all_data: bool = False,
    compute_weak_label_acc: Optional[bool] = None,
    compute_test_acc: Optional[bool] = None,
    do_dpo: bool = False,
    dpo_path: Optional[str] = None,  # path to the starting point of DPO
    dpo_beta: float = 0.1,
    dpo_loss_type: str = "sigmoid",
    saferpaca_mode: Optional[int] = None,  # 1: general only; 2: safety only.
) -> None:

    # set the folder for saving results
    if results_folder is None:
        load_dotenv()  # Load environment variables from the .env file
        results_folder = os.getenv("ILR_SAVE_PATH")
        if results_folder is None:
            raise ValueError("results_folder not provided and ILR_SAVE_PATH not set")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    logger.info(f"Saving to {os.path.join(results_folder, sweep_subfolder)}")

    if ds_name not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {ds_name} not in {VALID_DATASETS}")

    model_config = MODELS_DICT[model_size]

    if method is not None and "dpo" in method:
        do_dpo = True

    if lr is None:
        lr = model_config.default_dpo_lr if (do_dpo and round_id) else model_config.default_lr

    if optim is None:
        optim = model_config.default_dpo_optimizer if (do_dpo and round_id) else model_config.default_optimizer

    if max_len is None:
        max_len = DATASET_MAX_LEN[ds_name]

    if compute_weak_label_acc is None:
        compute_weak_label_acc = "paca" not in ds_name  # save budget
        logger.info(f"compute_weak_label_acc: {compute_weak_label_acc}")

    if compute_test_acc is None:
        # compute_test_acc = ds_name != "bird"  # BIRD evaluation during parallel training is slow and unstable
        compute_test_acc = True
        logger.info(f"compute_test_acc: {compute_test_acc}")

    if weak_labels_path is not None and not weak_labels_path.endswith("weak_labels"):
        if round_id:
            weak_labels_path = os.path.join(weak_labels_path, f"{method}_{round_id - 1}_weak_labels")
        else:
            weak_labels_path = os.path.join(weak_labels_path, "weak_labels")

    # only partial config is used in the folder name
    config = {
        "batch_size": batch_size,
        "max_len": max_len,
        "ds_name": ds_name,
        "n_docs": n_docs,
        "n_test_docs": n_test_docs,
        "model_size": model_size,
        "lr": lr,
        "optim": optim,
        "epochs": epochs,
        "seed": seed,
        "train_with_dropout": train_with_dropout,
        "lr_schedule": lr_schedule,
        "half_data": half_data,
        "round_id": round_id,
        "warmup_ratio": warmup_ratio,
        "lora_rank": lora_rank,
        "data_seed": data_seed,
    }

    if method is not None:
        if round_id == 0:
            config["weak_label_type"] = "cor" * ("cor" in method) + "weak"
        else:
            config["weak_label_type"] = method

    if do_dpo:
        config["do_dpo"] = True
        config["dpo_beta"] = dpo_beta
        config["dpo_loss_type"] = dpo_loss_type
        save_model = True

    if train_with_all_data:
        config["train_with_all_data"] = True

    if saferpaca_mode is not None:
        config["saferpaca_mode"] = saferpaca_mode

    if save_frequency:
        config["save_frequency"] = save_frequency

    eval_batch_size = model_config.eval_batch_size
    ds_eval_func = DATASET_EVAL_FUNC[ds_name]

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load and split dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))
    train_ds, test_ds = dataset["train"], dataset["test"]

    # set up the W2S and ILR data splits
    if train_with_all_data:
        train1_ds = train_ds
        train2_ds = None
        logger.info(f"Training with all data. len(train1): {len(train1_ds)}. No train2")
        config_name = get_config_foldername(config)
        config["weak_model_size"] = None
    elif weak_labels_path is None:
        split_data = train_ds.train_test_split(test_size=0.5, seed=seed)
        train1_ds, train2_ds = split_data["train"], split_data["test"]
        logger.info(f"len(train1): {len(train1_ds)} len(train2): {len(train2_ds)}")
        config_name = get_config_foldername(config)
        config["weak_model_size"] = None
    elif half_data:
        weak_labels = load_from_disk(weak_labels_path)

        split_data = weak_labels.train_test_split(test_size=0.5, seed=seed)
        if half_data == 1:
            train1_ds, train2_ds = split_data["train"], split_data["test"]
        elif half_data == 2:
            train1_ds, train2_ds = split_data["test"], split_data["train"]
        else:
            raise ValueError("half_data should be either 1 or 2")

        logger.info(f"Training half-data models. len(train1): {len(train1_ds)} len(train2): {len(train2_ds)}")

        weak_model_config = json.load(open(os.path.dirname(weak_labels_path) + "/experiment_config.json"))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config
    else:
        train1_ds = load_from_disk(weak_labels_path)
        train2_ds = None if not do_dpo else train_ds.train_test_split(test_size=0.5, seed=seed)["test"]
        logger.info(f"len(train1): {len(train1_ds)}" + "No train2" if train2_ds is None else f"len(train2): {len(train2_ds)}")

        weak_model_config = json.load(open(os.path.dirname(weak_labels_path) + "/experiment_config.json"))
        config["weak_model_size"] = weak_model_config["model_size"]
        config_name = get_config_foldername(config)
        config["weak_model"] = weak_model_config

    logger.info(f"config_name: {config_name}")
    save_path = os.path.join(results_folder, sweep_subfolder, config_name)

    if saferpaca_mode is not None:
        if saferpaca_mode == 1:
            train1_ds = train1_ds.filter(lambda x: x["category"] == "general")
        elif saferpaca_mode == 2:
            train1_ds = train1_ds.filter(lambda x: x["category"] == "safety")
        else:
            raise ValueError(f"Unknown saferpaca mode {saferpaca_mode}")
        logger.info(f"saferpaca_mode enabled: {saferpaca_mode}, len(train1): {len(train1_ds)}")

    if method is not None and "oradpo" in method and round_id:  # use comparisons with correct chosen answers
        train1_ds = train1_ds.filter(lambda x: x["acc"])
        logger.info(f"Filtering out preference pairs where the chosen answer is incorrect. len(train1): {len(train1_ds)}")
    if method is not None and method.startswith("cor") and round_id == 0:  # start with correct SFT data
        train1_ds = train1_ds.filter(lambda x: x["acc"] > 0)
        logger.info(f"Filtering out incorrect initial SFT data. len(train1): {len(train1_ds)}")

    # Eval or not
    if force_eval is None:
        force_eval = not half_data
    logger.info(f"force_eval: {force_eval}")

    # Print label accuracy
    try:
        acc = np.mean([x["acc"] for x in train1_ds])
        logger.info(f"train1 accuracy: {acc}")
    except Exception:
        pass

    # Filter out incorrectly formatted labels
    if ds_name == "gsm8k":
        logger.info("Filtering out incorrectly formatted labels")
        train1_ds = train1_ds.filter(lambda x: "####" in (x["chosen"] if do_dpo and round_id else x["answer"]))

    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train1_ds = tokenize_dataset(train1_ds, tokenizer, preference=do_dpo and round_id)
    test_ds = tokenize_dataset(test_ds, tokenizer)
    if train2_ds:
        train2_ds = tokenize_dataset(train2_ds, tokenizer)

    # Start training
    logger.info(f"Training model {model_size}")
    test_results, weak_ds = train_and_save_model(
        model_config=model_config,
        train_ds=train1_ds,
        test_ds=test_ds,
        ds_eval_func=ds_eval_func,
        inference_ds=train2_ds,
        batch_size=batch_size,
        save_path=save_path,
        lr=lr,
        epochs=epochs,
        max_len=max_len,
        force_retrain=force_retrain,
        eval_batch_size=eval_batch_size,
        minibatch_size_per_device=minibatch_size_per_device,
        train_with_dropout=train_with_dropout,
        lr_schedule=lr_schedule,
        optimizer_name=optim,
        save_frequency=save_frequency,
        generate_weak_labels=generate_weak_labels,
        num_samples=num_samples,
        label_temp=label_temp,
        inference_on_train=inference_on_train,
        warmup_ratio=warmup_ratio,
        force_eval=force_eval,
        use_wandb=use_wandb,
        lora_rank=lora_rank,
        seed=seed,
        gpu_usage=gpu_usage,
        data_seed=data_seed,
        compute_weak_label_acc=compute_weak_label_acc,
        compute_test_acc=compute_test_acc,
        do_dpo=do_dpo,
        round_id=round_id,
        tokenizer=tokenizer,
        dpo_path=dpo_path,
        dpo_loss_type=dpo_loss_type,
        dpo_beta=dpo_beta,
    )

    if weak_ds is not None:
        weak_ds.save_to_disk(save_path + "/" + "weak_labels")

    # final test accuracy
    if test_results:
        acc = np.mean([x["acc"] for x in test_results])
        res_dict = {"accuracy": acc}
        logger.info(f"Final test accuracy: {acc}")
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


if __name__ == "__main__":
    fire.Fire(main)
