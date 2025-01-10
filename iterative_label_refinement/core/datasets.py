import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import datasets
from dotenv import load_dotenv

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.metrics.eval_bird import bird_eval_func
from iterative_label_refinement.metrics.eval_gsm8k import gsm8k_eval_func
from iterative_label_refinement.metrics.eval_saferpaca import saferpaca_eval_func


@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], datasets.Dataset]
    # formats items to have keys 'question' and 'answer'
    formatter: Callable[[Any], Any]


# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}

# max token length of eahc dataset
DATASET_MAX_LEN: dict[str, int] = {}

# evaluation function for different datasets
DATASET_EVAL_FUNC: dict[str, int] = {}

# get env variable for BIRD data path
load_dotenv()
BIRD_DATA_PATH = os.getenv("ILR_BIRD_PATH")
if BIRD_DATA_PATH is None:
    logger.warning("Please set ILR_BIRD_PATH in .env if you are using the BIRD dataset.")


def register_dataset(name: str, max_len: int, eval_func: Callable, config: DatasetConfig) -> None:
    _REGISTRY[name] = config
    DATASET_MAX_LEN[name] = max_len
    DATASET_EVAL_FUNC[name] = eval_func


def load_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None) -> dict[str, datasets.Dataset]:
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name]
    results = {}
    for split, n_docs in split_sizes.items():
        ds = cfg.loader(split)
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            logger.warning(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(cfg.formatter, load_from_cache_file=False)
        ds = ds.shuffle(seed=seed, load_from_cache_file=False)
        results[split] = ds

    return results


def tokenize_dataset(
    raw_ds: datasets.Dataset,
    tokenizer: Callable,
    preference: bool = False,
) -> datasets.Dataset:
    """
    This function prepares the dataset for training.

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    preference: Whether to return the dataset in chosen-reject format for preference optimization.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(x: dict) -> dict:
        """
        For the preference dataset, we add _ to the keys to avoid conflicts with DPOTrainer's data collator.
        We do this to bypass the default tokenization data collation and use our own data format.
        """
        if preference:
            prompt, chosen, rejected = x["question"], x["chosen"], x["rejected"]

            # Tokenize the prompt
            prompt_encoded = tokenizer(f"USER:\n{prompt}\n\nASSISTANT:\n", add_special_tokens=False)
            prompt_input_ids = prompt_encoded["input_ids"]
            prompt_attention_mask = prompt_encoded["attention_mask"]

            # Tokenize the chosen answer
            chosen_encoded = tokenizer(f"{chosen}{tokenizer.eos_token}", add_special_tokens=False)
            chosen_input_ids = prompt_input_ids + chosen_encoded["input_ids"]
            chosen_attention_mask = prompt_attention_mask + chosen_encoded["attention_mask"]
            chosen_labels = [-100] * len(prompt_input_ids) + chosen_encoded["input_ids"]

            # Tokenize the rejected answer
            rejected_encoded = tokenizer(f"{rejected}{tokenizer.eos_token}", add_special_tokens=False)
            rejected_input_ids = prompt_input_ids + rejected_encoded["input_ids"]
            rejected_attention_mask = prompt_attention_mask + rejected_encoded["attention_mask"]
            rejected_labels = [-100] * len(prompt_input_ids) + rejected_encoded["input_ids"]

            # Gather the inputs
            inputs = {
                "prompt": f"USER:\n{prompt}\n\nASSISTANT:\n",
                "prompt_input_ids_": prompt_input_ids,
                "prompt_attention_mask_": prompt_attention_mask,
                "chosen_input_ids_": chosen_input_ids,
                "chosen_attention_mask_": chosen_attention_mask,
                "chosen_labels_": chosen_labels,
                "rejected_input_ids_": rejected_input_ids,
                "rejected_attention_mask_": rejected_attention_mask,
                "rejected_labels_": rejected_labels,
            }
        else:
            q, a = x["question"], x["answer"]

            # Need to encode the question and answer separately, otherwise label positions could be wrong
            question_encoded = tokenizer(f"USER:\n{q}\n\nASSISTANT:\n", add_special_tokens=False)
            answer_encoded = tokenizer(f"{a}{tokenizer.eos_token}", add_special_tokens=False)

            # Combine the question and answer encodings
            input_ids = question_encoded["input_ids"] + answer_encoded["input_ids"]
            attention_mask = question_encoded["attention_mask"] + answer_encoded["attention_mask"]

            # Create labels, masking the question part with -100
            labels = [-100] * len(question_encoded["input_ids"]) + answer_encoded["input_ids"]

            # Gather the inputs
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        return inputs

    ds = raw_ds.map(process_function, batched=False)

    logger.debug(f"Tokenized dataset: {ds}")
    logger.debug(f"First example: {ds[0]}")

    return ds


def hf_local_loader(hf_name, split_names=None, **kwargs) -> None:
    if split_names is None:
        split_names = dict()
    return lambda split: datasets.load_from_disk(hf_name, **kwargs)[split_names.get(split, split)]


########################
# Dataset Registration #
########################


def format_gsm8k(x):
    return x


def eval_gsm8k(q, pred, gt):
    return gsm8k_eval_func(q, pred, gt)


register_dataset(
    name="gsm8k",
    max_len=256,
    eval_func=eval_gsm8k,
    config=DatasetConfig(loader=hf_local_loader("data/gsm8k"), formatter=format_gsm8k),
)


def format_bird(x):
    global bird_q2db
    try:
        bird_q2db[x["question"]] = x["db_name"]
    except Exception:
        bird_q2db = {x["question"]: x["db_name"]}
    return x


def eval_bird(questions, pred_sqls, gt_sqls, meta_time_out=30):
    """
    BIRD only accepts batch evaluation.
    """
    if not isinstance(questions, list):
        raise ValueError("BIRD only accepts batch evaluation.")
    global bird_q2db
    db_names = [bird_q2db[q] for q in questions]
    return bird_eval_func(pred_sqls, gt_sqls, db_names, db_root_path=BIRD_DATA_PATH, meta_time_out=meta_time_out)


register_dataset(
    name="bird",
    max_len=256,
    eval_func=eval_bird,
    config=DatasetConfig(loader=hf_local_loader("data/bird"), formatter=format_bird),
)


def format_saferpaca(x):
    return x


def eval_saferpaca(q, pred, gt):
    return float(saferpaca_eval_func(q, pred, gt))


register_dataset(
    name="saferpaca",
    max_len=512,
    eval_func=eval_saferpaca,
    config=DatasetConfig(loader=hf_local_loader("data/saferpaca"), formatter=format_saferpaca),
)


VALID_DATASETS: list[str] = list(_REGISTRY.keys())
