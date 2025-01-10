import os
import pickle
import random
from typing import List, Optional, Tuple

import fire
import datasets
from vllm import LLM

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.core.eval import eval_model_vllm
from iterative_label_refinement.core.common import (
    EVALUATOR_PROMPT_TEMPLATE,
    clear_mem,
    get_config_foldername,
)
from iterative_label_refinement.core.datasets import (
    DATASET_EVAL_FUNC,
    DATASET_MAX_LEN,
    load_dataset,
)

random.seed(0)


def load_pairs(all_results: List[datasets.Dataset], save_path: Optional[str] = None) -> datasets.Dataset:

    # give earlier answers more weight since they are probably more diverse
    def weighted_choice(answers):
        n = len(answers)
        weights = [2 ** (n - i - 1) for i in range(n)]
        return random.choices(answers, weights=weights, k=1)[0]

    # load all inference results
    if save_path is not None:
        all_paths = [os.path.join(save_path, p) for p in os.listdir(save_path) if p.startswith("checkpoint")]
        all_paths = sorted(all_paths, key=lambda x: int(x.split("checkpoint-")[-1]))
        all_results = [pickle.load(open(os.path.join(p, "inference_on_train.pkl"), "rb")) for p in all_paths]

    # for all question, gather answers into pos set and neg set
    ds_pool = {x["question"]: {"pos": [], "neg": [], "gt": x["gt_answer"]} for x in all_results[0]}
    for ds in all_results:
        for x in ds:
            q = x["question"]
            a = x["answer"]
            if x["acc"]:
                ds_pool[q]["pos"].append(a)
            else:
                ds_pool[q]["neg"].append(a)

    # generate pairs
    ds = []
    for question, answers in ds_pool.items():
        if len(answers["pos"]):
            ds.append({"question": question, "pos": answers["gt"], "neg": weighted_choice(answers["pos"])})
            if len(answers["neg"]):
                ds.append(
                    {
                        "question": question,
                        "pos": weighted_choice(answers["pos"] + [answers["gt"]]),
                        "neg": weighted_choice(answers["neg"]),
                    }
                )
        else:
            ds.append({"question": question, "pos": answers["gt"], "neg": weighted_choice(answers["neg"])})

    return ds


def load_pairs_saferpaca(
    all_results_general: List[datasets.Dataset],
    all_results_safety: List[datasets.Dataset],
    ds_train: datasets.Dataset,
    save_paths: Optional[List[str]] = None,
) -> datasets.Dataset:

    # give earlier answers more weight since they are probably more diverse
    def weighted_choice(answers):
        n = len(answers)
        weights = [2 ** (n - i - 1) for i in range(n)]
        return random.choices(answers, weights=weights, k=1)[0]

    # map question to category
    q2category = {q["question"]: q["category"] for q in ds_train}

    # load all inference results
    if save_paths is not None:
        all_paths_general = [os.path.join(save_paths[0], p) for p in os.listdir(save_paths) if p.startswith("checkpoint")]
        all_paths_general = sorted(all_paths_general, key=lambda x: int(x.split("checkpoint-")[-1]))
        all_results_general = [pickle.load(open(os.path.join(p, "inference_on_train.pkl"), "rb")) for p in all_paths_general]

        all_paths_safety = [os.path.join(save_paths[1], p) for p in os.listdir(save_paths) if p.startswith("checkpoint")]
        all_paths_safety = sorted(all_paths_safety, key=lambda x: int(x.split("checkpoint-")[-1]))
        all_results_safety = [pickle.load(open(os.path.join(p, "inference_on_train.pkl"), "rb")) for p in all_paths_safety]

    # for all question, gather answers into pos set and neg set
    ds_pool = {x["question"]: {"gt": x["gt_answer"], "general": [], "safety": []} for x in all_results_general[0]}
    for ds in all_results_general:
        for x in ds:
            ds_pool[x["question"]]["general"].append(x["answer"])
    for ds in all_results_safety:
        for x in ds:
            ds_pool[x["question"]]["safety"].append(x["answer"])

    # generate pairs
    ds = []
    ds_reject_safety = []
    accept_safety_cnt = 0
    for question, answers in ds_pool.items():
        if q2category[question] == "safety":
            accept_safety_cnt += 1
            ds.append(
                {
                    "question": question,
                    "pos": answers["gt"],
                    "neg": weighted_choice(answers["safety"]),
                }
            )
        else:
            ds.append(
                {
                    "question": question,
                    "pos": answers["gt"],
                    "neg": weighted_choice(answers["general"]),
                }
            )
            ds_reject_safety.append(
                {
                    "question": question,
                    "pos": weighted_choice(answers["general"]),
                    "neg": weighted_choice(answers["safety"]),
                }
            )

    ds += ds_reject_safety[:accept_safety_cnt]
    random.shuffle(ds)

    return ds


def pairs_to_datasets(ds: datasets.Dataset) -> Tuple[datasets.Dataset, datasets.Dataset]:
    ds_train = datasets.Dataset.from_list(ds[: int(len(ds) * 0.9)])
    ds_test = datasets.Dataset.from_list(ds[int(len(ds) * 0.9) :])

    ds_train_final, ds_test_final = [], []
    for _, x in enumerate(ds_train):
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(q=x["question"], a=x["pos"], b=x["neg"])
        ds_train_final.append({"question": prompt, "answer": "<reject>"})
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(q=x["question"], a=x["neg"], b=x["pos"])
        ds_train_final.append({"question": prompt, "answer": "<accept>"})

    for _, x in enumerate(ds_test):
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(q=x["question"], a=x["pos"], b=x["neg"])
        ds_test_final.append({"question": prompt, "answer": "<reject>"})
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(q=x["question"], a=x["neg"], b=x["pos"])
        ds_test_final.append({"question": prompt, "answer": "<accept>"})

    ds_train_final = datasets.Dataset.from_list(ds_train_final)
    ds_test_final = datasets.Dataset.from_list(ds_test_final)

    return ds_train_final, ds_test_final


def get_inference_results(
    ds: datasets.Dataset,
    max_len: int,
    eval_func: callable,
    save_path: str,
    model_size: str,
    temperature: float = 0.6,
    gpu_usage: float = 0.9,
    force_regenerate: bool = False,
) -> List[datasets.Dataset] | List[List[datasets.Dataset]]:
    all_results = []
    ckpt_dirs = [p for p in os.listdir(save_path) if "checkpoint" in p]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("checkpoint-")[-1]))
    for ckpt_dir in ckpt_dirs:
        logger.info(f"Running inference for {ckpt_dir}...")
        ckpt_path = os.path.join(save_path, ckpt_dir)

        if os.path.exists(os.path.join(ckpt_path, "inference_on_train.pkl")) and not force_regenerate:
            results = pickle.load(open(os.path.join(ckpt_path, "inference_on_train.pkl"), "rb"))
            all_results.append(results)
            logger.info(f"Using existing inference results for {ckpt_dir}")
            continue

        model = LLM(
            model=model_size,
            tokenizer=model_size,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=gpu_usage,
        )

        test_results = eval_model_vllm(
            model,
            ds,
            max_len,
            eval_func,
            temperature=temperature,
            lora_path=ckpt_path,
        )

        all_results.append(test_results)
        with open(os.path.join(ckpt_path, "inference_on_train.pkl"), "wb") as f:
            pickle.dump(test_results, f)

        del model
        clear_mem()

    return all_results


def main(
    save_path: str,
    model_size: str,
    ds_name: str,
    temperature: float = 0.6,
    seed: int = 0,
    n_docs: int = 20000,
    n_test_docs: int = 10000,
    gpu_usage: float = 0.9,
    do_eval: bool = True,
    force_regenerate: bool = False,
) -> None:
    ds_full = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))
    split_data = ds_full["train"].train_test_split(test_size=0.5, seed=seed, load_from_cache_file=False)
    ds_train = split_data["train"]

    max_len = DATASET_MAX_LEN[ds_name]
    eval_func = DATASET_EVAL_FUNC[ds_name]

    if not do_eval:
        eval_func = lambda q, x, y: -1.0

    if "saferpaca" in ds_name:
        save_paths = [save_path, save_path.replace("-sm=1-", "-sm=2-")]
        all_results_general = get_inference_results(
            ds_train,
            max_len,
            eval_func,
            save_paths[0],
            model_size,
            temperature=temperature,
            gpu_usage=gpu_usage,
            force_regenerate=force_regenerate,
        )
        all_results_safety = get_inference_results(
            ds_train,
            max_len,
            eval_func,
            save_paths[1],
            model_size,
            temperature=temperature,
            gpu_usage=gpu_usage,
            force_regenerate=force_regenerate,
        )
        print(all_results_general)
        print(all_results_safety)
        ds_pairs = load_pairs_saferpaca(all_results_general, all_results_safety, ds_train)
    else:
        all_results = get_inference_results(
            ds_train,
            max_len,
            eval_func,
            save_path,
            model_size,
            temperature=temperature,
            gpu_usage=gpu_usage,
            force_regenerate=force_regenerate,
        )
        ds_pairs = load_pairs(all_results)

    ds_evaluator_train, ds_evaluator_test = pairs_to_datasets(ds_pairs)
    ds_evaluator = datasets.DatasetDict({"train": ds_evaluator_train, "test": ds_evaluator_test})

    save_folder = os.path.dirname(os.path.dirname(save_path))
    ds_save_path = os.path.join(save_folder, "evaluator-data")
    if not os.path.exists(ds_save_path):
        os.makedirs(ds_save_path)
    ds_name = get_config_foldername({"dataset": ds_name, "model_size": model_size, "seed": seed})
    ds_evaluator.save_to_disk(os.path.join(ds_save_path, ds_name))

    logger.debug(f"Evaluator data is {ds_evaluator}")
    logger.info(f"Saved to {os.path.join(ds_save_path, ds_name)}")


if __name__ == "__main__":
    fire.Fire(main)
