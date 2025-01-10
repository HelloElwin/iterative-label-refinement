import os
import random
from typing import List

import fire
import torch
import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from iterative_label_refinement.core.logger import logger
from iterative_label_refinement.core.common import EVALUATOR_PROMPT_TEMPLATE
from iterative_label_refinement.core.datasets import load_dataset


def run_inference(evaluator_path: str, prompts: List[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b" if "2b" in evaluator_path else "mistralai/Mistral-7B-v0.1"
    )
    config = AutoConfig.from_pretrained(
        "google/gemma-2b" if "gemma-2b" in evaluator_path else "mistralai/Mistral-7B-v0.1"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        evaluator_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to("cuda")

    model.eval()
    all_outputs = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Running inference"):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            all_outputs.append(output.logits.cpu().float())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_probs = torch.nn.functional.softmax(torch.tensor(all_outputs), dim=1).numpy()

    return all_probs


def flatten_pairs(ds: datasets.Dataset) -> datasets.Dataset:
    res = []
    for x in ds:
        all_answers = x["answer"]
        # (x[0], x[1]), (x[2], x[3]), ...
        for i in range(0, len(all_answers), 2):
            res.append(
                {
                    "question": x["question"],
                    "answer": all_answers[i : i + 2],
                    "gt_answer": x["gt_answer"],
                    "acc": x["acc"][i : i + 2],
                }
            )
    res = datasets.Dataset.from_list(res)
    res = res.shuffle(seed=0)
    return res


def main(
    ds_name: str,
    weak_model_path: str,
    strong_model_path: str,
    evaluator_path: str,
    round_id: int,
    save_results: bool = True,
    subsample: float = 1.0,
    seed: int = 0,
    use_oracle_feedback: bool = False,
    start_with_correct_labels: bool = False,
    beta: float = 0.1,
) -> None:
    if f"ri={round_id}" not in strong_model_path:
        raise ValueError("Round mismatch.")

    _ = load_dataset(ds_name, split_sizes={"train": 20000, "test": 10000})  # for initialization

    # print setup
    logger.info("Starting to refine weak labels")
    logger.info(f" - evaluator path: {evaluator_path}")
    logger.info(f" - weak model path: {weak_model_path}")
    logger.info(f" - strong model path: {strong_model_path}")
    logger.info(f" - dataset: {ds_name}")
    logger.info(f" - round: {round_id}")
    logger.info(f" - save: {save_results}")
    logger.info(f" - subsample rate: {subsample}")
    logger.info(f" - use oracle feedback: {use_oracle_feedback}")
    logger.info(f" - start with correct labels: {start_with_correct_labels}")
    logger.info(f" - seed: {seed}")

    # setup random seeds
    random.seed(seed)
    np.random.seed(seed)

    method = "cor" * start_with_correct_labels + "ora" * use_oracle_feedback + "dpo" + str(beta) + str(subsample)

    # load pairs
    ds = datasets.load_from_disk(os.path.join(strong_model_path, "weak_labels"))
    ds = flatten_pairs(ds)

    ds_preference = []

    all_questions = [x["question"] for x in ds]
    all_a1 = [x["answer"][0] for x in ds]
    all_a2 = [x["answer"][1] for x in ds]

    prompts = []
    prompts_inv = []
    for i in range(len(ds)):
        prompts.append(EVALUATOR_PROMPT_TEMPLATE.format(q=all_questions[i], a=all_a1[i], b=all_a2[i]))
        prompts_inv.append(EVALUATOR_PROMPT_TEMPLATE.format(q=all_questions[i], a=all_a1[i], b=all_a2[i]))

    shuffle_idx = np.random.permutation(len(ds))
    ds_shuffled = [ds[int(i)] for i in shuffle_idx]
    prompts = [prompts[i] for i in shuffle_idx]
    prompts_inv = [prompts_inv[i] for i in shuffle_idx]
    all_prompts = prompts + prompts_inv

    if use_oracle_feedback:
        scores = [x["acc"][1] > x["acc"][0] for x in ds_shuffled]
    else:
        all_probs = run_inference(evaluator_path, all_prompts)
        probs, probs_inv = all_probs[: len(prompts)], all_probs[len(prompts) :]
        scores = [np.mean([x[1], y[0]]) for x, y in zip(probs, probs_inv)]  # prob of choosing a2
    ranked_eval_set = sorted(zip(ds_shuffled, scores), key=lambda x: abs(x[1] - 0.5), reverse=True)

    for idx, (x, score) in enumerate(ranked_eval_set):
        if idx > len(ds) * subsample:
            break

        q = x["question"]
        a1 = x["answer"][0]
        a2 = x["answer"][1]
        a_gt = x["gt_answer"]
        acc_a1 = x["acc"][0]
        acc_a2 = x["acc"][1]

        choose_a2 = bool(score > 0.5)  # don't want np.bool_
        ds_preference.append(
            {
                "question": q,
                "chosen": a2 if choose_a2 else a1,
                "rejected": a1 if choose_a2 else a2,
                "gt_answer": a_gt,
                "score": float(score),
                "acc": acc_a2 if choose_a2 else acc_a1,
            }
        )

    ds_preference = datasets.Dataset.from_list(ds_preference)
    ds_preference = ds_preference.shuffle(seed=seed)
    logger.info("Done generating preference dataset.")

    if save_results:
        ds_preference.save_to_disk(os.path.join(weak_model_path, f"{method}_{round_id}_weak_labels"))
        logger.info(f"Saved preference dataset to {weak_model_path}/{method}_{round_id}_weak_labels")


if __name__ == "__main__":
    fire.Fire(main)
